import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

import medmnist
from medmnist import INFO


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(image_size):
    weights = EfficientNet_B0_Weights.DEFAULT
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


def make_dataset(split, transform, data_root, size):
    return medmnist.DermaMNIST(
        split=split,
        transform=transform,
        size=size,
        download=True,
        as_rgb=True,
        root=str(data_root),
    )


def get_targets(dataset):
    return np.asarray(dataset.labels).reshape(-1).astype(int)


def make_class_weights(targets, num_classes):
    counts = np.bincount(
        targets,
        minlength=num_classes,
    ).astype(np.float64)

    weights = counts.sum() / (
        num_classes * np.maximum(counts, 1)
    )

    return (
        torch.tensor(weights, dtype=torch.float32),
        counts.astype(int),
    )


def build_model(num_classes):
    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

    return (
        total_loss / total,
        correct / total,
        y_true,
        y_pred,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    project_root = Path(
        __file__
    ).resolve().parents[1]

    data_root = project_root / "data"
    model_root = project_root / "models"
    result_root = project_root / "results"

    model_root.mkdir(
        exist_ok=True
    )

    result_root.mkdir(
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    info = INFO["dermamnist"]

    class_names = [
        info["label"][str(i)]
        for i in range(
            len(info["label"])
        )
    ]

    num_classes = len(class_names)

    train_tf, eval_tf = build_transforms(
        args.image_size
    )

    train_ds = make_dataset(
        "train",
        train_tf,
        data_root,
        args.image_size,
    )

    val_ds = make_dataset(
        "val",
        eval_tf,
        data_root,
        args.image_size,
    )

    train_targets = get_targets(
        train_ds
    )

    class_weights, counts = (
        make_class_weights(
            train_targets,
            num_classes,
        )
    )

    print("\nClasses:")

    for i, (
        name,
        count,
        weight,
    ) in enumerate(
        zip(
            class_names,
            counts,
            class_weights.tolist(),
        )
    ):
        print(
            f"  {i}: {name} "
            f"| train={count} "
            f"| weight={weight:.4f}"
        )

    # V2: class-balanced sampling.
    # We use balanced sampling instead of class-weighted loss
    # so that minority classes are seen more frequently.
    sample_weights = (
        1.0 / np.maximum(
            counts[train_targets],
            1,
        )
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(
        num_classes
    ).to(device)

    # V3: fine-tune the pretrained EfficientNet-B0
    # instead of freezing the entire backbone.
    for parameter in model.parameters():
        parameter.requires_grad = True

    # V2 class-balanced sampler remains enabled.
    criterion = nn.CrossEntropyLoss()

    # Use a smaller learning rate for pretrained features.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    best_val_acc = -1.0

    best_path = (
        model_root
        / "best_efficientnet_b0_dermamnist.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        running_loss = 0.0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{args.epochs}"
            ),
        )

        for images, labels in progress:

            images = images.to(device)

            labels = (
                labels
                .squeeze()
                .long()
                .to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            seen += images.size(0)

            progress.set_postfix(
                loss=(
                    running_loss
                    / seen
                )
            )

        (
            val_loss,
            val_acc,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        train_loss = (
            running_loss / seen
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        class_names,

                    "image_size":
                        args.image_size,

                    "architecture":
                        "efficientnet_b0",

                    "best_val_accuracy":
                        best_val_acc,
                },
                best_path,
            )

            print(
                f"Saved best model -> "
                f"{best_path}"
            )

    metadata = {
        "architecture":
            "efficientnet_b0",

        "dataset":
            "DermaMNIST",

        "num_classes":
            num_classes,

        "classes":
            class_names,

        "image_size":
            args.image_size,

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "seed":
            args.seed,

        "device":
            str(device),

        "best_validation_accuracy":
            best_val_acc,

        "train_counts":
            counts.tolist(),
    }

    (
        result_root
        / "training_config.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("\nTraining complete.")

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Model: {best_path}"
    )


if __name__ == "__main__":
    main()
