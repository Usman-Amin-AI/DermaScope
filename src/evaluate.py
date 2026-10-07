import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import models, transforms

import medmnist
from medmnist import INFO


def main():
    parser = argparse.ArgumentParser()

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
        "--workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    root = Path(
        __file__
    ).resolve().parents[1]

    data_root = root / "data"

    model_path = (
        root
        / "models"
        / "best_efficientnet_b0_dermamnist.pt"
    )

    result_root = root / "results"
    result_root.mkdir(
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    info = INFO["dermamnist"]

    class_names = [
        info["label"][str(i)]
        for i in range(
            len(info["label"])
        )
    ]

    normalize = transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    )

    test_transform = transforms.Compose([
        transforms.Resize(
            (
                args.image_size,
                args.image_size,
            )
        ),
        transforms.ToTensor(),
        normalize,
    ])

    print("Loading test dataset...")

    test_ds = medmnist.DermaMNIST(
        split="test",
        transform=test_transform,
        size=args.image_size,
        download=True,
        as_rgb=True,
        root=str(data_root),
    )

    print(
        f"Test images: {len(test_ds)}"
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    print(
        f"Loading model: {model_path}"
    )

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(class_names),
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Device: {device}"
    )

    y_true = []
    y_pred = []

    print("\nRunning test evaluation...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = (
                labels
                .squeeze()
                .long()
                .to(device)
            )

            logits = model(images)

            predictions = (
                logits.argmax(dim=1)
            )

            y_true.extend(
                labels.cpu().tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall   : {recall:.4f}"
    )

    print(
        f"Macro F1       : {f1:.4f}"
    )

    print("\nClassification Report")
    print("=" * 50)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(cm)

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(
            len(class_names)
        ),
        yticks=np.arange(
            len(class_names)
        ),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="DermaMNIST Test Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    threshold = cm.max() / 2

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    confusion_path = (
        result_root
        / "confusion_matrix.png"
    )

    fig.savefig(
        confusion_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nConfusion matrix saved to:"
    )

    print(
        confusion_path
    )

    metrics_path = (
        result_root
        / "test_metrics.txt"
    )

    metrics_path.write_text(
        "\n".join([
            "DermaMNIST Test Results",
            "",
            f"Accuracy: {accuracy:.6f}",
            f"Macro Precision: {precision:.6f}",
            f"Macro Recall: {recall:.6f}",
            f"Macro F1: {f1:.6f}",
        ])
    )

    print(
        f"Metrics saved to: {metrics_path}"
    )


if __name__ == "__main__":
    main()
