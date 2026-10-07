from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from medmnist import DermaMNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "domain_gate"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DomainDataset(Dataset):
    def __init__(self, positive, negative, transform=None):
        self.positive = positive
        self.negative = negative
        self.transform = transform

    def __len__(self):
        return len(self.positive) + len(self.negative)

    def __getitem__(self, index):
        if index < len(self.positive):
            image, _ = self.positive[index]
            label = 1
        else:
            image, _ = self.negative[index - len(self.positive)]
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


print("Loading DermaMNIST...")
derma_train = DermaMNIST(
    split="train",
    root=DATA_DIR,
    download=True,
    as_rgb=True,
)

derma_val = DermaMNIST(
    split="val",
    root=DATA_DIR,
    download=True,
    as_rgb=True,
)

derma_test = DermaMNIST(
    split="test",
    root=DATA_DIR,
    download=True,
    as_rgb=True,
)

print("Loading CIFAR-10 negatives...")
cifar_train = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
)

cifar_test = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
)

# Keep train/validation/test sources disjoint.
negative_train = torch.utils.data.Subset(cifar_train, range(0, len(derma_train)))
negative_val = torch.utils.data.Subset(
    cifar_train,
    range(len(derma_train), len(derma_train) + len(derma_val)),
)
negative_test = torch.utils.data.Subset(
    cifar_test,
    range(0, len(derma_test)),
)

train_ds = DomainDataset(
    derma_train,
    negative_train,
    transform=train_transform,
)

val_ds = DomainDataset(
    derma_val,
    negative_val,
    transform=eval_transform,
)

test_ds = DomainDataset(
    derma_test,
    negative_test,
    transform=eval_transform,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")
print(f"Train: {len(train_ds)}")
print(f"Validation: {len(val_ds)}")
print(f"Test: {len(test_ds)}")


weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Freeze ImageNet feature extractor for this lightweight gate.
for parameter in model.features.parameters():
    parameter.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2,
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LR,
    weight_decay=1e-4,
)


def evaluate(loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return accuracy, precision, recall, f1


best_val_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    val_metrics = evaluate(val_loader)

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"val_acc={val_metrics[0]:.4f} "
        f"val_precision={val_metrics[1]:.4f} "
        f"val_recall={val_metrics[2]:.4f} "
        f"val_f1={val_metrics[3]:.4f}"
    )

    if val_metrics[3] > best_val_f1:
        best_val_f1 = val_metrics[3]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "model": "EfficientNet-B0",
                "task": "dermoscopic_vs_non_dermoscopic",
                "classes": ["non_dermoscopic", "dermoscopic"],
            },
            MODEL_DIR / "domain_gate_efficientnet_b0.pt",
        )


# Load best checkpoint.
checkpoint = torch.load(
    MODEL_DIR / "domain_gate_efficientnet_b0.pt",
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(test_loader)

print()
print("===== DOMAIN GATE TEST RESULTS =====")
print(f"Accuracy:  {test_metrics[0]:.4f}")
print(f"Precision: {test_metrics[1]:.4f}")
print(f"Recall:    {test_metrics[2]:.4f}")
print(f"Macro F1:  {test_metrics[3]:.4f}")

with open(RESULT_DIR / "domain_gate_metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": test_metrics[0],
            "precision": test_metrics[1],
            "recall": test_metrics[2],
            "f1": test_metrics[3],
            "train_size": len(train_ds),
            "validation_size": len(val_ds),
            "test_size": len(test_ds),
            "image_size": IMAGE_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "seed": SEED,
            "negative_dataset": "CIFAR-10",
            "positive_dataset": "DermaMNIST",
        },
        f,
        indent=2,
    )

print()
print("Saved:")
print(MODEL_DIR / "domain_gate_efficientnet_b0.pt")
print(RESULT_DIR / "domain_gate_metrics.json")
