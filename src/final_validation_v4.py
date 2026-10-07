from pathlib import Path
import json

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
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data"
    model_path = root / "models" / "v4_efficientnet_b0_dermamnist_224.pt"
    result_root = root / "results" / "final_validation_v4"
    result_root.mkdir(parents=True, exist_ok=True)

    image_size = 224
    batch_size = 32
    workers = 2

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    info = INFO["dermamnist"]

    class_names = [
        info["label"][str(i)]
        for i in range(len(info["label"]))
    ]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # EXACT SAME TRANSFORM AS THE VERIFIED EVALUATOR
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    print("=" * 60)
    print("FINAL VALIDATION — VERIFIED EVALUATION PATH")
    print("=" * 60)
    print(f"Model:  {model_path}")
    print(f"Data:   {data_root}")
    print(f"Device: {device}")
    print()

    # EXACT SAME DATASET CONSTRUCTION AS src/evaluate.py
    print("Loading test dataset...")

    test_ds = medmnist.DermaMNIST(
        split="test",
        transform=test_transform,
        size=image_size,
        download=True,
        as_rgb=True,
        root=str(data_root),
    )

    print(f"Test images: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    # EXACT SAME MODEL CONSTRUCTION
    print(f"Loading model: {model_path}")

    model = models.efficientnet_b0(weights=None)

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

    print("Model loaded successfully.")
    print()

    y_true = []
    y_pred = []
    confidences = []
    probabilities = []

    print("Running final evaluation...")

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)

            # EXACT SAME LABEL HANDLING
            labels = (
                labels
                .squeeze()
                .long()
                .to(device)
            )

            logits = model(images)

            probs = torch.softmax(logits, dim=1)

            confidence, predictions = probs.max(dim=1)

            y_true.extend(
                labels.cpu().tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

            confidences.extend(
                confidence.cpu().tolist()
            )

            probabilities.extend(
                probs.cpu().tolist()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)
    probabilities = np.array(probabilities)

    # ============================================================
    # METRICS
    # ============================================================

    accuracy = accuracy_score(y_true, y_pred)

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

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    correct = int((y_true == y_pred).sum())
    incorrect = int((y_true != y_pred).sum())

    print()
    print("=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print(f"Correct        : {correct}")
    print(f"Incorrect      : {incorrect}")
    print(f"Mean confidence: {confidences.mean():.4f}")

    print()
    print("Classification Report")
    print("=" * 60)
    print(report)

    print("Confusion Matrix")
    print("=" * 60)
    print(cm)

    # ============================================================
    # CONFIDENCE COVERAGE ANALYSIS
    # ============================================================

    print()
    print("=" * 60)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 60)

    threshold_results = {}

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:

        accepted = confidences >= threshold

        accepted_count = int(accepted.sum())

        if accepted_count > 0:
            accepted_accuracy = accuracy_score(
                y_true[accepted],
                y_pred[accepted],
            )
        else:
            accepted_accuracy = 0.0

        coverage = accepted_count / len(y_true)

        threshold_results[str(threshold)] = {
            "accepted": accepted_count,
            "coverage": float(coverage),
            "accuracy": float(accepted_accuracy),
        }

        print(
            f"{threshold:.2f} | "
            f"accepted={accepted_count:4d} | "
            f"coverage={coverage:.4f} | "
            f"accuracy={accepted_accuracy:.4f}"
        )

    # ============================================================
    # HIGH-CONFIDENCE ERRORS
    # ============================================================

    errors = np.where(y_true != y_pred)[0]

    error_records = []

    for idx in errors:

        error_records.append({
            "index": int(idx),
            "true_class": class_names[int(y_true[idx])],
            "predicted_class": class_names[int(y_pred[idx])],
            "confidence": float(confidences[idx]),
        })

    error_records.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    high_confidence_errors = [
        x for x in error_records
        if x["confidence"] >= 0.90
    ]

    print()
    print("=" * 60)
    print("HIGH-CONFIDENCE ERRORS (>= 0.90)")
    print("=" * 60)

    print(
        f"Count: {len(high_confidence_errors)}"
    )

    for item in high_confidence_errors[:20]:
        print(
            f"index={item['index']:4d} | "
            f"true={item['true_class']} | "
            f"pred={item['predicted_class']} | "
            f"confidence={item['confidence']:.4f}"
        )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    metrics = {
        "model": str(model_path),
        "architecture": "efficientnet_b0",
        "dataset": "DermaMNIST",
        "split": "test",
        "test_images": len(test_ds),
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "correct": correct,
        "incorrect": incorrect,
        "mean_confidence": float(confidences.mean()),
        "threshold_analysis": threshold_results,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "high_confidence_error_count": len(
            high_confidence_errors
        ),
    }

    with open(
        result_root / "final_validation.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        result_root / "final_validation_report.txt",
        "w",
    ) as f:
        f.write(
            "DermaScope — Final Validation Report\n"
        )
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Model: {model_path}\n"
        )
        f.write(
            f"Dataset: DermaMNIST test split\n"
        )
        f.write(
            f"Test images: {len(test_ds)}\n"
        )
        f.write(
            f"Image size: {image_size}x{image_size}\n\n"
        )

        f.write(
            f"Accuracy: {accuracy:.4f}\n"
        )
        f.write(
            f"Macro Precision: {precision:.4f}\n"
        )
        f.write(
            f"Macro Recall: {recall:.4f}\n"
        )
        f.write(
            f"Macro F1: {f1:.4f}\n"
        )
        f.write(
            f"Correct: {correct}\n"
        )
        f.write(
            f"Incorrect: {incorrect}\n"
        )
        f.write(
            f"Mean Confidence: {confidences.mean():.4f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )
        f.write("=" * 60 + "\n")
        f.write(report)

        f.write(
            "\n\nConfusion Matrix\n"
        )
        f.write("=" * 60 + "\n")
        f.write(
            np.array2string(cm)
        )

        f.write(
            "\n\nConfidence Threshold Analysis\n"
        )
        f.write("=" * 60 + "\n")

        for threshold, result in threshold_results.items():
            f.write(
                f"{threshold}: "
                f"accepted={result['accepted']}, "
                f"coverage={result['coverage']:.4f}, "
                f"accuracy={result['accuracy']:.4f}\n"
            )

    np.savetxt(
        result_root / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    with open(
        result_root / "high_confidence_errors.json",
        "w",
    ) as f:
        json.dump(
            high_confidence_errors,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        result_root / "final_validation.json"
    )
    print(
        result_root / "final_validation_report.txt"
    )
    print(
        result_root / "confusion_matrix.csv"
    )
    print(
        result_root / "high_confidence_errors.json"
    )

    print()
    print("Validation complete.")


if __name__ == "__main__":
    main()
