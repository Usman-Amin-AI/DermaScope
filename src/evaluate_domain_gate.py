from pathlib import Path
import json

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torchvision import datasets

from src.domain_gate import predict_domain


RESULTS_DIR = Path("results/domain_gate")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def collect_dermamnist_test():
    npz_path = Path("data/dermamnist.npz")

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)

    if "test_images" not in data:
        raise KeyError(
            f"Expected 'test_images' in {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    images = data["test_images"]

    return [
        Image.fromarray(image).convert("RGB")
        for image in images
    ]


def collect_cifar10_test():
    dataset = datasets.CIFAR10(
        root="data/cifar10",
        train=False,
        download=True,
    )

    return [
        Image.fromarray(np.asarray(image)).convert("RGB")
        for image, _ in dataset
    ]


def predict_probability(image):
    result = predict_domain(
        image,
        threshold=0.50,
    )

    return float(
        result["dermoscopic_probability"]
    )


def main():
    print("Loading DermaMNIST test set...")
    positive_images = collect_dermamnist_test()

    print("Loading CIFAR-10 test set...")
    negative_images = collect_cifar10_test()

    print(
        f"DermaMNIST samples: {len(positive_images)}"
    )
    print(
        f"CIFAR-10 samples:   {len(negative_images)}"
    )

    probabilities = []
    labels = []

    print("Evaluating DermaMNIST...")

    for index, image in enumerate(positive_images):
        probabilities.append(
            predict_probability(image)
        )
        labels.append(1)

        if (index + 1) % 500 == 0:
            print(
                f"  DermaMNIST: "
                f"{index + 1}/{len(positive_images)}"
            )

    print("Evaluating CIFAR-10...")

    for index, image in enumerate(negative_images):
        probabilities.append(
            predict_probability(image)
        )
        labels.append(0)

        if (index + 1) % 500 == 0:
            print(
                f"  CIFAR-10: "
                f"{index + 1}/{len(negative_images)}"
            )

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    thresholds = np.arange(
        0.30,
        0.951,
        0.05,
    )

    rows = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "accuracy": float(
                    accuracy_score(
                        labels,
                        predictions,
                    )
                ),
                "precision": float(
                    precision_score(
                        labels,
                        predictions,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        labels,
                        predictions,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        labels,
                        predictions,
                        zero_division=0,
                    )
                ),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    output = {
        "dataset": {
            "positive": "DermaMNIST test",
            "negative": "CIFAR-10 test",
            "positive_count": len(
                positive_images
            ),
            "negative_count": len(
                negative_images
            ),
        },
        "thresholds": rows,
    }

    output_path = (
        RESULTS_DIR /
        "threshold_evaluation.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        "===== DOMAIN-GATE THRESHOLD EVALUATION ====="
    )

    print(
        f"{'Threshold':>9} "
        f"{'Accuracy':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FP':>6} "
        f"{'FN':>6}"
    )

    for row in rows:
        print(
            f"{row['threshold']:>9.2f} "
            f"{row['accuracy']:>10.4f} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} "
            f"{row['false_positive']:>6} "
            f"{row['false_negative']:>6}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
