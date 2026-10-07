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

from src.domain_gate import predict_domain


RESULTS_DIR = Path("results/domain_gate")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_DIR = Path("data/domain_gate_negative")


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

    return [
        Image.fromarray(image).convert("RGB")
        for image in data["test_images"]
    ]


def collect_negative_images():
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    paths = sorted(
        path
        for path in NEGATIVE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not paths:
        raise RuntimeError(
            f"No negative images found in {NEGATIVE_DIR}"
        )

    images = []

    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append((path.name, image))
        except Exception as exc:
            print(
                f"Skipping unreadable image "
                f"{path.name}: {exc}"
            )

    return images


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

    print("Loading local real-world negative images...")
    negative_images = collect_negative_images()

    print(
        f"DermaMNIST positive samples: "
        f"{len(positive_images)}"
    )

    print(
        f"Real-world negative samples: "
        f"{len(negative_images)}"
    )

    probabilities = []
    labels = []

    print()
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

    print()
    print("Evaluating real-world negative images...")

    negative_details = []

    for index, (filename, image) in enumerate(
        negative_images
    ):
        probability = predict_probability(image)

        probabilities.append(probability)
        labels.append(0)

        negative_details.append(
            {
                "filename": filename,
                "dermoscopic_probability": probability,
            }
        )

        print(
            f"  {filename}: "
            f"{probability:.4f}"
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
        "evaluation_type": (
            "DermaMNIST test vs local "
            "real-world unrelated images"
        ),
        "positive_dataset": "DermaMNIST test",
        "negative_dataset": (
            "Local unrelated images"
        ),
        "positive_count": len(
            positive_images
        ),
        "negative_count": len(
            negative_images
        ),
        "negative_images": negative_details,
        "thresholds": rows,
    }

    output_path = (
        RESULTS_DIR /
        "threshold_evaluation_local.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        "===== DOMAIN-GATE LOCAL THRESHOLD "
        "EVALUATION ====="
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
