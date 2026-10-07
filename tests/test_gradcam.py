from pathlib import Path

import medmnist
import numpy as np

from src.gradcam import (
    CLASS_NAMES,
    create_overlay,
    generate_gradcam,
)


def test_gradcam_generation():
    root = Path(__file__).resolve().parents[1]

    dataset = medmnist.DermaMNIST(
        split="test",
        transform=None,
        size=224,
        download=True,
        as_rgb=True,
        root=str(root / "data"),
    )

    image, label = dataset[0]

    heatmap, predicted_class, confidence = generate_gradcam(image)

    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= 1.0
    assert 0.0 <= float(heatmap.max()) <= 1.0

    assert predicted_class in range(len(CLASS_NAMES))
    assert 0.0 <= confidence <= 1.0


def test_gradcam_overlay():
    root = Path(__file__).resolve().parents[1]

    dataset = medmnist.DermaMNIST(
        split="test",
        transform=None,
        size=224,
        download=True,
        as_rgb=True,
        root=str(root / "data"),
    )

    image, _ = dataset[0]

    heatmap, _, _ = generate_gradcam(image)
    overlay = create_overlay(image, heatmap)

    assert overlay.size == (224, 224)
    assert overlay.mode == "RGB"
