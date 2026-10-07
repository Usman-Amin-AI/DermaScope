from pathlib import Path

import numpy as np
from PIL import Image

from src.domain_gate import predict_domain


REAL_WORLD_NEGATIVE = Path(
    "/home/anonymous/Downloads/f510ade950cb16de7a3f5b9e1c32085f.jpg"
)


def test_domain_gate_output_structure():
    image = Image.fromarray(
        np.random.default_rng(42).integers(
            0,
            256,
            size=(224, 224, 3),
            dtype=np.uint8,
        )
    )

    result = predict_domain(image, threshold=0.60)

    assert set(result.keys()) == {
        "is_dermoscopic",
        "label",
        "dermoscopic_probability",
        "non_dermoscopic_probability",
        "threshold",
    }

    assert isinstance(result["is_dermoscopic"], bool)
    assert result["label"] in {"dermoscopic", "non_dermoscopic"}
    assert 0.0 <= result["dermoscopic_probability"] <= 1.0
    assert 0.0 <= result["non_dermoscopic_probability"] <= 1.0
    assert abs(
        result["dermoscopic_probability"]
        + result["non_dermoscopic_probability"]
        - 1.0
    ) < 1e-5
    assert result["threshold"] == 0.60


def test_domain_gate_rejects_known_unrelated_image():
    if not REAL_WORLD_NEGATIVE.exists():
        raise AssertionError(
            f"Expected test image does not exist: {REAL_WORLD_NEGATIVE}"
        )

    image = Image.open(REAL_WORLD_NEGATIVE).convert("RGB")

    result = predict_domain(image, threshold=0.60)

    assert result["is_dermoscopic"] is False
    assert result["label"] == "non_dermoscopic"


def test_domain_gate_threshold_changes_acceptance_logic():
    image = Image.fromarray(
        np.full(
            (224, 224, 3),
            128,
            dtype=np.uint8,
        )
    )

    result = predict_domain(image, threshold=0.60)

    probability = result["dermoscopic_probability"]

    low_threshold_result = predict_domain(
        image,
        threshold=max(0.0, probability - 0.01),
    )

    high_threshold_result = predict_domain(
        image,
        threshold=min(1.0, probability + 0.01),
    )

    assert low_threshold_result["is_dermoscopic"] is True
    assert high_threshold_result["is_dermoscopic"] is False
