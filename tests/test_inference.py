import math

from PIL import Image

from src.inference import CLASS_NAMES, predict


def test_prediction_structure():
    image = Image.new("RGB", (256, 256), (120, 80, 60))

    result = predict(image)

    assert "prediction" in result
    assert "confidence" in result
    assert "status" in result
    assert "probabilities" in result


def test_prediction_class_is_valid():
    image = Image.new("RGB", (256, 256), (120, 80, 60))

    result = predict(image)

    assert result["prediction"] in CLASS_NAMES


def test_probability_distribution_is_valid():
    image = Image.new("RGB", (256, 256), (120, 80, 60))

    result = predict(image)

    probabilities = result["probabilities"]

    assert set(probabilities.keys()) == set(CLASS_NAMES)

    for probability in probabilities.values():
        assert 0.0 <= probability <= 1.0
        assert math.isfinite(probability)

    assert abs(sum(probabilities.values()) - 1.0) < 1e-5


def test_confidence_matches_top_probability():
    image = Image.new("RGB", (256, 256), (120, 80, 60))

    result = predict(image)

    top_probability = max(result["probabilities"].values())

    assert abs(result["confidence"] - top_probability) < 1e-5


def test_uncertainty_status_is_consistent():
    image = Image.new("RGB", (256, 256), (120, 80, 60))

    result = predict(image, uncertainty_threshold=0.60)

    if result["confidence"] < 0.60:
        assert result["status"] == "Uncertain"
    else:
        assert result["status"] == "Prediction available"
