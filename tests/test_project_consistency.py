from pathlib import Path

from src.inference import (
    CLASS_NAMES as INFERENCE_CLASSES,
    IMAGE_SIZE as INFERENCE_IMAGE_SIZE,
    MODEL_PATH,
)

from src.gradcam import (
    CLASS_NAMES as GRADCAM_CLASSES,
    IMAGE_SIZE as GRADCAM_IMAGE_SIZE,
)


def test_class_definitions_match():
    assert INFERENCE_CLASSES == GRADCAM_CLASSES


def test_image_size_matches():
    assert INFERENCE_IMAGE_SIZE == GRADCAM_IMAGE_SIZE == 224


def test_frozen_model_exists():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 0
