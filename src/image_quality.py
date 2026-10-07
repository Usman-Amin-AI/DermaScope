from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageStat


@dataclass
class QualityResult:
    passed: bool
    score: float
    reasons: list[str]


def check_image_quality(
    image: Image.Image,
    min_width: int = 64,
    min_height: int = 64,
    min_std: float = 8.0,
    min_dynamic_range: float = 25.0,
) -> QualityResult:

    reasons = []

    image = image.convert("RGB")

    width, height = image.size

    # -----------------------------------------------------
    # Resolution check
    # -----------------------------------------------------

    if width < min_width or height < min_height:
        reasons.append(
            f"Image resolution is too small "
            f"({width}x{height})."
        )

    # -----------------------------------------------------
    # Pixel statistics
    # -----------------------------------------------------

    array = np.asarray(image).astype(np.float32)

    grayscale = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )

    std = float(np.std(grayscale))

    minimum = float(np.min(grayscale))
    maximum = float(np.max(grayscale))

    dynamic_range = maximum - minimum

    # -----------------------------------------------------
    # Nearly uniform / blank image check
    # -----------------------------------------------------

    if std < min_std:
        reasons.append(
            "Image has very little visual variation."
        )

    if dynamic_range < min_dynamic_range:
        reasons.append(
            "Image has insufficient brightness variation."
        )

    # -----------------------------------------------------
    # Very extreme brightness check
    # -----------------------------------------------------

    mean_brightness = float(
        np.mean(grayscale)
    )

    if mean_brightness < 15:
        reasons.append(
            "Image appears extremely dark."
        )

    if mean_brightness > 245:
        reasons.append(
            "Image appears extremely bright."
        )

    # -----------------------------------------------------
    # Quality score
    # -----------------------------------------------------

    score = 1.0

    if width < min_width or height < min_height:
        score -= 0.35

    if std < min_std:
        score -= 0.35

    if dynamic_range < min_dynamic_range:
        score -= 0.20

    if mean_brightness < 15 or mean_brightness > 245:
        score -= 0.15

    score = max(
        0.0,
        min(1.0, score),
    )

    passed = len(reasons) == 0

    return QualityResult(
        passed=passed,
        score=score,
        reasons=reasons,
    )
