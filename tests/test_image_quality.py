from PIL import Image

from src.image_quality import check_image_quality


def test_valid_image_passes():
    # Create a clearly non-uniform synthetic image with
    # sufficient grayscale variation for the quality checker.
    image = Image.new("RGB", (256, 256))
    pixels = image.load()

    for x in range(256):
        for y in range(256):
            value = (x + y) % 256
            pixels[x, y] = (
                value,
                (value * 3) % 256,
                (value * 5) % 256,
            )

    result = check_image_quality(image)

    assert result.passed is True
    assert 0.0 <= result.score <= 1.0
    assert result.reasons == []


def test_tiny_image_fails():
    image = Image.new("RGB", (32, 32), "gray")

    result = check_image_quality(image)

    assert result.passed is False
    assert any("too small" in reason for reason in result.reasons)


def test_dark_image_fails():
    image = Image.new("RGB", (256, 256), (0, 0, 0))

    result = check_image_quality(image)

    assert result.passed is False
    assert len(result.reasons) > 0


def test_bright_image_fails():
    image = Image.new("RGB", (256, 256), (255, 255, 255))

    result = check_image_quality(image)

    assert result.passed is False
    assert len(result.reasons) > 0
