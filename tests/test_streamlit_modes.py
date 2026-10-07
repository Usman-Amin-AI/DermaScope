from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.inference import CLASS_NAMES


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"
SAMPLE_IMAGE = APP_FILE.parent / "assets" / "Image1.png"


def _fake_model(stack: ExitStack):
    stack.enter_context(
        patch(
            "src.domain_gate.predict_domain",
            return_value={"dermoscopic_probability": 0.95},
        )
    )
    stack.enter_context(
        patch(
            "src.inference.predict",
            return_value={
                "prediction": "Melanoma",
                "confidence": 0.9,
                "status": "Prediction available",
                "probabilities": {name: 1 / len(CLASS_NAMES) for name in CLASS_NAMES},
            },
        )
    )
    stack.enter_context(
        patch(
            "src.gradcam.generate_gradcam",
            return_value=(np.zeros((224, 224)), 4, 0.9),
        )
    )
    stack.enter_context(
        patch("src.gradcam.create_overlay", return_value=Image.new("RGB", (224, 224)))
    )


def test_single_and_batch_upload_modes_render_without_errors():
    app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()

    assert not app.exception
    assert app.get_by_key("upload_mode").value == "Single image"
    assert any("Upload dermoscopic image" in item.value for item in app.subheader)

    app.get_by_key("upload_mode").set_value("Batch images").run()

    assert not app.exception
    assert app.get_by_key("upload_mode").value == "Batch images"
    assert app.info[0].value == "Select two or more images to begin a batch analysis."


def test_single_image_analysis_shows_downloadable_report():
    image_bytes = SAMPLE_IMAGE.read_bytes()
    app = AppTest.from_file(str(APP_FILE), default_timeout=60)

    with ExitStack() as stack:
        _fake_model(stack)
        app.run()
        app.file_uploader[0].set_value(
            ("lesion.png", image_bytes, "image/png")
        ).run()
        next(button for button in app.button if button.label == "Analyze Image").click().run()

    assert not app.exception
    download = app.get_by_key("single_report_download")
    assert download.proto.url.endswith(".csv")
    assert any(metric.label == "Predicted class" for metric in app.metric)


def test_batch_analysis_keeps_duplicate_and_unreadable_files_in_report():
    image_bytes = SAMPLE_IMAGE.read_bytes()
    app = AppTest.from_file(str(APP_FILE), default_timeout=60)

    with ExitStack() as stack:
        _fake_model(stack)
        app.run()
        app.get_by_key("upload_mode").set_value("Batch images").run()
        app.file_uploader[0].set_value(
            [
                ("same.png", image_bytes, "image/png"),
                ("same.png", image_bytes, "image/png"),
                ("broken.png", b"not an image", "image/png"),
            ]
        ).run()
        app.get_by_key("analyze_batch").click().run()

    assert not app.exception
    download = app.get_by_key("batch_report_download")
    assert download.proto.url.endswith(".csv")
    report_rows = app.dataframe[0].value
    assert report_rows["Filename"].tolist() == [
        "1. same.png",
        "2. same.png",
        "3. broken.png",
    ]
    assert report_rows["Quality"].tolist() == ["Passed", "Passed", "Unreadable"]


def test_public_upload_limits_reject_oversized_inputs():
    app = AppTest.from_file(str(APP_FILE), default_timeout=30)

    app.run()
    app.file_uploader[0].set_value(
        ("oversized.png", b"x" * (20 * 1024 * 1024 + 1), "image/png")
    ).run()

    assert not app.exception
    assert "20 MB" in app.error[0].value


def test_batch_limit_rejects_more_than_20_images():
    image_bytes = SAMPLE_IMAGE.read_bytes()
    app = AppTest.from_file(str(APP_FILE), default_timeout=30)

    app.run()
    app.get_by_key("upload_mode").set_value("Batch images").run()
    app.file_uploader[0].set_value(
        [(f"image-{index}.png", image_bytes, "image/png") for index in range(21)]
    ).run()

    assert not app.exception
    assert "20 images" in app.error[0].value