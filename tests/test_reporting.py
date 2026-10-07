import csv
import io

from src.reporting import build_batch_report_rows, build_csv_report


def test_batch_report_adds_probabilities_and_retains_failed_rows():
    rows = [
        {"Filename": "1. lesion.png", "Prediction": "Melanoma"},
        {"Filename": "2. broken.png", "Prediction": "Not run"},
    ]
    details = {
        "1. lesion.png": {
            "result": {"probabilities": {"Melanoma": 0.75, "Nevi": 0.25}}
        }
    }

    report_rows = build_batch_report_rows(rows, details, 0.6)

    assert report_rows[0]["Probability: Melanoma"] == 0.75
    assert report_rows[0]["Probability: Nevi"] == 0.25
    assert report_rows[1]["Prediction"] == "Not run"
    assert report_rows[1]["Uncertainty threshold"] == 0.6


def test_csv_report_preserves_multiple_rows_and_quoted_values():
    rows = [
        {"Filename": "2. lesion, left.png", "Decision": "Needs review"},
        {
            "Filename": "1. lesion, left.png",
            "Decision": "Prediction available",
            "Probability: Melanoma": 0.75,
        },
    ]

    report = build_csv_report(rows)
    parsed_rows = list(csv.DictReader(io.StringIO(report.decode("utf-8"))))

    assert parsed_rows == [
        {"Filename": "2. lesion, left.png", "Decision": "Needs review", "Probability: Melanoma": ""},
        {"Filename": "1. lesion, left.png", "Decision": "Prediction available", "Probability: Melanoma": "0.75"},
    ]


def test_csv_report_is_empty_when_there_are_no_results():
    assert build_csv_report([]) == b""