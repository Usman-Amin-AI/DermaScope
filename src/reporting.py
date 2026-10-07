import csv
import io
from collections.abc import Mapping, Sequence


def build_batch_report_rows(
    rows: Sequence[Mapping[str, object]],
    details: Mapping[str, Mapping[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    report_rows = []
    for row in rows:
        report_row = {
            "Model": "EfficientNet-B0 V4",
            "Uncertainty threshold": threshold,
            **row,
        }
        detail = details.get(str(row["Filename"]))
        result = detail.get("result") if detail else None
        if isinstance(result, Mapping):
            probabilities = result.get("probabilities")
            if isinstance(probabilities, Mapping):
                report_row.update(
                    {
                        f"Probability: {class_name}": probability
                        for class_name, probability in probabilities.items()
                    }
                )
        report_rows.append(report_row)
    return report_rows


def build_csv_report(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""

    output = io.StringIO(newline="")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")