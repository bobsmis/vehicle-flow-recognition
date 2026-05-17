"""Member 5: manual truth comparison and accuracy evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AccuracyEvaluation:
    counted_total: int
    manual_total: int | None
    error_total: int | None
    accuracy: float | None


def evaluate_count_accuracy(counted_total: int, manual_total: int | None) -> AccuracyEvaluation:
    """Compare program output with manual truth while preserving the old formula."""

    if manual_total is None:
        return AccuracyEvaluation(
            counted_total=counted_total,
            manual_total=None,
            error_total=None,
            accuracy=None,
        )

    error_total = abs(counted_total - manual_total)
    if manual_total == 0 and counted_total == 0:
        accuracy = 1.0
    else:
        accuracy = max(0.0, 1.0 - error_total / max(1, manual_total))

    return AccuracyEvaluation(
        counted_total=counted_total,
        manual_total=manual_total,
        error_total=error_total,
        accuracy=accuracy,
    )
