"""Small-sample confidence calibration with explicit deployment diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


CALIBRATION_VERSION = "beta_isotonic_v1"


@dataclass
class CalibrationBlock:
    min_score: float
    max_score: float
    positives: int
    total: int
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_score": self.min_score,
            "max_score": self.max_score,
            "positives": self.positives,
            "total": self.total,
            "probability": self.probability,
        }


@dataclass
class BetaIsotonicCalibrator:
    alpha: float = 1.0
    beta: float = 1.0
    min_positives: int = 2
    min_negatives: int = 2
    blocks: list[CalibrationBlock] = field(default_factory=list)
    n_samples: int = 0
    n_positives: int = 0
    status: str = "unfitted"

    @property
    def deployable(self) -> bool:
        return self.status == "ok"

    def fit(self, scores: Iterable[float], labels: Iterable[int | bool]) -> "BetaIsotonicCalibrator":
        pairs = sorted(
            (min(1.0, max(0.0, float(score))), int(bool(label)))
            for score, label in zip(scores, labels)
        )
        self.n_samples = len(pairs)
        self.n_positives = sum(label for _, label in pairs)
        negatives = self.n_samples - self.n_positives
        self.blocks = []
        if not pairs:
            self.status = "degenerate_no_samples"
            return self

        raw_blocks: list[dict[str, float | int]] = []
        for score, label in pairs:
            if raw_blocks and float(raw_blocks[-1]["max_score"]) == score:
                raw_blocks[-1]["positives"] = int(raw_blocks[-1]["positives"]) + label
                raw_blocks[-1]["total"] = int(raw_blocks[-1]["total"]) + 1
            else:
                raw_blocks.append(
                    {
                        "min_score": score,
                        "max_score": score,
                        "positives": label,
                        "total": 1,
                    }
                )

        def probability(block: dict[str, float | int]) -> float:
            return (int(block["positives"]) + self.alpha) / (
                int(block["total"]) + self.alpha + self.beta
            )

        index = 0
        while index < len(raw_blocks) - 1:
            if probability(raw_blocks[index]) <= probability(raw_blocks[index + 1]):
                index += 1
                continue
            left = raw_blocks[index]
            right = raw_blocks.pop(index + 1)
            left["max_score"] = right["max_score"]
            left["positives"] = int(left["positives"]) + int(right["positives"])
            left["total"] = int(left["total"]) + int(right["total"])
            if index:
                index -= 1

        self.blocks = [
            CalibrationBlock(
                min_score=float(block["min_score"]),
                max_score=float(block["max_score"]),
                positives=int(block["positives"]),
                total=int(block["total"]),
                probability=probability(block),
            )
            for block in raw_blocks
        ]
        if self.n_positives < self.min_positives:
            self.status = "degenerate_too_few_positives"
        elif negatives < self.min_negatives:
            self.status = "degenerate_too_few_negatives"
        elif len({score for score, _ in pairs}) < 2:
            self.status = "degenerate_constant_scores"
        else:
            self.status = "ok"
        return self

    def calibrate(self, score: float) -> float:
        if not self.blocks:
            return min(1.0, max(0.0, float(score)))
        value = min(1.0, max(0.0, float(score)))
        for block in self.blocks:
            if value <= block.max_score:
                return block.probability
        return self.blocks[-1].probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CALIBRATION_VERSION,
            "status": self.status,
            "deployable": self.deployable,
            "alpha": self.alpha,
            "beta": self.beta,
            "min_positives": self.min_positives,
            "min_negatives": self.min_negatives,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BetaIsotonicCalibrator":
        calibrator = cls(
            alpha=float(payload.get("alpha", 1.0)),
            beta=float(payload.get("beta", 1.0)),
            min_positives=int(payload.get("min_positives", 2)),
            min_negatives=int(payload.get("min_negatives", 2)),
        )
        calibrator.n_samples = int(payload.get("n_samples", 0))
        calibrator.n_positives = int(payload.get("n_positives", 0))
        calibrator.status = str(payload.get("status", "unfitted"))
        calibrator.blocks = [
            CalibrationBlock(
                min_score=float(block["min_score"]),
                max_score=float(block["max_score"]),
                positives=int(block["positives"]),
                total=int(block["total"]),
                probability=float(block["probability"]),
            )
            for block in payload.get("blocks", [])
        ]
        return calibrator


def calibration_metrics(
    scores: Iterable[float], labels: Iterable[int | bool], *, bins: int = 10
) -> dict[str, float | int]:
    pairs = [(min(1.0, max(0.0, float(s))), int(bool(y))) for s, y in zip(scores, labels)]
    if not pairs:
        return {"n": 0, "positives": 0, "brier": 0.0, "ece": 0.0}
    brier = sum((score - label) ** 2 for score, label in pairs) / len(pairs)
    ece = 0.0
    for index in range(max(1, bins)):
        low = index / bins
        high = (index + 1) / bins
        bucket = [pair for pair in pairs if low <= pair[0] < high or (index == bins - 1 and pair[0] == 1.0)]
        if not bucket:
            continue
        confidence = sum(score for score, _ in bucket) / len(bucket)
        accuracy = sum(label for _, label in bucket) / len(bucket)
        ece += len(bucket) / len(pairs) * abs(confidence - accuracy)
    return {
        "n": len(pairs),
        "positives": sum(label for _, label in pairs),
        "brier": brier,
        "ece": ece,
    }
