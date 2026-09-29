"""Input validation and single-person prediction."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import FEATURE_INFO, FEATURES, RISK_THRESHOLD
from .preprocessing import zeros_to_nan


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def validate_inputs(values: dict) -> list[str]:
    """Return a list of human-readable error messages (empty list = valid).

    Covers FR-3.2 / FR-3.3 and EH-1 / EH-2.
    """
    errors: list[str] = []
    for feat in FEATURES:
        info = FEATURE_INFO[feat]
        label = info["label"]
        v = values.get(feat)
        if _is_missing(v):
            if not info["optional"]:
                errors.append(f"{label} is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not math.isfinite(v):
            errors.append(f"{label} must be a finite number.")
            continue
        if v < info["min"] or v > info["max"]:
            unit = f" {info['unit']}" if info["unit"] else ""
            errors.append(
                f"{label} must be between {info['min']} and {info['max']}{unit} (you entered {v:g})."
            )
    return errors


def to_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame in training column order; None -> NaN; zeros-as-missing applied."""
    row = {f: (np.nan if _is_missing(values.get(f)) else float(values[f])) for f in FEATURES}
    return zeros_to_nan(pd.DataFrame([row], columns=FEATURES))


def predict_one(bundle: dict, values: dict) -> dict:
    """Predict risk for one person. Returns probability, label and the model-ready frame."""
    frame = to_frame(values)
    proba = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    threshold = bundle.get("threshold", RISK_THRESHOLD)
    is_high = proba >= threshold
    return {
        "probability": proba,
        "is_high": bool(is_high),
        "label": "High Risk" if is_high else "Low Risk",
        "frame": frame,
    }
