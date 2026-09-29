"""Preprocessing: disguised-zero handling, imputation, scaling."""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, ZERO_AS_MISSING


def zeros_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Turn biologically impossible zeros into NaN (FR-1.1 / TR-6.2.1).

    Returns a copy; the input is not modified.
    """
    X = X.copy()
    cols = [c for c in ZERO_AS_MISSING if c in X.columns]
    if cols:
        X[cols] = X[cols].astype(float).mask(X[cols] == 0)
    return X


def build_pipeline(model) -> Pipeline:
    """Median imputation -> standard scaling -> model.

    Everything is fitted on the training split only (inside cross-validation
    too), so there is no data leakage, and the whole thing is saved as one
    object so inference uses exactly the training-time preprocessing (FR-4.2).
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def transform_features(pipeline: Pipeline, X: pd.DataFrame):
    """Apply the fitted imputer + scaler (everything except the final model)."""
    return pipeline[:-1].transform(X[FEATURES])
