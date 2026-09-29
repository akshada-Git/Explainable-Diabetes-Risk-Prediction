"""Loading and validating the PIMA Indians Diabetes dataset."""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, DATA_URLS, FEATURES, TARGET


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or malformed."""


_MANUAL_HELP = (
    "Could not find or download the dataset.\n"
    "Please download 'diabetes.csv' (PIMA Indians Diabetes Database) from Kaggle\n"
    "  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
    f"and save it as: {DATA_PATH}"
)


def parse_csv_text(text: str) -> pd.DataFrame:
    """Parse CSV text, supporting both headed and header-less PIMA files."""
    expected = FEATURES + [TARGET]
    df = pd.read_csv(io.StringIO(text))
    if not set(expected).issubset(df.columns):
        # Some public copies have no header row (e.g. "pima-indians-diabetes.data.csv").
        df = pd.read_csv(io.StringIO(text), header=None)
        if df.shape[1] != len(expected):
            raise DatasetError(
                f"Unexpected number of columns: {df.shape[1]} (expected {len(expected)})."
            )
        df.columns = expected
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns/target and return a clean numeric frame (FEATURES + TARGET)."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {missing}")
    out = df[FEATURES + [TARGET]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[TARGET].isna().any():
        raise DatasetError("Target column 'Outcome' contains non-numeric/missing values.")
    if not set(out[TARGET].unique()).issubset({0, 1}):
        raise DatasetError("Target column 'Outcome' must contain only 0/1.")
    out[TARGET] = out[TARGET].astype(int)
    if len(out) < 50:
        raise DatasetError(f"Dataset is too small ({len(out)} rows).")
    return out.reset_index(drop=True)


def _download(urls=DATA_URLS, timeout: int = 20) -> pd.DataFrame:
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                text = resp.read().decode("utf-8")
            return validate_schema(parse_csv_text(text))
        except Exception as exc:  # network error, bad content, ...
            errors.append(f"{url}: {exc}")
    raise DatasetError(_MANUAL_HELP + "\n\nDownload attempts:\n  " + "\n  ".join(errors))


def load_dataset(path: Path | str = DATA_PATH, allow_download: bool = True) -> pd.DataFrame:
    """Load the dataset from `path`; download it (and cache to `path`) if absent."""
    path = Path(path)
    if path.exists():
        try:
            return validate_schema(parse_csv_text(path.read_text(encoding="utf-8")))
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"Could not read {path}: {exc}") from exc
    if not allow_download:
        raise DatasetError(_MANUAL_HELP)
    df = _download()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
