import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES, TARGET, ZERO_AS_MISSING
from src.data import DatasetError, load_dataset, parse_csv_text, validate_schema
from src.preprocessing import build_pipeline, transform_features, zeros_to_nan
from tests.helpers import make_synthetic


class TestData(unittest.TestCase):
    def test_validate_schema_ok(self):
        df = validate_schema(make_synthetic(100))
        self.assertEqual(list(df.columns), FEATURES + [TARGET])

    def test_validate_schema_missing_column(self):
        with self.assertRaises(DatasetError):
            validate_schema(make_synthetic(100).drop(columns=["Glucose"]))

    def test_validate_schema_bad_target(self):
        df = make_synthetic(100)
        df.loc[0, TARGET] = 7
        with self.assertRaises(DatasetError):
            validate_schema(df)

    def test_headerless_csv(self):
        df = make_synthetic(100)
        text = df.to_csv(index=False, header=False)
        parsed = parse_csv_text(text)
        self.assertEqual(list(parsed.columns), FEATURES + [TARGET])
        self.assertEqual(len(parsed), 100)

    def test_load_from_disk_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "diabetes.csv"
            make_synthetic(120).to_csv(p, index=False)
            self.assertEqual(len(load_dataset(p, allow_download=False)), 120)

    def test_missing_file_no_download(self):
        with self.assertRaises(DatasetError):
            load_dataset(Path(tempfile.gettempdir()) / "definitely_missing.csv", allow_download=False)


class TestPreprocessing(unittest.TestCase):
    def test_zeros_become_nan_only_in_listed_columns(self):
        X = pd.DataFrame([{f: 0 for f in FEATURES}])
        out = zeros_to_nan(X)
        for c in ZERO_AS_MISSING:
            self.assertTrue(np.isnan(out.loc[0, c]), c)
        self.assertEqual(out.loc[0, "Pregnancies"], 0)  # a real zero
        self.assertEqual(X.loc[0, "Glucose"], 0)        # input untouched

    def test_pipeline_imputes_and_scales(self):
        from sklearn.linear_model import LogisticRegression

        df = make_synthetic(300)
        X, y = zeros_to_nan(df[FEATURES]), df[TARGET]
        pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
        Xt = transform_features(pipe, X)
        self.assertFalse(np.isnan(Xt).any())
        np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=1e-6)
        proba = pipe.predict_proba(X.iloc[:5])
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


if __name__ == "__main__":
    unittest.main()
