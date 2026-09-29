import json
import tempfile
import unittest
from pathlib import Path

import joblib

from src.config import FEATURE_INFO, FEATURES, MIN_TEST_ROC_AUC, MODEL_PATH, REPORTS_DIR
from src.predict import predict_one
from src.training import run_training
from tests.helpers import make_synthetic


class TestTrainingPipeline(unittest.TestCase):
    """Runs the whole pipeline on synthetic data (fast, no dataset needed)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.bundle = run_training(df=make_synthetic(600), models_dir=root / "m",
                                  reports_dir=root / "r", quick=True, make_plots=False,
                                  verbose=False)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        self.assertTrue((self.root / "m" / "diabetes_model.joblib").exists())
        rep = json.loads((self.root / "r" / "metrics.json").read_text())
        self.assertIn(rep["selected_model"], {"Logistic Regression", "Decision Tree",
                                              "Random Forest", "XGBoost"})
        self.assertTrue((self.root / "r" / "model_comparison.csv").exists())

    def test_bundle_round_trip_and_predict(self):
        b = joblib.load(self.root / "m" / "diabetes_model.joblib")
        out = predict_one(b, {f: FEATURE_INFO[f]["default"] for f in FEATURES})
        self.assertTrue(0 <= out["probability"] <= 1)
        self.assertIn(out["label"], {"High Risk", "Low Risk"})

    def test_higher_glucose_bmi_raises_risk(self):
        base = {f: FEATURE_INFO[f]["default"] for f in FEATURES}
        low = dict(base, Glucose=85, BMI=22.0, Age=25)
        high = dict(base, Glucose=190, BMI=40.0, Age=55)
        self.assertGreater(predict_one(self.bundle, high)["probability"],
                           predict_one(self.bundle, low)["probability"])

    def test_metrics_reasonable_on_synthetic(self):
        self.assertGreater(self.bundle["metrics"]["roc_auc"], 0.6)


class TestSavedRealModel(unittest.TestCase):
    """TR-12.2: quality gate on the model you actually deploy (skipped until trained)."""

    def test_saved_model_meets_threshold(self):
        report = REPORTS_DIR / "metrics.json"
        if not (MODEL_PATH.exists() and report.exists()):
            self.skipTest("Run `python train.py` first.")
        m = json.loads(report.read_text())["test_metrics_selected"]
        self.assertGreaterEqual(m["roc_auc"], MIN_TEST_ROC_AUC)


if __name__ == "__main__":
    unittest.main()
