import unittest

import numpy as np
import pandas as pd

from src.config import FEATURE_INFO, FEATURES
from src.explain import (_extract_positive, additivity_gap, build_contributions,
                         build_explainer, contribution_figure, figure_to_png_bytes,
                         model_units, plain_language_summary, shap_for)
from src.predict import predict_one
from src.preprocessing import transform_features
from src.report import build_pdf
from src.training import run_training
from tests.helpers import make_synthetic
import tempfile
from pathlib import Path


class TestShapShapes(unittest.TestCase):
    def test_list_output(self):
        sv, base = _extract_positive([np.zeros((3, 8)), np.ones((3, 8))], np.array([0.4, 0.6]))
        self.assertEqual(sv.shape, (3, 8)); self.assertTrue((sv == 1).all()); self.assertEqual(base, 0.6)

    def test_3d_output(self):
        arr = np.zeros((3, 8, 2)); arr[:, :, 1] = 2
        sv, base = _extract_positive(arr, np.array([0.3, 0.7]))
        self.assertTrue((sv == 2).all()); self.assertEqual(base, 0.7)

    def test_2d_output_scalar_base(self):
        sv, base = _extract_positive(np.ones((2, 8)), -0.5)
        self.assertEqual(sv.shape, (2, 8)); self.assertEqual(base, -0.5)


class TestExplainWithRealShap(unittest.TestCase):
    """Uses the real `shap` package; skipped if it is not installed."""

    @classmethod
    def setUpClass(cls):
        try:
            import shap  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("shap not installed")
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.bundle = run_training(df=make_synthetic(600), models_dir=root / "m",
                                  reports_dir=root / "r", quick=True, make_plots=False,
                                  verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_additivity_and_contributions(self):  # TR-12.4
        b = self.bundle
        vals = {f: FEATURE_INFO[f]["default"] for f in FEATURES}
        pred = predict_one(b, vals)
        expl = build_explainer(b["pipeline"], b["background"])
        Xt = transform_features(b["pipeline"], pred["frame"])
        sv, base = shap_for(expl, Xt)
        self.assertEqual(sv.shape, (1, len(FEATURES)))
        gap = additivity_gap(b["pipeline"].named_steps["model"], b["units"], sv, base, Xt)
        self.assertLess(gap, 0.02)
        contrib = build_contributions(b["pipeline"], pred["frame"], sv[0])
        self.assertEqual(len(contrib), len(FEATURES))


class TestPresentation(unittest.TestCase):
    def setUp(self):
        self.contrib = pd.DataFrame({
            "feature": ["Glucose", "Age", "BMI"],
            "label": ["Glucose", "Age", "BMI"],
            "value_text": ["148 mg/dL", "50 years", "not provided (typical value 30 kg/m\u00b2 used)"],
            "impact": [0.25, -0.05, 0.1],
            "direction": ["raises risk", "lowers risk", "raises risk"],
        })

    def test_summary_mentions_top_factor(self):
        s = plain_language_summary(0.72, "High Risk", self.contrib.sort_values("impact", ascending=False))
        self.assertIn("72%", s); self.assertIn("Glucose", s); self.assertIn("not a diagnosis", s)

    def test_units(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        self.assertEqual(model_units(LogisticRegression()), "log-odds")
        self.assertEqual(model_units(RandomForestClassifier()), "probability")

    def test_chart_and_pdf(self):
        png = figure_to_png_bytes(contribution_figure(self.contrib, "probability"))
        self.assertTrue(png.startswith(b"\x89PNG"))
        result = {"inputs": {f: FEATURE_INFO[f]["default"] for f in FEATURES},
                  "probability": 0.72, "label": "High Risk"}
        pdf = build_pdf(result, self.contrib, png, "Random Forest", "summary & <test>")
        self.assertTrue(pdf.startswith(b"%PDF"))
        pdf2 = build_pdf(result, None, None)  # EH-4 style: no explanation
        self.assertTrue(pdf2.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
