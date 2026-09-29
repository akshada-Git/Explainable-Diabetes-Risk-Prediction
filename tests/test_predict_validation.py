import unittest

from src.config import FEATURE_INFO, FEATURES
from src.predict import to_frame, validate_inputs


def valid():
    return {f: FEATURE_INFO[f]["default"] for f in FEATURES}


class TestValidation(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(validate_inputs(valid()), [])

    def test_missing_required(self):  # EH-1
        v = valid(); v["Glucose"] = None
        self.assertTrue(any("Glucose" in e and "required" in e for e in validate_inputs(v)))

    def test_optional_may_be_missing(self):
        v = valid(); v["Insulin"] = None; v["SkinThickness"] = None
        self.assertEqual(validate_inputs(v), [])

    def test_negative_age(self):  # EH-2
        v = valid(); v["Age"] = -5
        self.assertTrue(any("Age" in e for e in validate_inputs(v)))

    def test_out_of_range_and_nan_and_text(self):
        v = valid(); v["BMI"] = 500
        self.assertTrue(any("BMI" in e for e in validate_inputs(v)))
        v = valid(); v["BMI"] = float("nan")
        self.assertTrue(any("BMI" in e for e in validate_inputs(v)))
        v = valid(); v["BMI"] = "abc"
        self.assertTrue(any("BMI" in e for e in validate_inputs(v)))

    def test_to_frame_order_and_nan(self):
        v = valid(); v["Insulin"] = None
        f = to_frame(v)
        self.assertEqual(list(f.columns), FEATURES)
        self.assertTrue(f["Insulin"].isna().all())


if __name__ == "__main__":
    unittest.main()
