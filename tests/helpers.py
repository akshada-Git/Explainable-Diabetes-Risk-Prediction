"""Synthetic PIMA-like data so the tests run without the real dataset."""
import numpy as np
import pandas as pd

from src.config import FEATURES, TARGET


def make_synthetic(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Pregnancies": rng.poisson(3.5, n).clip(0, 17),
        "Glucose": rng.normal(122, 30, n).clip(50, 199).round(),
        "BloodPressure": rng.normal(70, 12, n).clip(30, 122).round(),
        "SkinThickness": rng.normal(26, 10, n).clip(7, 99).round(),
        "Insulin": rng.gamma(2.0, 60, n).clip(15, 846).round(),
        "BMI": rng.normal(32, 6.5, n).clip(18, 67).round(1),
        "DiabetesPedigreeFunction": rng.gamma(2.0, 0.25, n).clip(0.08, 2.4).round(3),
        "Age": rng.gamma(4.0, 8.0, n).clip(21, 81).round(),
    })
    logit = (0.04 * (df.Glucose - 122) + 0.07 * (df.BMI - 32) + 0.03 * (df.Age - 33)
             + 0.8 * (df.DiabetesPedigreeFunction - 0.5) - 0.6)
    df[TARGET] = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # PIMA-style disguised missing values
    for col, frac in [("Insulin", 0.45), ("SkinThickness", 0.28), ("BloodPressure", 0.04),
                      ("Glucose", 0.007), ("BMI", 0.014)]:
        df.loc[rng.rand(n) < frac, col] = 0
    return df[FEATURES + [TARGET]]
