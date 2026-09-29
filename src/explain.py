"""Explainability layer (SHAP) plus plain-language and chart helpers.

`shap` is imported lazily so the rest of the project (and most tests) still
work if it is not installed; the app then shows the EH-4 fallback notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import FEATURE_INFO, FEATURES


# ----------------------------------------------------------------- units ----
def model_units(clf) -> str:
    """Scale in which SHAP values are expressed for this model.

    Random Forest / Decision Tree -> probability (values add up to P(diabetes)).
    Logistic Regression / XGBoost  -> log-odds.
    """
    name = clf.__class__.__name__
    if isinstance(clf, LogisticRegression) or name.startswith(("XGB", "LGBM")):
        return "log-odds"
    return "probability"


def unit_label(units: str) -> str:
    if units == "log-odds":
        return "Impact on risk (log-odds)"
    return "Impact on risk (probability points)"


# --------------------------------------------------------------- explainer --
def build_explainer(pipeline, background):
    """Create the right SHAP explainer for the fitted model inside `pipeline`."""
    import shap  # lazy import

    clf = pipeline.named_steps["model"]
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, background)
    return shap.TreeExplainer(clf)


def _extract_positive(shap_values, expected_value):
    """Normalise SHAP output across shap versions to (2-D array, float base).

    Depending on the shap version and model, binary-classifier output is a
    list of two arrays, a 3-D array (n, features, 2) or a plain 2-D array.
    We always return the values for the positive class ("has diabetes").
    """
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    ev = np.atleast_1d(np.asarray(expected_value, dtype=float)).ravel()
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    base = float(ev[1]) if ev.size > 1 else float(ev[0])
    return sv, base


def shap_for(explainer, X_scaled):
    """SHAP values (n, n_features) and base value for the positive class."""
    try:
        sv = explainer.shap_values(X_scaled, check_additivity=False)
    except TypeError:  # e.g. LinearExplainer has no check_additivity argument
        sv = explainer.shap_values(X_scaled)
    return _extract_positive(sv, explainer.expected_value)


def additivity_gap(clf, units: str, values, base: float, X_scaled) -> float:
    """Max |(base + sum(SHAP)) - model output|. Should be ~0 (TR-12.4)."""
    if units == "log-odds":
        if hasattr(clf, "decision_function"):
            out = clf.decision_function(X_scaled)
        else:
            p = np.clip(clf.predict_proba(X_scaled)[:, 1], 1e-9, 1 - 1e-9)
            out = np.log(p / (1 - p))
    else:
        out = clf.predict_proba(X_scaled)[:, 1]
    recon = base + np.asarray(values).sum(axis=1)
    return float(np.max(np.abs(recon - out)))


# ---------------------------------------------------------- contributions ---
def _fmt_value(feat: str, v: float, imputed_value: float | None) -> str:
    info = FEATURE_INFO[feat]
    unit = f" {info['unit']}" if info["unit"] else ""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"not provided (typical value {imputed_value:g}{unit} used)"
    return f"{v:g}{unit}"


def build_contributions(pipeline, frame: pd.DataFrame, values_row) -> pd.DataFrame:
    """Per-feature table for one prediction, sorted by absolute impact."""
    stats = pipeline.named_steps["imputer"].statistics_
    rows = []
    for i, feat in enumerate(FEATURES):
        raw = frame.iloc[0][feat]
        impact = float(values_row[i])
        rows.append(
            {
                "feature": feat,
                "label": FEATURE_INFO[feat]["label"],
                "value_text": _fmt_value(feat, raw, float(stats[i])),
                "impact": impact,
                "direction": ("raises risk" if impact > 0 else "lowers risk" if impact < 0 else "no effect"),
            }
        )
    df = pd.DataFrame(rows)
    order = df["impact"].abs().sort_values(ascending=False).index
    return df.loc[order].reset_index(drop=True)


def plain_language_summary(probability: float, label: str, contrib: pd.DataFrame,
                           n_up: int = 3, n_down: int = 2) -> str:
    """Rule-based one-paragraph explanation (FR-5.4). No LLM needed."""
    up = contrib[contrib["impact"] > 0].head(n_up)
    down = contrib[contrib["impact"] < 0].head(n_down)

    def _names(df):
        return ", ".join(f"{r.label} ({r.value_text})" for r in df.itertuples())

    parts = [f"The model estimates a {probability * 100:.0f}% chance of diabetes ({label})."]
    if len(up):
        parts.append(f"The factors pushing the estimate up the most were {_names(up)}.")
    if len(down):
        parts.append(f"The factors pulling it down the most were {_names(down)}.")
    parts.append(
        "These are patterns the model learned from data, not medical causes, "
        "and they are not a diagnosis."
    )
    return " ".join(parts)


def contribution_figure(contrib: pd.DataFrame, units: str, top_n: int = 8):
    """Horizontal bar chart of SHAP contributions (red raises, blue lowers risk)."""
    from matplotlib.figure import Figure

    d = contrib.head(top_n).iloc[::-1]  # biggest at the top
    fig = Figure(figsize=(7.5, 0.5 * len(d) + 1.4))
    ax = fig.subplots()
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d["impact"]]
    ylabels = [
        f"{r.label} = " + ("not provided" if r.value_text.startswith("not provided") else r.value_text)
        for r in d.itertuples()
    ]
    ax.barh(range(len(d)), d["impact"], color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(unit_label(units) + "\n(red = raises risk, blue = lowers risk)", fontsize=9)
    ax.set_title("Why did the model give this result?", fontsize=11)
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()
