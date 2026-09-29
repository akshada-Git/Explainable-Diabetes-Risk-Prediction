"""Offline training pipeline.

Steps: load -> clean -> stratified split -> tune 3 (or 4) models with 5-fold CV
-> pick the best by cross-validated ROC-AUC (training data only) -> evaluate
once on the held-out test set -> global SHAP insights -> save everything.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from .config import (CV_FOLDS, FEATURES, MODEL_FILENAME, MODELS_DIR, RANDOM_STATE,
                     REPORTS_DIR, RISK_THRESHOLD, TARGET, TEST_SIZE)
from .data import load_dataset, validate_schema
from .explain import (additivity_gap, build_explainer, model_units, shap_for, unit_label)
from .preprocessing import build_pipeline, transform_features, zeros_to_nan


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def candidate_models(quick: bool = False) -> dict:
    """name -> (estimator, param_grid). Grids use the pipeline's `model__` prefix."""
    rs = RANDOM_STATE
    cands = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
            {"model__C": [0.1, 1, 10] if quick else [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=rs),
            {"model__max_depth": [3, 5] if quick else [3, 4, 5, 6],
             "model__min_samples_leaf": [10] if quick else [5, 10, 20]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=rs, n_jobs=1),
            {"model__n_estimators": [150] if quick else [200, 400],
             "model__max_depth": [4, 6] if quick else [4, 6, 8],
             "model__min_samples_leaf": [5] if quick else [2, 5, 10]},
        ),
    }
    try:  # optional 4th model
        from xgboost import XGBClassifier

        cands["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=rs, n_jobs=1),
            {"model__n_estimators": [100] if quick else [100, 200],
             "model__max_depth": [2, 3] if quick else [2, 3, 4],
             "model__learning_rate": [0.1] if quick else [0.03, 0.1]},
        )
    except Exception:  # not installed -> skip silently
        pass
    return cands


def evaluate(pipe, X_test, y_test, threshold: float = RISK_THRESHOLD):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return {k: float(v) for k, v in metrics.items()}, proba, pred


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _global_shap(pipe, X_test, X_test_raw, units, reports_dir: Path, make_plots: bool,
                 background, verbose: bool) -> dict:
    """Compute global SHAP importance on the test set and save table + plots."""
    explainer = build_explainer(pipe, background)
    Xt = transform_features(pipe, X_test_raw)
    values, base = shap_for(explainer, Xt)
    gap = additivity_gap(pipe.named_steps["model"], units, values, base, Xt)
    imp = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(values).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp.to_csv(reports_dir / "global_importance.csv", index=False)

    top3 = imp["feature"].head(3).tolist()
    sanity = {
        "top_features": imp["feature"].head(5).tolist(),
        "glucose_in_top3": "Glucose" in top3,
        "bmi_or_age_in_top5": bool({"BMI", "Age"} & set(imp["feature"].head(5))),
        "additivity_max_gap": gap,
        "units": units,
    }
    if make_plots:
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()
        d = imp.iloc[::-1]
        ax.barh(d["feature"], d["mean_abs_shap"], color="#1f77b4")
        ax.set_xlabel(f"Mean |SHAP value|  ({unit_label(units)})")
        ax.set_title("Global feature importance (SHAP, test set)")
        fig.tight_layout()
        _save_fig(fig, reports_dir / "shap_global_bar.png")
        try:  # beeswarm needs pyplot; optional nicety
            import matplotlib.pyplot as plt
            import shap

            plt.figure()
            shap.summary_plot(values, X_test_raw[FEATURES], feature_names=FEATURES, show=False)
            plt.tight_layout()
            plt.savefig(reports_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:
            _log(f"  (beeswarm plot skipped: {exc})", verbose)
    return sanity


def run_training(df: pd.DataFrame | None = None, models_dir: Path | str = MODELS_DIR,
                 reports_dir: Path | str = REPORTS_DIR, quick: bool = False,
                 make_plots: bool = True, verbose: bool = True) -> dict:
    """Train, evaluate, explain and save. Returns the model bundle (also saved)."""
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset() if df is None else validate_schema(df)
    X = zeros_to_nan(df[FEATURES])
    y = df[TARGET]
    _log(f"Dataset: {len(df)} rows, {y.mean() * 100:.1f}% positive (diabetic).", verbose)
    _log(f"Missing values after treating zeros as missing:\n{X.isna().sum()[X.isna().sum() > 0].to_string()}",
         verbose)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results, fitted, probas = {}, {}, {}
    for name, (est, grid) in candidate_models(quick).items():
        _log(f"Tuning {name} ...", verbose)
        gs = GridSearchCV(build_pipeline(est), grid, scoring="roc_auc", cv=cv,
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        m, proba, _ = evaluate(gs.best_estimator_, X_test, y_test)
        results[name] = {"cv_roc_auc": float(gs.best_score_),
                         "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                         **{f"test_{k}": v for k, v in m.items()}}
        fitted[name], probas[name] = gs.best_estimator_, proba
        _log(f"  CV ROC-AUC={gs.best_score_:.3f} | test: " +
             ", ".join(f"{k}={v:.3f}" for k, v in m.items()), verbose)

    # Model selection uses ONLY cross-validation on the training set.
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best = fitted[best_name]
    _log(f"\nSelected model: {best_name}", verbose)

    metrics_best, proba_best, pred_best = evaluate(best, X_test, y_test)
    units = model_units(best.named_steps["model"])

    comp = pd.DataFrame(results).T.drop(columns="best_params")
    comp.insert(0, "model", comp.index)
    comp["selected"] = comp["model"] == best_name
    comp.to_csv(reports_dir / "model_comparison.csv", index=False)

    if make_plots:
        fig = Figure(figsize=(4.5, 4))
        ax = fig.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred_best, display_labels=["Low risk", "High risk"], ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix - {best_name}")
        _save_fig(fig, reports_dir / "confusion_matrix.png")

        fig = Figure(figsize=(5.5, 4.5))
        ax = fig.subplots()
        for name, p in probas.items():
            RocCurveDisplay.from_predictions(y_test, p, name=name, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title("ROC curves (held-out test set)")
        _save_fig(fig, reports_dir / "roc_curves.png")

    # Background sample for explainers that need one (LinearExplainer).
    Xtr_t = transform_features(best, X_train)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(Xtr_t), size=min(100, len(Xtr_t)), replace=False)
    background = Xtr_t[idx]

    sanity = None
    try:
        sanity = _global_shap(best, X_test, X_test, units, reports_dir, make_plots,
                              background, verbose)
        _log(f"SHAP top features: {sanity['top_features']} (additivity gap "
             f"{sanity['additivity_max_gap']:.2e})", verbose)
    except Exception as exc:  # shap missing / version problem: model still usable
        _log(f"WARNING: SHAP global analysis skipped: {exc}", verbose)

    tn, fp, fn, tp = confusion_matrix(y_test, pred_best, labels=[0, 1]).ravel()
    bundle = {
        "pipeline": best,
        "features": FEATURES,
        "model_name": best_name,
        "units": units,
        "threshold": RISK_THRESHOLD,
        "background": background,
        "train_ranges": {f: (float(X[f].min()), float(X[f].max())) for f in FEATURES},
        "metrics": metrics_best,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    joblib.dump(bundle, models_dir / MODEL_FILENAME)

    report = {
        "selected_model": best_name,
        "selection_rule": "highest mean ROC-AUC in 5-fold CV on the training split",
        "test_metrics_selected": metrics_best,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "models": results,
        "dataset": {"rows": int(len(df)), "positive_rate": float(y.mean()),
                    "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "shap_sanity": sanity,
        "risk_threshold": RISK_THRESHOLD,
        "created": bundle["created"],
        "versions": bundle["versions"],
    }
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=_jsonable))
    _log(f"\nSaved model  -> {models_dir / MODEL_FILENAME}", verbose)
    _log(f"Saved report -> {reports_dir}", verbose)
    return bundle
