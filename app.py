"""Explainable Diabetes Risk Prediction - Streamlit app.

Run locally:   streamlit run app.py
"""
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from src.config import (DISCLAIMER, FEATURE_INFO, FEATURES, MODEL_PATH, REPORTS_DIR)
from src.explain import (build_contributions, build_explainer, contribution_figure,
                         figure_to_png_bytes, plain_language_summary, shap_for, unit_label)
from src.predict import predict_one, validate_inputs
from src.preprocessing import transform_features
from src.report import build_pdf

st.set_page_config(page_title="Explainable Diabetes Risk Prediction",
                   page_icon="\U0001FA7A", layout="wide")


# ---------------------------------------------------------------- loading ---
def _self_test(bundle: dict) -> None:
    """Make sure a loaded model really works (catches version-mismatch pickles)."""
    values = {f: FEATURE_INFO[f]["default"] for f in FEATURES}
    out = predict_one(bundle, values)
    assert 0.0 <= out["probability"] <= 1.0


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts() -> dict:
    """Load the saved model + explainer. Retrains automatically if the file is
    missing or unusable (e.g. built with a different scikit-learn version)."""
    notes, bundle = [], None
    try:
        bundle = joblib.load(MODEL_PATH)
        _self_test(bundle)
    except Exception as exc:  # EH-3: never crash on a bad/missing model file
        notes.append(f"The saved model could not be used ({type(exc).__name__}). "
                     "Training a fresh one - this takes about a minute.")
        bundle = None

    if bundle is None:
        try:
            from src.training import run_training

            bundle = run_training(quick=True, make_plots=True, verbose=False)
            _self_test(bundle)
        except Exception as exc:
            return {"error": f"Model file not found and automatic training failed: {exc}"}

    explainer, explain_error = None, None
    try:
        explainer = build_explainer(bundle["pipeline"], bundle["background"])
    except Exception as exc:
        explain_error = str(exc)
    return {"bundle": bundle, "explainer": explainer, "notes": notes,
            "explain_error": explain_error}


def read_report() -> dict | None:
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


# ------------------------------------------------------------ computation ---
def compute_result(art: dict, values: dict) -> dict:
    bundle = art["bundle"]
    pred = predict_one(bundle, values)
    result = {
        "inputs": values,
        "probability": pred["probability"],
        "label": pred["label"],
        "is_high": pred["is_high"],
        "contrib": None,
        "chart_png": None,
        "summary": "",
        "explain_error": None,
        "pdf": None,
        "range_warnings": [],
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    for f, (lo, hi) in bundle.get("train_ranges", {}).items():
        v = values.get(f)
        if v is not None and (v < lo or v > hi):
            unit = f" {FEATURE_INFO[f]['unit']}" if FEATURE_INFO[f]["unit"] else ""
            result["range_warnings"].append(
                f"{FEATURE_INFO[f]['label']} = {v:g}{unit} is outside the range the model was "
                f"trained on ({lo:g} to {hi:g}); treat the result with extra caution.")

    # EH-4: if SHAP fails, still show the prediction.
    try:
        if art["explainer"] is None:
            raise RuntimeError(art.get("explain_error") or "explainer unavailable")
        Xt = transform_features(bundle["pipeline"], pred["frame"])
        sv, _ = shap_for(art["explainer"], Xt)
        contrib = build_contributions(bundle["pipeline"], pred["frame"], sv[0])
        fig = contribution_figure(contrib, bundle["units"])
        result["contrib"] = contrib
        result["chart_png"] = figure_to_png_bytes(fig)
        result["summary"] = plain_language_summary(pred["probability"], pred["label"], contrib)
    except Exception as exc:
        result["explain_error"] = str(exc)

    try:
        result["pdf"] = build_pdf(result, result["contrib"], result["chart_png"],
                                  bundle["model_name"], result["summary"])
    except Exception:
        result["pdf"] = None
    return result


# --------------------------------------------------------------- rendering --
def render_form() -> tuple[bool, dict]:
    values = {}
    with st.form("input_form"):
        st.markdown("Enter the health measurements below, then press **Predict**.")
        cols = st.columns(2)
        for i, feat in enumerate(FEATURES):
            info = FEATURE_INFO[feat]
            unit = f" ({info['unit']})" if info["unit"] else ""
            with cols[i % 2]:
                if info["kind"] == "int":
                    v = st.number_input(info["label"] + unit, min_value=int(info["min"]),
                                        max_value=int(info["max"]), value=int(info["default"]),
                                        step=int(info["step"]), help=info["help"],
                                        key=f"in_{feat}")
                else:
                    fmt = "%.2f" if info["step"] < 0.1 else "%.1f"
                    v = st.number_input(info["label"] + unit, min_value=float(info["min"]),
                                        max_value=float(info["max"]), value=float(info["default"]),
                                        step=float(info["step"]), format=fmt, help=info["help"],
                                        key=f"in_{feat}")
                unknown = False
                if info["optional"]:
                    unknown = st.checkbox("I don't know / not measured", key=f"unk_{feat}")
                values[feat] = None if unknown else v
        submitted = st.form_submit_button("Predict", type="primary")
    return submitted, values


def render_result(result: dict, art: dict) -> None:
    bundle = art["bundle"]
    st.divider()
    st.subheader("Result")
    for w in result["range_warnings"]:
        st.warning(w)

    c1, c2 = st.columns(2)
    c1.metric("Risk category", result["label"])
    c2.metric("Estimated probability", f"{result['probability'] * 100:.1f}%")
    st.progress(min(max(result["probability"], 0.0), 1.0))
    if result["is_high"]:
        st.error("**High Risk** - the model estimates an elevated chance of diabetes based on "
                 "these values. This is not a diagnosis; please consider talking to a doctor "
                 "and getting a proper blood test.")
    else:
        st.success("**Low Risk** - the model estimates a lower chance of diabetes based on these "
                   "values. This does not mean zero risk, and it is not a diagnosis.")

    st.subheader("Why this result?")
    if result["contrib"] is None:
        st.info("Explanation temporarily unavailable.")  # EH-4
    else:
        st.write(result["summary"])
        st.image(result["chart_png"])
        with st.expander("Show contribution table"):
            tbl = result["contrib"][["label", "value_text", "impact", "direction"]].copy()
            tbl.columns = ["Feature", "Your value", unit_label(bundle["units"]), "Effect"]
            st.dataframe(tbl)
        if bundle["units"] == "probability":
            st.caption("Bars show how many probability points each factor moved the estimate away "
                       "from the average person in the training data.")
        else:
            st.caption("Bars are in log-odds (the scale Logistic Regression works on): positive "
                       "values raise the risk estimate, negative values lower it.")

    if result["pdf"]:
        st.download_button("Download report (PDF)", data=result["pdf"],
                           file_name="diabetes_risk_report.pdf", mime="application/pdf")


def render_global(art: dict) -> None:
    st.subheader("Which factors matter most overall?")
    csv = REPORTS_DIR / "global_importance.csv"
    if not csv.exists():
        st.info("Global insights are not available yet. Run `python train.py` and commit the "
                "`reports/` folder.")
        return
    st.write("Average absolute SHAP value of each feature on the held-out test set. "
             "Larger = the model relies on that feature more.")
    bar = REPORTS_DIR / "shap_global_bar.png"
    if bar.exists():
        st.image(str(bar))
    summ = REPORTS_DIR / "shap_summary.png"
    if summ.exists():
        st.markdown("**Beeswarm view** - each dot is one person; red = high feature value, blue = low.")
        st.image(str(summ))
    st.dataframe(pd.read_csv(csv))
    rep = read_report()
    san = rep.get("shap_sanity") if rep else None
    if san:
        ok = "yes" if san.get("glucose_in_top3") else "no"
        st.info(f"Sanity check: top features are {', '.join(san['top_features'])}. "
                f"Glucose in the top 3 (a well-known clinical risk factor): **{ok}**.")


def render_performance(art: dict) -> None:
    st.subheader("Model comparison and evaluation")
    rep = read_report()
    comp = REPORTS_DIR / "model_comparison.csv"
    if not rep or not comp.exists():
        st.info("Evaluation report not found. Run `python train.py` and commit the `reports/` folder.")
        return
    st.write(f"**Selected model:** {rep['selected_model']}  \n"
             f"**Selection rule:** {rep['selection_rule']}. The test set was used only once, "
             "for the numbers below.")
    m = rep["test_metrics_selected"]
    cols = st.columns(5)
    for col, (k, name) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                     ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]):
        col.metric(name, f"{m[k]:.3f}")
    df = pd.read_csv(comp)
    st.dataframe(df)
    c1, c2 = st.columns(2)
    if (REPORTS_DIR / "confusion_matrix.png").exists():
        c1.image(str(REPORTS_DIR / "confusion_matrix.png"))
    if (REPORTS_DIR / "roc_curves.png").exists():
        c2.image(str(REPORTS_DIR / "roc_curves.png"))
    st.caption("Recall matters most in a screening context: a false negative is a missed at-risk person.")


def render_about() -> None:
    st.subheader("About this project")
    st.markdown(
        """
**Explainable Diabetes Risk Prediction** - B.Sc. Data Science TY mini project.

* **Data:** PIMA Indians Diabetes Dataset (768 women, 8 measurements, diabetic yes/no).
  Zeros in Glucose, Blood pressure, Skin thickness, Insulin and BMI are treated as missing
  and filled with the training median.
* **Models compared:** Logistic Regression, Decision Tree, Random Forest (XGBoost if installed),
  tuned with 5-fold cross-validation; the best by cross-validated ROC-AUC is used.
* **Explainability:** SHAP shows how much each of your values pushed the estimate up or down.
* **Limitations:** small dataset from one population (adult women of Pima heritage, aged 21+),
  so results may not generalise to everyone. The model finds statistical patterns, not causes.
* **Privacy:** nothing you type is stored or sent anywhere; predictions are computed in this app.
        """
    )
    st.warning(DISCLAIMER)


# ------------------------------------------------------------------- main ---
st.title("\U0001FA7A Explainable Diabetes Risk Prediction")
st.warning(DISCLAIMER)  # FR-6.2 - always visible

art = load_artifacts()
if "error" in art:
    st.error(art["error"])  # EH-3
    st.stop()
for note in art["notes"]:
    st.info(note)

with st.sidebar:
    st.header("Model")
    b = art["bundle"]
    st.write(f"**{b['model_name']}**")
    st.write(f"Test ROC-AUC: {b['metrics']['roc_auc']:.3f}")
    st.write(f"Test recall: {b['metrics']['recall']:.3f}")
    st.caption("Educational project. Not for diagnosis.")

tab_predict, tab_global, tab_perf, tab_about = st.tabs(
    ["Predict", "Global insights", "Model performance", "About"])

with tab_predict:
    submitted, values = render_form()
    if submitted:
        errors = validate_inputs(values)
        if errors:
            st.session_state.pop("result", None)
            for e in errors:
                st.error(e)  # EH-1 / EH-2
        else:
            with st.spinner("Calculating..."):
                st.session_state["result"] = compute_result(art, values)
    if st.session_state.get("result"):
        render_result(st.session_state["result"], art)

with tab_global:
    render_global(art)
with tab_perf:
    render_performance(art)
with tab_about:
    render_about()

st.divider()
st.caption(DISCLAIMER)
