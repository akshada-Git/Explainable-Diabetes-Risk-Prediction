# Product Requirements Document (PRD)

**Project:** Explainable Diabetes Risk Prediction
**Course:** B.Sc. Data Science — TY Mini Project

---

## 1. Problem Statement & Objective

**Problem:** Most diabetes risk prediction tools output only a binary/probability result ("high risk" / "low risk") without explaining *why*. This makes the model feel like a black box, reduces trust, and limits its usefulness for awareness or education.

**Objective:** Build a machine learning system that (a) predicts a person's diabetes risk from structured health/lifestyle data, and (b) explains the prediction in human-understandable terms using Explainable AI (XAI) techniques such as SHAP or feature importance — so the user sees both the "what" and the "why."

---

## 2. Target Users

- **Students/self-testers** — individuals curious about their diabetes risk based on basic health metrics (educational use, not diagnosis).
- **Healthcare awareness programs / NGOs** — for screening camps or awareness drives.
- **Evaluators/faculty** — assessing the project's ML + XAI implementation.
- **Future developers** — who may extend the project into a real app.

*Explicitly out of scope:* clinicians using it for actual diagnosis (must be labeled "for educational/awareness purposes only").

---

## 3. Key Features / Functional Requirements

1. **Input form** — user enters health attributes (age, BMI, glucose, blood pressure, insulin, pregnancies, etc., based on chosen dataset).
2. **Risk prediction** — model outputs risk category (e.g., High/Low) and probability score.
3. **Explanation panel** — shows top contributing factors per prediction (SHAP values or feature importance), with a simple chart (bar/waterfall plot).
4. **Global model insights** — a dashboard/section showing overall feature importance across the dataset (for report/demo purposes).
5. **Model comparison (backend)** — train/evaluate 2–3 algorithms, select the best based on metrics.
6. **Disclaimer** — persistent notice that this is not medical advice.
7. *(Optional stretch)* Downloadable PDF/report of the user's result + explanation.

---

## 4. System Workflow

```
User Input (form)
      ↓
Data Preprocessing (scaling, encoding, missing-value handling)
      ↓
Trained ML Model (.pkl) → Prediction (risk class + probability)
      ↓
XAI Module (SHAP/feature importance) → Explanation for this specific input
      ↓
Streamlit UI → Displays prediction + explanation chart + disclaimer
```

**Offline/training pipeline (done once, separately):**
Dataset → Cleaning/EDA → Feature Engineering → Train/Test Split → Train multiple models → Evaluate → Pick best model → Save model (`joblib`/`pickle`) → Load in app.

---

## 5. FREE/Open-Source Tools & Technologies

| Purpose | Tool | Notes |
|---|---|---|
| Language | Python 3.x | Free, standard for ML |
| Data handling | Pandas, NumPy | Free, open-source |
| Visualization | Matplotlib, Seaborn, Plotly | Free |
| ML | Scikit-learn | Free, open-source, GitHub: scikit-learn/scikit-learn |
| Explainability | SHAP | Free, open-source, GitHub: shap/shap |
| Notebook/dev | Jupyter Notebook / Google Colab | Free (Colab gives free GPU/CPU) |
| App/UI | Streamlit | Free, open-source, easiest for ML demos |
| Version control | Git + GitHub | Free |
| Model storage | joblib/pickle | Built into Python ecosystem |

No paid APIs or proprietary software are needed anywhere in this pipeline.

---

## 6. FREE/Open-Source LLM Models

An LLM is **not required** for the core prediction task (this is a tabular classification problem — classic ML is the right tool, not an LLM). However, if you want to add a **natural-language explanation layer** (e.g., "In plain English, why is my risk high?"), a small local/free LLM can turn SHAP output into a readable sentence.

| Model | Size | Why it fits |
|---|---|---|
| **Phi-3-mini (Microsoft)** | 3.8B | Best balance of quality and speed for a laptop; runs via Ollama/llama.cpp with 4-bit quantization (~2–3GB RAM); great at simple summarization tasks like "explain these 3 factors in one paragraph." |
| **Llama 3.2 3B (Meta)** | 3B | Free, open weights, runs locally via Ollama; good general-purpose text generation for explanation summaries. |
| **Gemma 2 2B (Google)** | 2B | Very lightweight, good for low-RAM laptops (4–8GB), decent instruction-following for short templated explanations. |

**Recommendation:** Use **Phi-3-mini via Ollama** if you want this feature — it's optional/stretch, not core. Skip the LLM entirely if time is short; SHAP charts + a rule-based text template ("Your glucose level (X) is the top contributor to your risk") achieve the same explainability goal without any LLM.

---

## 7. FREE/Open-Source AI/ML Models/Algorithms

| Category | Algorithm | Why |
|---|---|---|
| Baseline | Logistic Regression | Interpretable, fast, good baseline; coefficients are inherently explainable |
| Tree-based | Decision Tree | Naturally interpretable, visualize the tree itself |
| Tree-based (recommended primary) | **Random Forest** | Strong accuracy on tabular health data, robust to noise, works very well with SHAP |
| Boosting (optional upgrade) | XGBoost / LightGBM | Free, open-source, often best accuracy on tabular data, SHAP has first-class support (`TreeExplainer`) |
| Deep learning (optional, not necessary) | Simple MLP (Keras/TensorFlow or PyTorch) | Only worth it if dataset is large; adds complexity without much accuracy gain on small tabular datasets — mention in report as "explored but not primary" if you want to show DL awareness |

**Recommendation:** Train Logistic Regression, Decision Tree, Random Forest, and (optionally) XGBoost. Pick the best by evaluation metrics. Use **Random Forest or XGBoost** as your final model — both pair well with SHAP's fast `TreeExplainer`.

---

## 8. Recommended Tech Stack (single primary stack)

| Layer | Choice |
|---|---|
| **Frontend/UI** | Streamlit (Python-only, no separate frontend needed) |
| **Backend** | Streamlit's built-in Python backend (no separate Flask/FastAPI server needed for MVP) |
| **Database** | Not required for MVP (stateless predictions); if you want to log user submissions, use **SQLite** (free, file-based, zero setup) |
| **AI/ML** | Scikit-learn (Random Forest / XGBoost) |
| **Explainability** | SHAP |
| **LLM (optional)** | Phi-3-mini via Ollama (local) |
| **APIs** | None required — everything runs locally |
| **Deployment** | Streamlit Community Cloud (free tier) or local `streamlit run app.py` for demo |

This single stack keeps the project laptop-friendly, free, and simple to explain in a viva/demo — everything is Python end-to-end.

---

## 9. Dataset Requirements & Sources

- **Primary recommendation: PIMA Indians Diabetes Dataset**
  - Source: UCI Machine Learning Repository / Kaggle (`uciml/pima-indians-diabetes-database`)
  - ~768 rows, 8 features (Glucose, BMI, Blood Pressure, Insulin, Age, Pregnancies, Skin Thickness, Diabetes Pedigree Function) + binary target.
  - Free, widely used in academic projects, small enough to run on any laptop.
- **Alternative (larger, more diverse):** CDC Diabetes Health Indicators Dataset (Kaggle/UCI) — larger (~250k+ rows), more features, better if you want to show scale.
- **Requirements:** CSV format, target column labeled (diabetic/non-diabetic), needs cleaning (PIMA has disguised missing values as 0s in Glucose/BMI/BloodPressure — must handle this).

---

## 10. Evaluation Metrics

- **Accuracy** — overall correctness (use cautiously; dataset may be imbalanced)
- **Precision, Recall, F1-score** — important for medical-risk context (false negatives = missed risk = more costly)
- **ROC-AUC** — overall discriminative ability across thresholds
- **Confusion Matrix** — visualize TP/FP/TN/FN
- **SHAP consistency check** — sanity-check that top SHAP features align with known medical risk factors (e.g., glucose, BMI) to validate explainability quality, not just accuracy

---

## 11. MVP Scope

- Clean and preprocess PIMA dataset
- Train Logistic Regression, Decision Tree, Random Forest; select best by F1/ROC-AUC
- Generate SHAP explanations (local, per-prediction + global feature importance)
- Streamlit app: input form → prediction + probability → SHAP bar chart → disclaimer text
- Basic model evaluation report (metrics + confusion matrix) included in the project report

**Out of MVP (defer):** LLM-based natural-language explanations, database logging, deployment to cloud with authentication, mobile app.

---

## 12. Future Enhancements

- Natural-language explanation layer using a small local LLM (Phi-3-mini/Llama 3.2 3B)
- Larger/more diverse dataset (e.g., CDC dataset) for better generalization
- Deep learning model comparison (MLP) if dataset size grows
- Web/mobile app version with user accounts and history tracking
- Personalized recommendations (diet/lifestyle tips) tied to top risk factors
- Multi-disease risk screening (extend beyond diabetes)
- Proper privacy/security handling if real user data is ever collected

---

## 13. Feasibility & Estimated Resources

- **Hardware:** Runs fully on a standard laptop (8GB RAM sufficient; no GPU required for classic ML; optional LLM feature needs ~4–8GB free RAM if used).
- **Cost:** ₹0 — every tool, library, dataset, and model listed is free and open-source.
- **Time estimate (TY mini-project scope):**
  - Data collection + EDA + preprocessing: 3–4 days
  - Model training + evaluation: 3–4 days
  - SHAP integration: 2–3 days
  - Streamlit UI: 3–4 days
  - Testing, report writing, polishing: 4–5 days
  - **Total: ~3–4 weeks** of part-time work — realistic for a TY mini-project timeline.
- **Skill prerequisites:** Basic Python, Pandas/Scikit-learn familiarity — all learnable via free documentation/tutorials during the project itself.
- **Risk factors:** Low — dataset is small and well-known, no paid dependencies, no server infrastructure needed for MVP.
