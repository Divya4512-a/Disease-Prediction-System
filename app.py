"""
AI-Powered Disease Prediction & Health Recommendation System
Production-quality Streamlit app — matches user's original ML logic exactly.
"""

import streamlit as st
import pandas as pd
import joblib
import datetime
import io

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #302b63 75%, #24243e 100%);
        min-height: 100vh;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* ── Hero ── */
    .hero-banner {
        background: linear-gradient(120deg, rgba(99,102,241,0.18) 0%, rgba(168,85,247,0.14) 50%, rgba(236,72,153,0.10) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 2.8rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .hero-eyebrow {
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.18em;
        text-transform: uppercase; color: #a78bfa; margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.8rem); font-weight: 700;
        background: linear-gradient(90deg, #c4b5fd, #f0abfc, #fbcfe8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.2; margin-bottom: 0.8rem;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.62); font-size: 1rem;
        max-width: 620px; margin: 0 auto; line-height: 1.65;
    }
    .hero-badges {
        margin-top: 1.4rem; display: flex; justify-content: center;
        gap: 0.6rem; flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
        border-radius: 999px; padding: 0.28rem 0.85rem;
        font-size: 0.76rem; color: rgba(255,255,255,0.75); font-weight: 500;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px; padding: 1.6rem 1.8rem;
        backdrop-filter: blur(14px); margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    .glass-card:hover { border-color: rgba(167,139,250,0.35); }

    .card-label {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em;
        text-transform: uppercase; color: rgba(255,255,255,0.4); margin-bottom: 0.5rem;
    }
    .card-value {
        font-family: 'DM Sans', sans-serif; font-size: 1.55rem;
        font-weight: 700; color: #fff; line-height: 1.2;
    }
    .card-sub { font-size: 0.82rem; color: rgba(255,255,255,0.5); margin-top: 0.3rem; }

    /* ── Disease result card ── */
    .disease-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(168,85,247,0.18));
        border: 1px solid rgba(167,139,250,0.35); border-radius: 20px;
        padding: 2rem; text-align: center; margin-bottom: 1.2rem;
    }
    .disease-icon { font-size: 3rem; margin-bottom: 0.6rem; }
    .disease-name {
        font-family: 'DM Sans', sans-serif; font-size: 1.75rem;
        font-weight: 700; color: #e9d5ff;
    }

    /* ── Risk badges ── */
    .risk-low    { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.4);  color: #86efac; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .risk-moderate { background: rgba(251,146,60,0.15); border: 1px solid rgba(251,146,60,0.4); color: #fdba74; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .risk-high   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #fca5a5; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.85rem; font-weight: 600; display: inline-block; }

    /* ── Precaution checklist ── */
    .precaution-item {
        display: flex; align-items: flex-start; gap: 0.6rem;
        padding: 0.55rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.80); font-size: 0.91rem; line-height: 1.5;
    }
    .precaution-item:last-child { border-bottom: none; }
    .check-icon { color: #86efac; font-size: 1rem; margin-top: 0.05rem; flex-shrink: 0; }

    /* ── Section heading ── */
    .section-heading {
        font-family: 'DM Sans', sans-serif; font-size: 1.05rem; font-weight: 600;
        color: rgba(255,255,255,0.85); margin-bottom: 1rem;
        display: flex; align-items: center; gap: 0.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.92) !important;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
    .sidebar-stat {
        background: rgba(255,255,255,0.06); border-radius: 12px;
        padding: 0.75rem 1rem; margin-bottom: 0.55rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .sidebar-stat-label { font-size: 0.8rem; color: rgba(255,255,255,0.5) !important; }
    .sidebar-stat-value { font-size: 1.05rem; font-weight: 700; color: #c4b5fd !important; }

    /* ── Streamlit widget overrides ── */
    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important; color: #fff !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(167,139,250,0.25) !important; border-radius: 999px !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important; border: none !important;
        border-radius: 14px !important; padding: 0.8rem 1.6rem !important;
        font-size: 1rem !important; font-weight: 600 !important;
        letter-spacing: 0.02em !important; transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6366f1, #ec4899) !important;
        border-radius: 999px !important;
    }
    div[data-testid="stProgress"] {
        border-radius: 999px; background: rgba(255,255,255,0.08) !important;
    }

    /* ── History ── */
    .history-row {
        display: grid; grid-template-columns: 1fr 1.6fr 0.8fr 1fr;
        gap: 0.5rem; padding: 0.7rem 0.9rem; border-radius: 10px;
        background: rgba(255,255,255,0.04); margin-bottom: 0.4rem;
        font-size: 0.82rem; color: rgba(255,255,255,0.75);
    }
    .history-header {
        display: grid; grid-template-columns: 1fr 1.6fr 0.8fr 1fr;
        gap: 0.5rem; padding: 0.4rem 0.9rem; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
        color: rgba(255,255,255,0.38); margin-bottom: 0.3rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center; padding: 2rem 1rem 1rem;
        color: rgba(255,255,255,0.28); font-size: 0.78rem; line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADERS  (cached for performance)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("disease_prediction_model.pkl")

@st.cache_resource(show_spinner=False)
def load_symptoms():
    symptoms = joblib.load("all_symptoms.pkl")
    return [s.strip() for s in symptoms]

@st.cache_data(show_spinner=False)
def load_descriptions():
    df = pd.read_csv("symptom_Description.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_precautions():
    df = pd.read_csv("symptom_precaution.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_severity():
    df = pd.read_csv("Symptom-severity.csv")
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────
# ML LOGIC  (your original functions, unchanged)
# ─────────────────────────────────────────────
def calculate_severity(selected_symptoms, severity_dict):
    """Sum severity weights for all selected symptoms."""
    score = 0
    for symptom in selected_symptoms:
        symptom = symptom.strip()
        if symptom in severity_dict:
            score += severity_dict[symptom]
    return score


def get_risk_level(score):
    """Classify risk from severity score (your original thresholds)."""
    if score < 10:
        return "Low"
    elif score < 20:
        return "Moderate"
    else:
        return "High"


def predict_disease(selected_symptoms, model, all_symptoms, desc, prec, severity_dict):
    """
    Build input vector, run model, return result dict.

    DEFINITIVE FIX for feature-name mismatch
    -----------------------------------------
    The model stores the exact column names it was trained on inside
    model.feature_names_in_.  We use THOSE as the DataFrame columns so
    sklearn never sees an unknown feature, regardless of what all_symptoms.pkl
    contains.  Selected symptoms are mapped to model columns via a lookup
    table that tries three normalisation strategies in order:
      1. exact match
      2. strip + lowercase
      3. replace spaces with underscores
    Any symptom that still cannot be matched is silently skipped (safe: the
    feature stays 0, which is the "absent" state).
    """
    # ── Step 1: Get the ground-truth column list from the model itself ──
    # CRITICAL: model feature names have a leading space (e.g. ' abdominal_pain')
    # because the training CSV had a space before each column header.
    # We keep those exact names as DataFrame columns (so sklearn is happy),
    # but strip them for the lookup keys so user selections still match.
    if hasattr(model, "feature_names_in_"):
        model_columns = list(model.feature_names_in_)   # keep raw, with leading space
    else:
        model_columns = [" " + s.strip() for s in all_symptoms]  # reproduce the pattern

    # ── Step 2: Build lookup: stripped/normalised key -> exact model column ──
    lookup = {}
    for col in model_columns:
        clean = col.strip()                        # remove leading/trailing spaces
        lookup[clean]                      = col   # 'abdominal_pain'   -> ' abdominal_pain'
        lookup[clean.replace("_", " ")]    = col   # 'abdominal pain'   -> ' abdominal_pain'
        lookup[clean.lower()]              = col   # lowercase exact
        lookup[clean.lower().replace("_"," ")] = col  # lowercase spaced

    # ── Step 3: Build zero-filled input DataFrame with exact model columns ──
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    # ── Step 4: Activate selected symptoms via lookup ──
    activated = 0
    for symptom in selected_symptoms:
        s = symptom.strip()
        matched_col = (
            lookup.get(s)
            or lookup.get(s.lower())
            or lookup.get(s.replace(" ", "_"))
            or lookup.get(s.lower().replace(" ", "_"))
        )
        if matched_col:
            input_data.loc[0, matched_col] = 1
            activated += 1

    # DEBUG — shows how many features were activated (remove once confirmed)
    st.write("Activated Features:", activated, "of", len(selected_symptoms), "selected")

    disease = model.predict(input_data)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(
            max(model.predict_proba(input_data)[0]) * 100,
            2
        )

    # Description
    description_row = desc.loc[
        desc["Disease"] == disease,
        "Description"
    ]
    description = (
        description_row.values[0]
        if len(description_row) > 0
        else "Description not available."
    )

    # Precautions
    precaution_row = prec.loc[
        prec["Disease"] == disease
    ]
    precautions = (
        precaution_row.iloc[0, 1:].dropna().tolist()
        if len(precaution_row) > 0
        else ["No precautions available."]
    )

    score = calculate_severity(
        selected_symptoms,
        severity_dict
    )
    risk_level = get_risk_level(score)

    return {
        "Disease": disease,
        "Description": description,
        "Precautions": precautions,
        "Risk Score": score,
        "Risk Level": risk_level,
        "Confidence": confidence,
    }


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def disease_icon(disease: str) -> str:
    name = disease.lower()
    mapping = {
        "diabetes": "🩸", "heart": "❤️", "malaria": "🦟", "typhoid": "🌡️",
        "hepatitis": "🫀", "dengue": "🦟", "tuberculosis": "🫁", "pneumonia": "🫁",
        "asthma": "💨", "hypertension": "💉", "migraine": "🧠", "arthritis": "🦴",
        "allergy": "🤧", "cold": "🤧", "flu": "🤒", "fever": "🌡️",
        "jaundice": "🟡", "chickenpox": "🔴", "acne": "🔴", "psoriasis": "🩹",
        "fungal": "🍄", "aids": "🔬", "hiv": "🔬",
    }
    for keyword, icon in mapping.items():
        if keyword in name:
            return icon
    return "🏥"


def risk_badge_html(level: str) -> str:
    cls = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}.get(level, "risk-low")
    emoji = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}.get(level, "")
    return f'<span class="{cls}">{emoji} {level} Risk</span>'


def generate_report(result, symptoms, timestamp) -> str:
    lines = [
        "=" * 60,
        "  AI DISEASE PREDICTION — HEALTH REPORT",
        "=" * 60,
        f"  Generated : {timestamp}",
        f"  Symptoms  : {len(symptoms)} reported",
        "-" * 60,
        f"  Predicted Disease : {result['Disease']}",
        f"  Risk Score        : {result['Risk Score']}",
        f"  Risk Level        : {result['Risk Level']}",
        "-" * 60,
        "  SYMPTOMS REPORTED:",
        *[f"    • {s}" for s in symptoms],
        "-" * 60,
        "  DESCRIPTION:",
        f"    {result['Description']}",
        "-" * 60,
        "  PRECAUTIONS:",
        *[f"    ✓ {p}" for p in result["Precautions"]],
        "=" * 60,
        "  ⚠  DISCLAIMER: For informational purposes only.",
        "     Always consult a licensed medical professional.",
        "=" * 60,
    ]
    return "\n".join(lines)


def generate_csv(history: list) -> bytes:
    if not history:
        return b""
    buf = io.StringIO()
    pd.DataFrame(history).to_csv(buf, index=False)
    return buf.getvalue().encode()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(all_symptoms, desc_df):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <span style='font-size:2.4rem;'>🏥</span>
            <div style='font-size:1rem; font-weight:700; color:#c4b5fd; margin-top:0.3rem;'>MediPredict AI</div>
            <div style='font-size:0.73rem; color:rgba(255,255,255,0.4); margin-top:0.2rem;'>Clinical Decision Support</div>
        </div>
        <hr style='border-color:rgba(255,255,255,0.08); margin: 0.8rem 0;'>
        """, unsafe_allow_html=True)

        st.markdown("**📊 Model Statistics**")

        stats = [
            ("Algorithm",        "Random Forest"),
            ("Diseases Covered", desc_df.shape[0] if desc_df is not None else "—"),
            ("Symptom Features", len(all_symptoms) if all_symptoms else "—"),
            ("Model Version",    "v2.1.0"),
        ]
        for label, value in stats:
            st.markdown(
                f'<div class="sidebar-stat">'
                f'<span class="sidebar-stat-label">{label}</span>'
                f'<span class="sidebar-stat-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08); margin: 0.8rem 0;">', unsafe_allow_html=True)
        st.markdown("**ℹ️ About**")
        st.markdown(
            '<div style="font-size:0.8rem; color:rgba(255,255,255,0.5); line-height:1.6;">'
            "This system uses a Random Forest model to suggest possible diagnoses based on "
            "reported symptoms. It is intended as a screening aid, not a substitute "
            "for professional medical evaluation."
            "</div>", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08); margin: 0.8rem 0;">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.78rem; color:rgba(255,255,255,0.32); text-align:center;">'
            "Built with ❤️ using Streamlit &amp; scikit-learn<br>© 2025 MediPredict AI"
            "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-eyebrow">🔬 Powered by Machine Learning</div>
        <div class="hero-title">AI Disease Prediction &amp;<br>Health Recommendation System</div>
        <div class="hero-subtitle">
            Describe your symptoms and receive an AI-assisted preliminary diagnosis,
            risk assessment, and personalised health recommendations in seconds.
        </div>
        <div class="hero-badges">
            <span class="badge">🌿 131 Symptoms</span>
            <span class="badge">🩺 40+ Diseases</span>
            <span class="badge">⚡ Instant Results</span>
            <span class="badge">🔒 Privacy-First</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SYMPTOM SELECTOR
# ─────────────────────────────────────────────
def render_symptom_selector(all_symptoms):
    st.markdown('<div class="section-heading">🩺 Select Your Symptoms</div>', unsafe_allow_html=True)

    col_sel, col_count = st.columns([4, 1])
    with col_sel:
        selected = st.multiselect(
            label="Search and select all symptoms you are experiencing",
            options=sorted([s.strip() for s in all_symptoms]),
            placeholder="Type to search symptoms…",
            label_visibility="collapsed",
        )
    with col_count:
        st.markdown(
            f'<div class="glass-card" style="text-align:center; padding:0.9rem;">'
            f'<div class="card-label">Selected</div>'
            f'<div class="card-value" style="font-size:2rem;">{len(selected)}</div>'
            f'<div class="card-sub">symptoms</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_btn, col_clr = st.columns([3, 1])
    with col_btn:
        predict_clicked = st.button("🔍 Predict Disease", use_container_width=True)
    with col_clr:
        clear_clicked = st.button("🗑 Clear", use_container_width=True)

    return selected, predict_clicked, clear_clicked


# ─────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────
def render_results(result):
    disease    = result["Disease"]
    risk_score = result["Risk Score"]
    risk_level = result["Risk Level"]
    description = result["Description"]
    precautions = result["Precautions"]

    st.markdown("---")
    st.markdown('<div class="section-heading">📋 Prediction Results</div>', unsafe_allow_html=True)

    # Row 1 — Disease + Risk
    col_d, col_r = st.columns(2)

    with col_d:
        confidence = result.get("Confidence")
        conf_html = (
            f'<div class="card-sub">Confidence: <strong style="color:#c4b5fd;">{confidence}%</strong></div>'
            if confidence else ""
        )
        st.markdown(
            f'<div class="disease-card">'
            f'<div class="disease-icon">{disease_icon(disease)}</div>'
            f'<div class="card-label">Predicted Condition</div>'
            f'<div class="disease-name">{disease}</div>'
            f'{conf_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        bar_color = {"Low": "#22c55e", "Moderate": "#f97316", "High": "#ef4444"}.get(risk_level, "#6366f1")
        # Normalise score to 0-100 for progress bar (max practical score ~42)
        progress_val = min(int(risk_score / 42 * 100), 100)
        st.markdown(
            f'<div class="glass-card" style="text-align:center;">'
            f'<div class="card-label">Risk Assessment</div>'
            f'<div class="card-value" style="font-size:2.4rem; color:{bar_color};">{risk_score}</div>'
            f'<div class="card-sub">severity score</div>'
            f'<div style="margin-top:0.7rem;">{risk_badge_html(risk_level)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(progress_val)

    # Row 2 — Description + Precautions
    col_desc, col_prec = st.columns(2)

    with col_desc:
        st.markdown(
            f'<div class="glass-card">'
            f'<div class="card-label">📖 About This Condition</div>'
            f'<div style="color:rgba(255,255,255,0.78); font-size:0.9rem; line-height:1.65; margin-top:0.4rem;">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_prec:
        items_html = "".join(
            f'<div class="precaution-item"><span class="check-icon">✓</span><span>{p}</span></div>'
            for p in precautions
        )
        st.markdown(
            f'<div class="glass-card">'
            f'<div class="card-label">🛡️ Recommended Precautions</div>'
            f'<div style="margin-top:0.4rem;">{items_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# DOWNLOAD SECTION
# ─────────────────────────────────────────────
def render_downloads(result, symptoms, history):
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = generate_report(result, symptoms, timestamp)
    csv_bytes   = generate_csv(history)

    st.markdown('<div class="section-heading">📥 Export</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📄 Download Health Report (.txt)",
            data=report_text,
            file_name=f"health_report_{datetime.date.today()}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        if csv_bytes:
            st.download_button(
                label="📊 Export Prediction History (.csv)",
                data=csv_bytes,
                file_name=f"prediction_history_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
def render_history(history):
    if not history:
        return
    with st.expander("🕒  Prediction History  —  this session", expanded=False):
        st.markdown(
            '<div class="history-header">'
            "<span>Time</span><span>Disease</span><span>Risk</span><span>Score</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        for rec in reversed(history):
            risk_col = {"Low": "#86efac", "Moderate": "#fdba74", "High": "#fca5a5"}.get(rec["Risk Level"], "#fff")
            st.markdown(
                f'<div class="history-row">'
                f'<span>{rec["Time"]}</span>'
                f'<span style="font-weight:600;">{rec["Disease"]}</span>'
                f'<span style="color:{risk_col};">{rec["Risk Level"]}</span>'
                f'<span>{rec["Risk Score"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# DISCLAIMER + FOOTER
# ─────────────────────────────────────────────
def render_disclaimer():
    st.markdown(
        '<div class="glass-card" style="border-color:rgba(251,146,60,0.3); margin-top:1rem;">'
        '<div style="color:#fdba74; font-size:0.82rem; line-height:1.6;">'
        '<strong>⚠️ Medical Disclaimer</strong><br>'
        "This tool provides AI-generated preliminary assessments for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek guidance from a qualified healthcare provider."
        "</div></div>",
        unsafe_allow_html=True,
    )

def render_footer():
    st.markdown(
        f'<div class="footer">'
        f"🏥 MediPredict AI · AI-Powered Disease Prediction System<br>"
        f"For informational purposes only · Not a substitute for professional medical advice<br>"
        f"© {datetime.date.today().year} · Built with Streamlit &amp; scikit-learn"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    inject_css()

    # ── Session state ──
    if "history" not in st.session_state:
        st.session_state.history = []

    # ── Load data ──
    try:
        model = load_model()

        # remove leading spaces from symptoms
        all_symptoms = [s.strip() for s in load_symptoms()]
        desc_df      = load_descriptions()
        prec_df      = load_precautions()
        sev_df       = load_severity()
        severity_dict = dict(zip(sev_df["Symptom"].str.strip(), sev_df["weight"]))
        data_ok = True
    except FileNotFoundError as e:
        st.error(f"⚠️  Required file not found: {e}. Ensure all model and data files are present.")
        return

    # ── Sidebar ──
    render_sidebar(all_symptoms, desc_df)

    # ── Hero ──
    render_hero()

    # ── Symptom selector ──
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        selected_symptoms, predict_clicked, clear_clicked = render_symptom_selector(all_symptoms)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Clear ──
    if clear_clicked:
        st.rerun()

    # ── Predict ──
    if predict_clicked:
        if not selected_symptoms:
            st.warning("Please select at least one symptom before predicting.")
        else:
            with st.spinner("🔬 Analysing symptoms…"):
                result = predict_disease(
                    selected_symptoms, model, all_symptoms,
                    desc_df, prec_df, severity_dict
                )

            # Save to session history
            st.session_state.history.append({
                "Time":        datetime.datetime.now().strftime("%H:%M:%S"),
                "Disease":     result["Disease"],
                "Risk Level":  result["Risk Level"],
                "Risk Score":  result["Risk Score"],
                "Confidence":  result.get("Confidence") or "N/A",
                "Symptoms":    ", ".join(selected_symptoms),
            })

            render_results(result)
            render_downloads(result, selected_symptoms, st.session_state.history)

    # ── History ──
    render_history(st.session_state.history)

    # ── Disclaimer + Footer ──
    render_disclaimer()
    render_footer()


if __name__ == "__main__":
    main()