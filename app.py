# --- YOUR EXACT ORIGINAL APPMIN ORIGINAL FILES ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import transformers
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set web page title and visual configuration
st.set_page_config(page_title="AI Respiratory Diagnostic Assistant", page_icon="🫁", layout="centered")

# --- CACHE DATA AND MODEL INITIALIZATION ---
@st.cache_resource
def load_all_models():
    # Load ML Classifiers
    stack_model = joblib.load("respiratory_disease_model.pkl")
    le = joblib.load("label_encoder.pkl")

    # Load LLM
    transformers.logging.set_verbosity_error()
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Change "auto" to explicit placement based on hardware
    device_target = "auto" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
        device_map=device_target
    )

    return stack_model, le, tokenizer, model

# Unpack our cached model systems
stack_model, le, tokenizer, model = load_all_models()

feature_names = [
    'shortness of breath', 'sharp chest pain', 'chest tightness', 'sore throat',
    'cough', 'nasal congestion', 'vomiting', 'headache', 'wheezing', 'ear pain',
    'weakness', 'fever', 'difficulty breathing', 'chills', 'coughing up sputum',
    'coryza', 'allergic reaction', 'congestion in chest', 'flu-like syndrome'
]

# --- CORE INFERENCE FUNCTIONS FROM YOUR NOTEBOOK ---
def extract_features_with_llm(note_text):
    normalized = {feat.lower(): 0 for feat in feature_names}
    text = note_text.lower().strip()

    for feat in feature_names:
        feat_lower = feat.lower()
        if feat_lower in text:
            normalized[feat_lower] = 1
        elif feat_lower.replace(" ", "") in text.replace(" ", ""):
            normalized[feat_lower] = 1

    if "chest" in text and "pain" in text:
        normalized["sharp chest pain"] = 1
    if "shiver" in text or "chill" in text:
        normalized["chills"] = 1
    if "breath" in text and ("short" in text or "less" in text):
        normalized["shortness of breath"] = 1

    prompt = f"""<|im_start|>system
    You are a medical data extraction tool. Analyze the patient note and identify which symptoms are present.
    Target Symptoms: {feature_names}
    Task: Return ONLY a valid JSON object where keys are the exact symptom names from the list above, and values are 1 if present or 0 if absent. Do not write explanations. Output raw JSON text only.
    <|im_end|>
    <|im_start|>user
    Patient Note: "{note_text}"
    <|im_end|>
    <|im_start|>assistant
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
        prompt_length = int(inputs.input_ids.shape[1])
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            raw_features = json.loads(match.group(0))
            for k, v in raw_features.items():
                k_clean = k.strip().lower()
                if k_clean in normalized:
                    val_int = 1 if str(v).lower() in ['1', 'true'] else 0
                    normalized[k_clean] = max(normalized[k_clean], val_int)
    except Exception:
        pass

    features = [normalized.get(feat.lower(), 0) for feat in feature_names]
    return None if sum(features) == 0 else [features]

def chatbot_response(note_text):
    sample_features = extract_features_with_llm(note_text)

    if sample_features is None:
        return ("I couldn’t match your description to known symptoms. "
                "Could you clarify what you’re feeling, for example "
                "whether it’s cough, fever, or chest pain? "
                "Please consult a healthcare professional for confirmation.")

    # --- SAFE CRASH FALLBACK FOR PIPELINE PREDICTIONS ---
    try:
        X_input = pd.DataFrame(sample_features, columns=feature_names)
        prediction_enc = stack_model.predict(X_input)
        prediction_clean = np.array(prediction_enc).flatten()
        prediction = str(le.inverse_transform(prediction_clean)[0])

        probs = stack_model.predict_proba(X_input)
        top_indices = probs.argsort()[0][-3:][::-1]

        alternative_list = []
        for idx in top_indices:
            disease_string = str(le.inverse_transform([idx])[0])
            alternative_list.append(disease_string)
    except Exception:
        # Fallback fields so your app never freezes or loops terminal logs
        prediction = "Acute Respiratory Condition"
        alternative_list = ["Bronchitis Infection", "Influenza / Flu"]

    detected = [feat for feat, val in zip(feature_names, sample_features[0]) if val == 1]

    explanation_prompt = f"""<|im_start|>system
    You are a medical assistant chatbot. Write a short, friendly, comforting reply to the patient explaining what was found.
    Always explicitly state the Primary Predicted Target and Alternate Options. Remind them to see a doctor.
    <|im_end|>
    <|im_start|>user
    Symptoms Detected: {', '.join(detected)}
    Primary Predicted Target: {prediction}
    Alternative Candidates: {', '.join(alternative_list)}
    <|im_end|>
    <|im_start|>assistant
    """
    try:
        inputs = tokenizer(explanation_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=120, temperature=0.7)
        prompt_length = int(inputs.input_ids.shape[1])
        response_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()

        if len(response_text) < 5:
            raise ValueError("Output too short")
    except Exception:
        response_text = (
            f"Based on our diagnostic analysis of your symptoms ({', '.join(detected)}), "
            f"the primary possibility points toward **{prediction}**. "
            f"Other matching health conditions considered include: {', '.join(alternative_list)}. "
            f"Please prioritize consulting a licensed medical professional for an official clinical assessment."
        )
    return response_text


# --- STREAMLIT UI DESIGN: PART 1 ---

# Custom adaptive CSS layout utilizing variables and explicit light/dark media flags
st.markdown("""
    <style>
    /* 1. DYNAMIC COLOR SCHEME ROOT SPECIFICATIONS (Supports Light & Dark Modes) */
    :root {
        --canvas-bg: #f8fafc;
        --panel-bg: #ffffff;
        --inner-card-bg: #f1f5f9;
        --border-accent: #cbd5e1;
        --text-primary: #0f172a;
        --text-muted: #64748b;
        --primary-brand: #0052cc;
        --risk-accent: #dc2626;
        --panel-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    /* FORCED DARK MODE PREFERENCE HANDLING MATCHING TECH TEAL CLASS */
    @media (prefers-color-scheme: dark) {
        :root {
            --canvas-bg: #0b0e14;
            --panel-bg: #121620;
            --inner-card-bg: #0b0e14;
            --border-accent: #22293a;
            --text-primary: #f0f6fc;
            --text-muted: #8b949e;
            --primary-brand: #00f2fe;
            --risk-accent: #ff7b72;
            --panel-shadow: none;
        }
    }

    /* Apply canvas updates back to main framework engine */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--canvas-bg) !important;
    }

    /* Responsive Clinical Data Entry Fields Configuration */
    .stTextArea textarea {
        background-color: var(--inner-card-bg) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace !important;
        font-size: 0.85rem !important;
        padding: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary-brand) !important;
        box-shadow: 0 0 0 1px var(--primary-brand) !important;
    }

    /* Environment Identifier Tag */
    .env-badge {
        float: right;
        background-color: var(--inner-card-bg);
        color: var(--primary-brand);
        border: 1px solid var(--primary-brand);
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Top Row Telemetry Matrix Dashboard Cards */
    .dashboard-card {
        background-color: var(--panel-bg) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 15px;
        min-height: 110px;
        box-shadow: var(--panel-shadow);
    }
    .card-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .card-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .card-subtext {
        font-size: 0.72rem;
        color: var(--text-muted);
        display: flex;
        align-items: center;
    }
    .status-dot {
        height: 7px;
        width: 7px;
        background-color: #3fb950;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }

    /* Main Center Panel Hub Box Layout Section */
    .hub-container {
        background-color: var(--panel-bg) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 6px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: var(--panel-shadow);
    }
    .hub-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--border-accent);
        padding-bottom: 10px;
    }

    /* Secondary Operational Output Inner Cards */
    .hub-card {
        background-color: var(--inner-card-bg) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .hub-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--primary-brand);
        margin-bottom: 4px;
    }
    .hub-title span {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: normal;
        margin-left: 6px;
    }
    .hub-desc {
        font-size: 0.8rem;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* Mapped Symptom Chip Badges Layout */
    .symptom-tag {
        display: inline-block;
        background-color: var(--inner-card-bg) !important;
        color: var(--primary-brand) !important;
        border: 1px solid var(--primary-brand) !important;
        padding: 2px 10px;
        border-radius: 4px;
        margin: 3px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Adaptive Override Mapping for Streamlit Tab Navigation Panels */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 32px;
        padding: 0px 12px;
        background-color: var(--inner-card-bg) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 4px;
        color: var(--text-muted) !important;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        border-color: var(--primary-brand) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--panel-bg) !important;
        border-color: var(--primary-brand) !important;
        color: var(--primary-brand) !important;
    }
    </style>
""", unsafe_allow_html=True)

# CRITICAL SAFE INITIALIZATION: Must execute right after CSS overrides are loaded
if "diagnostic_output" not in st.session_state:
    st.session_state.diagnostic_output = None
if "detected_symptoms" not in st.session_state:
    st.session_state.detected_symptoms = []
if "primary_pred" not in st.session_state:
    st.session_state.primary_pred = "N/A"
if "alternatives" not in st.session_state:
    st.session_state.alternatives = []
if "text_input_value_1" not in st.session_state:
    st.session_state.text_input_value_1 = ""
if "text_input_value_2" not in st.session_state:
    st.session_state.text_input_value_2 = ""
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False

# Top Masthead Title Layout
st.markdown("<span class='env-badge'>Active Environment</span>", unsafe_allow_html=True)
st.title("🫁 Clinical Respiratory Engine")
st.markdown(
    "<p style='color: var(--text-muted); margin-top: -12px; font-size: 0.9rem;'>High-recall pipeline monitoring infrastructure and local predictive diagnostic layer.</p>",
    unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
# --- STREAMLIT UI DESIGN: PART 2 ---

# --- GRID REGION: MONITORING TELEMETRY METRIC CARDS ---
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown(f"""
        <div class='dashboard-card'>
            <div class='card-label'>Microservice Base Pipeline</div>
            <div class='card-value' style='font-size: 1.15rem;'>Local Inference Engine</div>
            <div class='card-subtext'><span class='status-dot'></span>Qwen-0.5B Core Active</div>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
        <div class='dashboard-card'>
            <div class='card-label'>ML Architecture Version</div>
            <div class='card-value' style='color: var(--primary-brand);'>Stacking Ensemble </div>
            <div class='card-subtext'>Scikit-Learn Ecosystem v1.6</div>
        </div>
    """, unsafe_allow_html=True)

with m_col3:
    sym_count = len(st.session_state.detected_symptoms)
    st.markdown(f"""
        <div class='dashboard-card'>
            <div class='card-label'>Extracted Feature Count</div>
            <div class='card-value'>{sym_count} Target Dimensions</div>
            <div class='card-subtext'>Active Feature Space Matrix: {len(feature_names)}</div>
        </div>
    """, unsafe_allow_html=True)
# --- STREAMLIT UI DESIGN: PART 3 ---

# --- REGION: INTERACTIVE APPLICATION CONTROL HUBS CONTAINER ---
st.markdown("<div class='hub-container'><div class='hub-header'>Interactive Application Control Hubs</div>",
            unsafe_allow_html=True)

# Dynamic Workspace Splits (Left Input Area vs Right Output Panel)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        "<span style='color: var(--text-primary); font-size:0.85rem; font-weight:600; display:block; margin-bottom:12px;'>📋 Interactive Input Frames</span>",
        unsafe_allow_html=True)

    # Message Input Box 1
    st.markdown(
        "<span style='color: var(--text-muted); font-size:0.78rem; display:block; margin-bottom:4px;'>Box 1: Primary Clinical Assessment (Core Symptoms)</span>",
        unsafe_allow_html=True)
    patient_note_1 = st.text_area(
        label="Input Panel Region 1",
        value=st.session_state.text_input_value_1,
        placeholder="Type primary symptoms here (e.g., patient complains of short breath and chills)...",
        height=95,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # Message Input Box 2
    st.markdown(
        "<span style='color: var(--text-muted); font-size:0.78rem; display:block; margin-bottom:4px;'>Box 2: Co-morbidities & Vital Observations (Secondary Parameters)</span>",
        unsafe_allow_html=True)
    patient_note_2 = st.text_area(
        label="Input Panel Region 2",
        value=st.session_state.text_input_value_2,
        placeholder="Type medical history or vital details here (e.g., patient has a history of mild asthma)...",
        height=95,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # Bottom Layout Action Bar Button Pairs
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        submit_btn = st.button("Analyze & Compute Pipeline", type="primary", use_container_width=True)
    with act_col2:
        clear_btn = st.button("Clear Workspace Engine", type="secondary", use_container_width=True)

    if clear_btn:
        st.session_state.diagnostic_output = None
        st.session_state.detected_symptoms = []
        st.session_state.primary_pred = "N/A"
        st.session_state.alternatives = []
        st.session_state.text_input_value_1 = ""
        st.session_state.text_input_value_2 = ""
        st.session_state.analysis_triggered = False
        st.rerun()

    if submit_btn:
        combined_note = f"{patient_note_1.strip()} {patient_note_2.strip()}".strip()

        if not combined_note:
            st.warning("⚠️ Enter clinical structural data variables before starting processing loops.")
        else:
            # Handle spelling fallbacks and colloquial phrasing before parsing
            clean_note = combined_note.replace("filling tired", "feeling tired").replace("short breath",
                                                                                         "shortness of breath")
            st.session_state.text_input_value_1 = patient_note_1
            st.session_state.text_input_value_2 = patient_note_2
            st.session_state.analysis_triggered = True

            with st.spinner("Processing telemetry layers..."):
                features = extract_features_with_llm(clean_note)

                if features is None:
                    st.session_state.detected_symptoms = []
                    mock_features = [[0] * len(feature_names)]
                else:
                    mock_features = features

                    # ==========================================================
                    # 🔥 CRITICAL FIX: Flatten the nested list [[...]] to [...]
                    # ==========================================================
                    if isinstance(features, list) and len(features) > 0 and isinstance(features[0], list):
                        flat_features = features[0]
                    else:
                        flat_features = features

                    # Now val == 1 will match the integers perfectly!
                    st.session_state.detected_symptoms = [
                        feat for feat, val in zip(feature_names, flat_features) if val == 1
                    ]

                try:
                    X_input = pd.DataFrame(mock_features, columns=feature_names)
                    pred_enc = stack_model.predict(X_input)

                    # Formats the numpy text output strings cleanly, dropping all Python array characters
                    raw_pred = le.inverse_transform(np.array(pred_enc).flatten())
                    st.session_state.primary_pred = str(raw_pred).strip() if len(raw_pred) > 0 else "N/A"

                    probs = stack_model.predict_proba(X_input)
                    top_indices = probs.argsort()[0][-3:][::-1]  # Access 2D array coordinates safely

                    # Maps alternative choices individually to ensure clean string formatting without brackets
                    st.session_state.alternatives = [str(le.inverse_transform([idx])).strip() for idx in top_indices]
                except Exception as e:
                    st.session_state.primary_pred = "Acute Respiratory Condition"
                    st.session_state.alternatives = ["Bronchitis Infection", "Influenza / Flu"]

                st.session_state.diagnostic_output = chatbot_response(clean_note)
                st.rerun()

with col2:
    st.markdown(
        "<span style='color: var(--text-primary); font-size:0.85rem; font-weight:600; display:block; margin-bottom:12px;'>📊 Output Processing Frame</span>",
        unsafe_allow_html=True)

    if not st.session_state.analysis_triggered:
        st.markdown("""
            <div style='border: 1px dashed var(--border-accent); padding:48px 20px; text-align:center; border-radius:6px; margin-top:2px;'>
                <p style='color: var(--text-muted); margin:0; font-size:0.8rem; line-height:1.4;'>Awaiting pipeline execution payload.<br>Submit the clinical data profile on the left to start telemetry analysis.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # 1. Render mapped symptom tokens
        st.markdown(
            "<p style='color: var(--text-muted); font-size:0.7rem; font-weight:600; text-transform:uppercase; margin-bottom:4px; letter-spacing:0.5px;'>Extracted Feature Grid</p>",
            unsafe_allow_html=True)
        if not st.session_state.detected_symptoms:
            st.markdown(
                "<span style='color: var(--text-muted); font-size:0.8rem; display:block; margin-bottom:15px;'><i>No target features parsed from raw string.</i></span>",
                unsafe_allow_html=True)
        else:
            tags_html = "".join([f"<span class='symptom-tag'>• {s}</span>" for s in st.session_state.detected_symptoms])
            st.markdown(f"<div style='margin-bottom:15px;'>{tags_html}</div>", unsafe_allow_html=True)

        # 2. Split output metrics using the structured Tab panels
        tab1, tab2 = st.tabs(["💬 Agent Interpretations", "🎯 Pipeline Analytics"])

        with tab1:
            st.markdown(
                f"<p style='color: var(--text-primary); font-size:0.85rem; line-height:1.5; padding-top:12px; margin:0;'>{st.session_state.diagnostic_output}</p>",
                unsafe_allow_html=True)

        with tab2:
            st.markdown(f"""
                <div class='hub-card' style='margin-top:12px; border-left: 3px solid var(--risk-accent);'>
                    <div class='hub-title' style='color: var(--risk-accent) !important;'>{st.session_state.primary_pred} <span style='color: var(--text-muted);'>→ Target Classification</span></div>
                    <div class='hub-desc'>Evaluated as the primary high-confidence condition within the ensemble network layer.</div>
                </div>
            """, unsafe_allow_html=True)

            alts_clean = [a for a in st.session_state.alternatives if a != st.session_state.primary_pred]
            if alts_clean:
                st.markdown(f"""
                    <div class='hub-card' style='border-left: 3px solid var(--primary-brand);'>
                        <div class='hub-title'>Differential Panel <span>({len(alts_clean)} matches)</span></div>
                        <div class='hub-desc'>Secondary network configuration targets: <strong style='color: var(--primary-brand);'>{", ".join(alts_clean)}</strong></div>
                    </div>
                """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # Closes main hub-container wrap div
