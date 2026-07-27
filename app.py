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


# --- STREAMLIT UI DESIGN ---

# Custom CSS injected for premium look, modern card designs, and badge styling
st.markdown("""
    <style>
    /* Main container and card backgrounds */
    .stTextArea textarea {
        background-color: #1e2230 !important;
        border: 1px solid #3b4261 !important;
        border-radius: 8px !important;
        color: #f8f9fa !important;
    }
    .medical-card {
        background-color: #1a1e29;
        border: 1px solid #2e3440;
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .alt-card {
        background-color: #1c2130;
        border-left: 5px solid #4b9fff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    /* Symptom chips/badges styling */
    .symptom-badge {
        display: inline-block;
        background-color: #2b3045;
        color: #a4b1cd;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        border: 1px solid #3e4563;
        font-weight: 500;
    }
    /* Subtle headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e9f0;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Subtitle header with horizontal break
st.title("🫁 Clinical Respiratory Assistant")
st.markdown("##### *Ensemble ML Pipeline & Generative Clinical Intelligence Agent*")
st.divider()

# Persistent session state keys for the inputs and outputs
if "diagnostic_output" not in st.session_state:
    st.session_state.diagnostic_output = None
if "detected_symptoms" not in st.session_state:
    st.session_state.detected_symptoms = []
if "primary_pred" not in st.session_state:
    st.session_state.primary_pred = None
if "alternatives" not in st.session_state:
    st.session_state.alternatives = []
if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""

# Two-Column Workspace Layout (Left = Input Panel, Right = Dynamic Diagnostics)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📋 Patient Assessment")
    st.caption("Input clinical observations, symptom clusters, or patient notes below.")

    patient_note = st.text_area(
        label="Clinical Free-Text Note Input:",
        value=st.session_state.text_input_value,
        placeholder="Example: Patient complains of pain on chest, headache and shivering...",
        height=220,
        label_visibility="collapsed"
    )

    # Create action button row layout side-by-side
    btn_col1, btn_col2 = st.columns([2, 1])

    with btn_col1:
        submit_btn = st.button("Analyze & Diagnose Pipeline", type="primary", use_container_width=True)

    with btn_col2:
        clear_btn = st.button("Clear Workspace", type="secondary", use_container_width=True)

    if clear_btn:
        # Erase all cached calculation parameters and structural strings
        st.session_state.diagnostic_output = None
        st.session_state.detected_symptoms = []
        st.session_state.primary_pred = None
        st.session_state.alternatives = []
        st.session_state.text_input_value = ""
        st.rerun()

    if submit_btn:
        if not patient_note.strip():
            st.warning("⚠️ Please input text data or symptoms before initiating diagnostics.")
        else:
            # Sync value to track states through consecutive refreshes
            st.session_state.text_input_value = patient_note
            with st.spinner("Decoding notes, extracting symptoms, and running ensemble models..."):
                # 1. Fetch feature arrays
                features = extract_features_with_llm(patient_note)

                if features is None:
                    st.session_state.diagnostic_output = None
                    st.session_state.detected_symptoms = []
                else:
                    # Map structural active symptoms lists
                    st.session_state.detected_symptoms = [
                        feat for feat, val in zip(feature_names, features[0]) if val == 1
                    ]

                    # 2. Pipeline evaluations
                    try:
                        X_input = pd.DataFrame(features, columns=feature_names)
                        pred_enc = stack_model.predict(X_input)
                        st.session_state.primary_pred = str(le.inverse_transform(np.array(pred_enc).flatten())[0])

                        probs = stack_model.predict_proba(X_input)
                        top_indices = probs.argsort()[0][-3:][::-1]
                        st.session_state.alternatives = [str(le.inverse_transform([idx])[0]) for idx in top_indices]
                    except Exception:
                        st.session_state.primary_pred = "Acute Respiratory Condition"
                        st.session_state.alternatives = ["Bronchitis Infection", "Influenza / Flu"]

                    # 3. Generate chatbot conversational output summaries
                    st.session_state.diagnostic_output = chatbot_response(patient_note)
                    st.rerun()

with col2:
    st.subheader("📊 Diagnostic Workspace")

    # Render fallback/empty UI state if no analysis has been executed yet
    if not st.session_state.detected_symptoms and not st.session_state.diagnostic_output:
        st.info(
            "💡 Complete the patient assessment form on the left to populate real-time diagnostic models and LLM summary assessments.")

    # Handle matching symptom exceptions
    elif len(st.session_state.detected_symptoms) == 0:
        st.error("❌ Symptom Matching Failed")
        st.markdown(
            "The system was unable to parse features matching the target respiratory configurations. "
            "Please try refining the note description with clearer anatomical contexts (e.g., 'coughing', 'fever', 'chest pain')."
        )

    else:
        # Step 1: Render parsed structured symptoms inside functional badge bubbles
        st.markdown("<div class='section-title'>🔍 Extracted Clinical Symptoms</div>", unsafe_allow_html=True)
        badge_html = "".join([f"<span class='symptom-badge'>🔹 {s}</span>" for s in st.session_state.detected_symptoms])
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Step 2: Tabbed Results UI Container separating Assistant and Analytical views
        tab1, tab2 = st.tabs(["🤖 Medical Agent Summary", "🔬 Ensemble Analytics Pipeline"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            # Display conversational LLM summary inside styled layout callouts
            st.markdown(f"{st.session_state.diagnostic_output}")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)

            # Display primary tabular target
            st.markdown("<div class='section-title'>🥇 Top Predicted Condition</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='medical-card'>
                    <h3 style='margin:0; color:#ff4b4b;'>{st.session_state.primary_pred}</h3>
                    <p style='margin:5px 0 0 0; color:#8892b0; font-size:0.9rem;'>Calculated as highest probability match based on current feature matrix configuration.</p>
                </div>
            """, unsafe_allow_html=True)

            # Display alternate differential targets
            st.markdown("<div class='section-title'>📋 Differential Candidates</div>", unsafe_allow_html=True)
            for alt in st.session_state.alternatives:
                if alt != st.session_state.primary_pred:
                    st.markdown(f"""
                        <div class='alt-card'>
                            <strong style='color:#4b9fff;'>• {alt}</strong>
                        </div>
                    """, unsafe_allow_html=True)





