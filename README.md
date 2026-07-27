# 📑 Project Documentation: Clinical Respiratory Assistant Using Stacking Ensembles & Local LLMs

---

## Part 1: Product Requirement Document (PRD)

### 1. Document Control
*   **Project Name:** AI Clinical Respiratory Assistant Using Stacking Ensembles & Local LLMs
*   **Version:** 3.0.0 (Production-Ready Stacking & Local LLM Modular Integration)
*   **Author:** AI Project Lead / Developer
*   **Status:** Ready for Final Submission / Production Deployment

### 2. Executive Summary & Objectives
The AI Clinical Respiratory Assistant is an advanced hybrid machine learning and natural language processing application. It couples an optimized multi-model machine learning Stacking Ensemble Classifier with a lightweight local Large Language Model (LLM) to parse unstructured patient clinical notes, extract granular symptom dimensions, and predict 19 distinct respiratory conditions while ensuring maximum predictive stability, data privacy, and end-user explainability.

### 3. Functional Requirements (FR)
*   **FR-1 (Domain Scope):** The system must accept free-text patient clinical entries covering 19 distinct target respiratory symptoms (e.g., shortness of breath, sharp chest pain, fever, coughing up sputum, wheezing).
*   **FR-2 (Multi-Model Stacking Ensemble):** The machine learning core must use a multi-model stacking framework to eliminate single-algorithm bias and optimize predictive capabilities.
*   **FR-3 (Probabilistic Candidate Extraction):** The interface must display not only the primary predicted diagnosis but also compute and present the top 3 alternative candidate conditions ranked by prediction probability.
*   **FR-4 (Local LLM Feature Extraction):** The parsing pipeline must extract symptoms directly from unstructured clinical strings into structured numeric arrays using a locally hosted LLM to maintain data isolation.
*   **FR-5 (Explainable Conversational Output):** The final output must pass the classification results back through a generative model to translate abstract diagnostic targets into a friendly, context-aware, human-readable summary.

### 4. Non-Functional Requirements (NFR)
*   **Data Minimization & Privacy:** The application must enforce absolute data minimization by analyzing only specified symptoms. It must exclude all Protected Attributes (race, age, sex at birth, ethnicity) and Personally Identifiable Information (PII) like home addresses or phone numbers.
*   **System Robustness & Fail-safes:** The application must implement a dual-engine architecture. If the local LLM runtime encounters a CPU constraint or timeout, the extraction layer must automatically fall back to high-speed contextual regex keyword rules to ensure 100% operational uptime.
*   **Local Infrastructure Deployment:** The deep learning components must execute entirely inside a local runtime footprint using optimized data types to avoid dependency on paid cloud APIs.

---

## Part 2: Technical Requirement Document (TRD)

### 1. Technology Stack Selection
*   **Core Programming Language:** Python 3.12+
*   **Data Serialization & Formats:** Pandas, NumPy, JSON, Regular Expressions (Regex)
*   **Core Machine Learning Engine:** Scikit-Learn (`RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`)
*   **Advanced Gradient Boosting Engine:** Extreme Gradient Boosting (`XGBoost` / `XGBClassifier`)
*   **Deep Learning & Local Tokenization Engine:** PyTorch (`torch`), Hugging Face `transformers` pipeline
*   **Foundational LLM Core:** `Qwen/Qwen2.5-0.5B-Instruct` (Configured for adaptive FP16/FP32 execution based on target hardware computing capabilities)
*   **Pipeline Serialization Layer:** `joblib` (For compiling and freezing model weights and `LabelEncoder` properties)
*   **Presentation UI Layer:** Streamlit Dashboard Framework

### 2. Algorithmic Specifications & Optimization
The machine learning prediction layer abandons weak standalone predictors in favor of an optimized, robust **Stacking Meta-Learner** architecture:
*   **Base Estimator 1 (Random Forest):** Configured with 200 estimators, max depth of 8, a minimum sample leaf size of 5, and balanced class weights to control variance.
*   **Base Estimator 2 (Gradient Boosting):** Configured with 300 estimators, a learning rate of 0.05, and a max depth of 5 to control system bias.
*   **Base Estimator 3 (XGBoost):** Configured with 300 estimators, a max depth of 8, and a learning rate of 0.05 to handle deep interaction features.
*   **Meta-Learner (Combiner):** A regularized `LogisticRegression` algorithm configured with 1000 maximum iterations to ingest base probability maps and yield final diagnoses.
*   **Validation Validation**: The model is validated via stratified 5-fold cross-validation loops tracking Macro F1-scores, ensuring accurate class separation for imbalanced condition fields.

### 3. Scientific Justification of the Hybrid Architecture
Traditional "Black-Box" Large Language Models are intentionally excluded from performing the direct diagnostic prediction for two primary medical-safety reasons:
1.  **Hallucination Prevention:** Generative LLMs frequently hallucinate or invent medical patterns when presented with rare variations. Entrusting the diagnosis entirely to an ensemble machine learning classifier trained on static, verified clinical tables ensures deterministic accuracy.
2.  **Privacy-Centric Modularization:** By keeping the machine learning weights (`.pkl`) completely distinct from the text processing blocks, the app can run on consumer-grade hardware. The LLM handles unstructured syntax mining, while the mathematical classifier handles numerical calculations—delivering speed, reliability, and security without massive cloud costs.

---

## Part 3: Hybrid System Architecture Diagram

```text
+-----------------------------------------------------------------------------------+

|                            STREAMLIT UI USER FRONTEND                             |
|          User Input: "Patient complains of chest pain, chills, and cough..."     |
+------------------------------------------+----------------------------------------+
                                           | Unstructured Clinical Note Text
                                           v
+-----------------------------------------------------------------------------------+

|                        DATA PARSING & HYBRID EXTRACTION LAYER                     |
|  [Primary Engine]: Qwen2.5-0.5B-Instruct parses zero-shot structured JSON.        |
|  [Backup Engine]: Localized Contextual Regex rules patch inputs if LLM lags.      |
+------------------------------------------+----------------------------------------+
                                           | Clean 19-Dimensional Numeric Array (1s/0s)
                                           v
+-----------------------------------------------------------------------------------+

|                    MODELING LAYER: STACKING META-LEARNER ENGINE                   |
|  Base Estimators:  [Random Forest]  +  [Gradient Boosting]  +  [XGBoost]          |
|  Meta-Learner:     Ingests probabilities via penalized Logistic Regression.       |
+------------------------------------------+----------------------------------------+
                                           | Primary Diagnosis + Probability Ranks
                                           v
+-----------------------------------------------------------------------------------+

|                       GENERATIVE EXPLAINER ANSWER GENERATOR                       |
|  Prompt: "Write a short comforting reply stating diagnosis and alternatives..."  |
+------------------------------------------+----------------------------------------+
                                           | Formatted Natural Language Summary Response
                                           v
+-----------------------------------------------------------------------------------+

|                         STREAMLIT RENDERED INTERFACE OUTPUT                       |
|     Displays Patient Symptoms, Core Target Condition, and Reassuring Report.      |
+-----------------------------------------------------------------------------------+
```


# AI Clinical Respiratory Assistant: Stacking Ensemble & Hybrid LLM Pipeline

An end-to-end, production-ready AI application that transforms raw, unstructured clinical text notes into granular symptom vectors, runs them through an optimized machine learning stacking ensemble classifier to predict respiratory conditions, and generates human-readable diagnostic summaries using a local Large Language Model (LLM).

## 🚀 Live Visual Dashboard
*(Tip: Replace this line with a screenshot of your working Streamlit web interface to serve as your visual portfolio anchor!)*

---

## 🏗️ Technical Architecture Overview

The system operates as a modular three-tier software pipeline, eliminating the friction between chaotic real-world user entries and rigid machine learning classification matrix rules.

*   **Step 1**: [Unstructured Patient Note] Input
*   **Step 2**: [Hybrid LLM/Rule Extractor] Powered by Qwen2.5-0.5B-Instruct
*   **Step 3**: [19-Dimensional Numeric Array] Extracted features mapping
*   **Step 4**: [Stacking Ensemble Classifier] RF + Gradient Boosting + XGBoost
*   **Step 5**: [Output Probability Ranks] Primary Diagnosis evaluation
*   **Step 6**: [Natural Language Explainer] Context-Aware Summary Generator
*   **Step 7**: [Streamlit UI Live Web App] Final presentation layer

### 1. Data Parsing Layer (Hybrid Extraction)
* **The Challenge**: Users write clinical symptoms using varied formatting, inconsistent spaces (e.g., "head ache" vs "headache"), and complex medical phrasing. Traditional text parsers miss these edge cases, leading to empty input vectors.
* **The Solution**: A dual-engine extraction function combines localized context-aware regex keyword validation with zero-shot text token parsing powered by **Qwen2.5-0.5B-Instruct**. It dynamically maps raw notes into an exact 19-dimensional mathematical array representing target respiratory symptoms.

### 2. Modeling Layer (Stacking Ensemble Classifier)
To maximize predictive stability and eliminate model bias across complex tabular features, the classification layer uses a **Stacking Meta-Learner** architecture:
* **Base Learners**: A diversified trio comprising `RandomForestClassifier`, `GradientBoostingClassifier`, and `XGBClassifier` individual models, heavily tuned with balanced class weights to offset target group skewness.
* **Meta-Learner**: A penalized `LogisticRegression` algorithm serves as the final arbiter, ingesting the prediction probability vectors of the base models to calculate the definitive primary disease label.
* **Evaluation Pipeline**: Cross-validation loops and per-class Macro F1-score evaluation metrics guarantee performance tracking and structural robustness before deployment.

### 3. Deployment Layer (Production-Ready Architecture)
* Migrated from disconnected prototyping Jupyter cells into a unified backend interface file (`app.py`).
* Model weights and text array string mapping boundaries are fully serialized via `joblib` pipelines.
* Built utilizing a standalone **Streamlit** micro-frontend architecture linked to cloud hosting proxies.

---

## 🛠️ Technology Stack & Frameworks

* **Programming Core**: Python 3
* **Machine Learning & Math**: Scikit-Learn, XGBoost, NumPy, Pandas
* **Deep Learning & Language**: PyTorch, Hugging Face Transformers Engine
* **Foundational Model Core**: Qwen/Qwen2.5-0.5B-Instruct
* **Interface & Cloud Infrastructure**: Streamlit Dashboard UI framework, Ngrok Network Tunnels

---

## 📦 How to Run Locally

To take this application completely offline and run it independently on a local machine, execute the following commands in your computer's terminal:

1. Clone or download your project folder containing `app.py`, `respiratory_disease_model.pkl`, and `label_encoder.pkl`.
2. Install the necessary system dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost transformers torch joblib streamlit
   ```
3. Boot up the independent desktop dashboard server instantly:
   ```bash
   streamlit run app.py
   ```
