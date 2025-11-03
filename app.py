import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the model
import shap  # For XAI
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Language Translation Dictionary (Your latest version) ---
LANGUAGES = {
    "English": {
        "title": "Pregnency Health Risk Assessment",
        "form_header": "Enter your data",
        "col_profile": "Profile",
        "age": "1. Age",
        "height_ft": "2. Height (Feet)",
        "height_in": "3. Height (Inches)",
        "weight": "4. Weight (in kg)",
        "col_vitals": "Vitals",
        "systolic_bp": "5. Upper Blood Pressure",
        "diastolic_bp": "6. Lower Blood Pressure",
        "bs": "7. Blood Sugar",
        "bs_help": "A number like 7.2 or 11.0",
        "body_temp": "8. Body Temperature (C)",
        "body_temp_help": "Normal is 37°C",
        "heart_rate": "9. Heart Rate (bpm)",
        "col_history": "History",
        "prev_comp": "10. Problems in past pregnancies?",
        "prev_comp_help": "Any health problems during a previous pregnancy.",
        "pre_diabetes": "11. Had diabetes before pregnancy?",
        "gest_diabetes": "12. Had diabetes during this pregnancy?",
        "mental_health": "13. Any mental health concerns?",
        "yes": "Yes",
        "no": "No",
        "submit_button": "Check Risk",
        "result_header": "--- Your Result ---",
        "result_high": "Result: High-Risk",
        "result_low": "Result: Low-Risk",
        "confidence": "Sureness",
        "advice_high": "What to do: Please see a doctor soon for extra check-ups.",
        "advice_low": "What to do: Continue with your normal check-ups.",
        "breakdown_header": "--- Why the model decided this (SHAP Plot) ---",
        "xai_explainer": "Red arrows push risk HIGHER (towards High-Risk). Blue arrows push risk LOWER.",
        "data_header": "The Data You Entered:"
    },
    "Assamese": {
        "title": "ଗৰ্ভাৱস্থাৰ স্বাস্থ্য আশংকা পৰীক্ষক",
        "form_header": "আপোনাৰ তথ্য দিয়ক",
        "col_profile": "প্ৰফাইল",
        "age": "১. বয়স",
        "height_ft": "২. উচ্চতা (ফুট)",
        "height_in": "৩. উচ্চতা (ইঞ্চি)",
        "weight": "৪. ওজন (kg)",
        "col_vitals": "ভিটেলছ (Vitals)",
        "systolic_bp": "৫. উচ্চ ৰক্তচাপ (Systolic)",
        "diastolic_bp": "৬. নিম্ন ৰক্তচাপ (Diastolic)",
        "bs": "৭. তেজৰ শৰ্কৰা",
        "bs_help": "এটা সংখ্যা যেনে ৭.২ বা ১১.০",
        "body_temp": "৮. শৰীৰৰ উষ্ণতা (C)",
        "body_temp_help": "সাধাৰণতে ৩৭°C",
        "heart_rate": "৯. হৃদস্পন্দন (bpm)",
        "col_history": "পূৰ্ব ইতিহাস",
        "prev_comp": "১০. আগৰ গৰ্ভধাৰণত সমস্যা হৈছিল?",
        "prev_comp_help": "পূৰ্বৰ গৰ্ভধাৰণৰ সময়ত যিকোনো স্বাস্থ্য সমস্যা।",
        "pre_diabetes": "১১. গৰ্ভধাৰণৰ আগতে ডায়েবেটিচ আছিল ?",
        "gest_diabetes": "১২. এই গৰ্ভধাৰণৰ সময়ত ডায়েবেটিচ হৈছিল ?",
        "mental_health": "১৩. কোনো মানসিক স্বাস্থ্যৰ চিন্তা?",
        "yes": "হয়",
        "no": "নাই",
        "submit_button": "আশংকা পৰীক্ষা কৰক",
        "result_header": "--- আপোনাৰ ফলাফল ---",
        "result_high": "ফলাফল: উচ্চ-আশংকা",
        "result_low": "ফলাফল: কম-আশংকা",
        "confidence": "নিশ্চয়তা",
        "advice_high": "কি কৰিব: অনুগ্ৰহ কৰি সোনকালে অতিৰিক্ত পৰীক্ষাৰ বাবে এজন চিকিৎসকক দেখুৱাওক।",
        "advice_low": "কি কৰিব: আপোনাৰ সাধাৰণ পৰীক্ষা অব্যাহত ৰাখক।",
        "breakdown_header": "--- মডেলটোৱে কিয় এই সিদ্ধান্ত ল'লে (SHAP Plot) ---",
        "xai_explainer": "ৰঙা কাড়বোৰে আশংকাক ওপৰলৈ (High-Risk) ঠেলে। নীলা কাড়বোৰে আশংকাক তললৈ ঠেলে।",
        "data_header": "আপুনি দিয়া তথ্য:"
    }
}

# --- 2. Load Pre-Trained Model ---
@st.cache_resource
def load_model_and_names():
    try:
        model = joblib.load('model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please upload 'model.joblib' and 'feature_names.joblib' to your GitHub repo.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading model files: {e}")
        st.stop()

model, feature_names = load_model_and_names()

# --- 3. Set Language ---
if 'lang' not in st.session_state:
    st.session_state.lang = "English"

lang_choice = st.sidebar.selectbox("Language / ভাষা", ["English", "Assamese"])
st.session_state.lang = lang_choice 
lang = LANGUAGES[st.session_state.lang] 


# --- 4. Streamlit App Interface ---
st.set_page_config(page_title="Maternal Risk Assessor", layout="wide")
st.title(lang["title"])

# --- 5. Input Form ---
patient_input = {}

with st.form(key='patient_form'):
    st.subheader(lang["form_header"])
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header(lang["col_profile"])
        patient_input['age'] = st.slider(lang["age"], 15, 60, 30)
        st.write(lang["height_ft"])
        height_ft = st.number_input("Feet", min_value=3, max_value=8, value=5, step=1, label_visibility="collapsed")
        st.write(lang["height_in"])
        height_in = st.number_input("Inches", min_value=0, max_value=11, value=3, step=1, label_visibility="collapsed")
        weight_kg = st.number_input(lang["weight"], min_value=30.0, max_value=200.0, value=65.0, step=0.1)

    with col2:
        st.header(lang["col_vitals"])
        patient_input['systolic_bp'] = st.slider(lang["systolic_bp"], 80, 180, 120)
        patient_input['diastolic'] = st.slider(lang["diastolic_bp"], 50, 120, 80)
        patient_input['bs'] = st.number_input(lang["bs"], min_value=5.0, max_value=20.0, value=7.2, step=0.1, help=lang["bs_help"])
        temp_c = st.slider(lang["body_temp"], min_value=35.0, max_value=41.0, value=37.0, step=0.1, help=lang["body_temp_help"])
        patient_input['heart_rate'] = st.slider(lang["heart_rate"], 60, 100, 75)

    with col3:
        st.header(lang["col_history"])
        patient_input['previous_complications'] = st.selectbox(lang["prev_comp"], (0, 1), format_func=lambda x: lang["yes"] if x == 1 else lang["no"], help=lang["prev_comp_help"])
        patient_input['preexisting_diabetes'] = st.selectbox(lang["pre_diabetes"], (0, 1), format_func=lambda x: lang["yes"] if x == 1 else lang["no"])
        patient_input['gestational_diabetes'] = st.selectbox(lang["gest_diabetes"], (0, 1), format_func=lambda x: lang["yes"] if x == 1 else lang["no"])
        patient_input['mental_health'] = st.selectbox(lang["mental_health"], (0, 1), format_func=lambda x: lang["yes"] if x == 1 else lang["no"])

    st.write("")
    submit_button = st.form_submit_button(label=lang["submit_button"], use_container_width=True)
    st.write("")

# --- 6. Prediction Logic & XAI ---
if submit_button:
    # --- CONVERSIONS ---
    total_inches = (height_ft * 12) + height_in
    height_cm = total_inches * 2.54
    height_m = height_cm / 100.0
    patient_input['bmi'] = weight_kg / (height_m ** 2)
    patient_input['body_temp'] = (temp_c * 9/5) + 32
    
    # 1. Convert to DataFrame
    patient_df = pd.DataFrame([patient_input], columns=feature_names)

    # 2. Make predictions
    prediction = model.predict(patient_df)
    probability = model.predict_proba(patient_df)
    pred_label_key = "result_high" if prediction[0] == 1 else "result_low"
    pred_prob = probability[0][prediction[0]] * 100

    # 3. Display Results
    st.subheader(lang["result_header"])
    if pred_label_key == "result_high":
        st.error(f"**{lang[pred_label_key]}** ({lang['confidence']}: {pred_prob:.2f}%)")
        st.warning(lang["advice_high"])
    else:
        st.success(f"**{lang[pred_label_key]}** ({lang['confidence']}: {pred_prob:.2f}%)")
        st.info(lang["advice_low"])
    
    # --- 4. SHAP EXPLAINABLE AI (XAI) PLOT ---
    st.subheader(lang["breakdown_header"])
    st.write(lang["xai_explainer"])
    
    try:
        # Create the SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # --- THIS IS THE FINAL ROBUST FIX ---
        # Get SHAP values. For a scikit-learn RF, this returns a list of 2 arrays:
        # [shap_values_for_class_0, shap_values_for_class_1]
        shap_values_list = explainer.shap_values(patient_df)
        
        # Get the base values. This also returns a list of 2 base values:
        # [base_value_for_class_0, base_value_for_class_1]
        base_values_list = explainer.expected_value
        
        # Check if the output is a list (for binary classification)
        if isinstance(shap_values_list, list) and len(shap_values_list) == 2:
            # We want the explanation for CLASS 1 (High-Risk)
            shap_values_for_plot = shap_values_list[1][0] # class 1, first patient
            base_value_for_plot = base_values_list[1] # base value for class 1
        else:
            # Fallback for unexpected format (e.g., single-output)
            shap_values_for_plot = shap_values_list[0] # first patient
            base_value_for_plot = base_values_list # base value is a single float
        # --- END OF FIX ---

        # Create the force plot
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            base_value_for_plot,
            shap_values_for_plot,
            patient_df,
            matplotlib=True,
            show=False,
            text_rotation=0
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
    except Exception as e:
        st.error(f"Could not generate SHAP plot: {e}")

    
    # 5. Display input data
    st.subheader(lang["data_header"])
    display_df = patient_df.copy()
    display_df['bmi'] = round(display_df['bmi'], 2)
    display_df['body_temp'] = round(display_df['body_temp'], 1)
    st.dataframe(display_df)
