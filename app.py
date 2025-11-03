import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the model
import shap  # For XAI
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Language Translation Dictionary (UPDATED) ---
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
        
        "breakdown_header": "--- Main Risk Factors ---",
        "xai_explainer": "The model suggests these factors are contributing to a higher risk:",
        "xai_none": "The model did not find any significant factors contributing to high risk.",
        "xai_is_factor": "is a factor contributing to risk.", # Sentence end
        "factor_names": {
            "age": "Your Age",
            "systolic_bp": "Your Upper Blood Pressure",
            "diastolic": "Your Lower Blood Pressure",
            "bs": "Your Blood Sugar",
            "body_temp": "Your Body Temperature",
            "bmi": "Your BMI (Body Mass Index)",
            "previous_complications": "Your history of Previous Complications",
            "preexisting_diabetes": "Your history of Preexisting Diabetes",
            "gestational_diabetes": "Your Gestational Diabetes status",
            "mental_health": "Your Mental Health status",
            "heart_rate": "Your Heart Rate"
        },
        "data_header": "The Data You Entered:"
    },
    "Assamese": {
        "title": "গৰ্ভাৱস্থাৰ স্বাস্থ্য আশংকা পৰীক্ষক",
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

        "breakdown_header": "--- মূল আশংকাৰ কাৰকসমূহ ---",
        "xai_explainer": "মডেলটোৱে এই কাৰকসমূহক উচ্চ আশংকাৰ বাবে দায়ী বুলি চিনাক্ত কৰিছে:",
        "xai_none": "মডেলটোৱে কোনো গুৰুত্বপূৰ্ণ আশংকাৰ কাৰক বিচাৰি পোৱা নাই।",
        "xai_is_factor": "আশংকা বৃদ্ধি কৰা এটা কাৰক।", # Sentence end
        "factor_names": {
            "age": "আপোনাৰ বয়স",
            "systolic_bp": "আপোনাৰ উচ্চ ৰক্তচাপ",
            "diastolic": "আপোনাৰ নিম্ন ৰক্তচাপ",
            "bs": "আপোনাৰ তেজৰ শৰ্কৰা",
            "body_temp": "আপোনাৰ শৰীৰৰ উষ্ণতা",
            "bmi": "আপোনাৰ BMI (বডি মাচ ইণ্ডেক্স)",
            "previous_complications": "আপোনাৰ পূৰ্বৰ জটিলতাৰ ইতিহাস",
            "preexisting_diabetes": "আপোনাৰ পূৰ্বৰ ডায়েবেটিচৰ ইতিহাস",
            "gestational_diabetes": "আপোনাৰ গৰ্ভকালীন ডায়েবেটিচৰ স্থিতি",
            "mental_health": "আপোনাৰ মানসিক স্বাস্থ্যৰ স্থিতি",
            "heart_rate": "আপোনাৰ হৃদস্পন্দন"
        },
        "data_header": "আপুনি দিয়া তথ্য:"
    }
}

# --- 2. Load Model, Features, and Explainer ---
@st.cache_resource
def load_all():
    try:
        model = joblib.load('model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        explainer = shap.TreeExplainer(model)
        return model, feature_names, explainer
    except FileNotFoundError:
        st.error("Model files not found. Please upload 'model.joblib' and 'feature_names.joblib' to your GitHub repo.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading model files: {e}")
        st.stop()

model, feature_names, explainer = load_all()

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
    
    # --- 4. SHAP EXPLAINABLE AI (LIST) ---
    st.subheader(lang["breakdown_header"])
    
    try:
        # 1. Use the explainer
        shap_explanation = explainer(patient_df)

        # 2. Get explanation for CLASS 1 (High-Risk)
        explanation_for_class_1 = shap_explanation[0, :, 1]
        
        # 3. Get raw SHAP values
        shap_values_for_class_1 = explanation_for_class_1.values
        
        # 4. Get the feature names
        feature_names_for_plot = explanation_for_class_1.feature_names
        
        # 5. Create a list of (feature_name, shap_value) tuples
        feature_shap_list = list(zip(feature_names_for_plot, shap_values_for_class_1))
        
        # 6. Find all features that are "abnormal" (pushing risk UP)
        risk_factors = [
            (name, val) for name, val in feature_shap_list if val > 0.01
        ]
        
        # 7. Sort the factors
        risk_factors.sort(key=lambda x: x[1], reverse=True)
        
        # 8. Display the list
        if risk_factors:
            st.write(lang["xai_explainer"])
            
            # Get the dictionary of friendly names
            factor_name_map = lang.get("factor_names", {})
            # Get the sentence end
            sentence_end = lang.get("xai_is_factor", "is a factor contributing to risk.")
            
            # Loop and print friendly sentences
            for factor_name, factor_value in risk_factors:
                # Find the translated, "friendly" name
                friendly_name = factor_name_map.get(factor_name, factor_name.replace('_', ' ').title())
                st.markdown(f"- **{friendly_name}** {sentence_end}")
        else:
            st.write(lang["xai_none"])

    except Exception as e:
        st.error(f"Could not generate risk factor list: {e}")

    
    # 5. Display input data
    st.subheader(lang["data_header"])
    display_df = patient_df.copy()
    display_df['bmi'] = round(display_df['bmi'], 2)
    display_df['body_temp'] = round(display_df['body_temp'], 1)
    st.dataframe(display_df)
