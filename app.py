import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
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
        "systolic_bp": "5. Upper Blood Pressure ",
        "diastolic_bp": "6. Lower Blood Pressure ",
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
        "breakdown_header": "--- How the model decided ---",
        "prob_low": "Chance of Low-Risk",
        "prob_high": "Chance of High-Risk",
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
        "breakdown_header": "--- মডেলটোৱে কেনেকৈ সিদ্ধান্ত ল'লে ---",
        "prob_low": "কম-আশংকা হোৱাৰ সম্ভাৱনা",
        "prob_high": "উচ্চ-আশংকা হোৱাৰ সম্ভাৱনা",
        "data_header": "আপুনি দিয়া তথ্য:"
    }
}

# --- 2. Model Training (Cached) ---
@st.cache_data
def load_and_train_model(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f"Error: '{dataset_path}' not found.")
        st.stop()
    
    # --- Data Cleaning ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    if 'bmi' in df.columns and df['bmi'].isnull().sum() > 0:
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
    df['risk_level'] = df['risk_level'].map({'High': 1, 'Low': 0})
    df.dropna(subset=['risk_level'], inplace=True)
    df['risk_level'] = df['risk_level'].astype(int)

    # --- Define Features (X) and Target (y) ---
    y = df['risk_level']
    X = df.drop('risk_level', axis=1)
    feature_names = X.columns.tolist()
    
    # --- Train the Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model trained successfully.")
    return model, feature_names

# --- 3. Load Model and Set Language ---
model, feature_names = load_and_train_model('Dataset - Updated.csv')

# Add language selector to sidebar
lang_choice = st.sidebar.selectbox("Language / ভাষা", ["English", "Assamese"])
lang = LANGUAGES[lang_choice]


# --- 4. Streamlit App Interface ---
st.set_page_config(page_title="Maternal Risk Assessor", layout="wide")
st.title(lang["title"])

# --- 5. Input Form (UPDATED) ---
patient_input = {}

with st.form(key='patient_form'):
    st.subheader(lang["form_header"])
    
    col1, col2, col3 = st.columns(3)
    
    # --- Column 1: Patient Profile ---
    with col1:
        st.header(lang["col_profile"])
        patient_input['age'] = st.slider(lang["age"], 15, 60, 30)
        
        # --- NEW HEIGHT INPUTS ---
        st.write(lang["height_ft"])
        height_ft = st.number_input("Feet", min_value=3, max_value=8, value=5, step=1)
        st.write(lang["height_in"])
        height_in = st.number_input("Inches", min_value=0, max_value=11, value=3, step=1)
        
        weight_kg = st.number_input(lang["weight"], min_value=30.0, max_value=200.0, value=65.0, step=0.1)

    # --- Column 2: Vitals ---
    with col2:
        st.header(lang["col_vitals"])
        patient_input['systolic_bp'] = st.slider(lang["systolic_bp"], 80, 180, 120)
        patient_input['diastolic'] = st.slider(lang["diastolic_bp"], 50, 120, 80)
        patient_input['bs'] = st.number_input(lang["bs"], min_value=5.0, max_value=20.0, value=7.2, step=0.1, help=lang["bs_help"])
        
        # --- NEW TEMPERATURE INPUT ---
        temp_c = st.slider(lang["body_temp"], min_value=35.0, max_value=41.0, value=37.0, step=0.1, help=lang["body_temp_help"])
        
        patient_input['heart_rate'] = st.slider(lang["heart_rate"], 60, 100, 75)

    # --- Column 3: Medical History ---
    with col3:
        st.header(lang["col_history"])
        patient_input['previous_complications'] = st.selectbox(
            lang["prev_comp"], (0, 1), 
            format_func=lambda x: lang["yes"] if x == 1 else lang["no"],
            help=lang["prev_comp_help"]
        )
        patient_input['preexisting_diabetes'] = st.selectbox(
            lang["pre_diabetes"], (0, 1), 
            format_func=lambda x: lang["yes"] if x == 1 else lang["no"]
        )
        patient_input['gestational_diabetes'] = st.selectbox(
            lang["gest_diabetes"], (0, 1), 
            format_func=lambda x: lang["yes"] if x == 1 else lang["no"]
        )
        patient_input['mental_health'] = st.selectbox(
            lang["mental_health"], (0, 1), 
            format_func=lambda x: lang["yes"] if x == 1 else lang["no"]
        )

    # --- Submit Button ---
    st.write("")
    submit_button = st.form_submit_button(label=lang["submit_button"], use_container_width=True)
    st.write("")


# --- 6. Prediction Logic (UPDATED) ---
if submit_button:
    
    # --- CONVERSIONS ---
    # 1. Convert Height (Feet+Inches) to CM, then to BMI
    total_inches = (height_ft * 12) + height_in
    height_cm = total_inches * 2.54
    height_m = height_cm / 100.0
    patient_input['bmi'] = weight_kg / (height_m ** 2)
    
    # 2. Convert Temperature (Celsius) to Fahrenheit
    patient_input['body_temp'] = (temp_c * 9/5) + 32
    # ---
    
    
    # 1. Convert to DataFrame
    try:
        patient_df = pd.DataFrame([patient_input], columns=feature_names)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

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
    
    # 4. Show confidence breakdown
    st.write(lang["breakdown_header"])
    prob_low = probability[0][0] * 100
    prob_high = probability[0][1] * 100
    st.write(f"{lang['prob_low']}: {prob_low:.2f}%")
    st.write(f"{lang['prob_high']}: {prob_high:.2f}%")

    # 5. Display input data
    st.subheader(lang["data_header"])
    display_df = patient_df.copy()
    display_df['bmi'] = round(display_df['bmi'], 2)
    display_df['body_temp'] = round(display_df['body_temp'], 1) # Show temp in F
    st.dataframe(display_df)
