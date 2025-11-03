import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Model Training (Cached) ---
@st.cache_data
def load_and_train_model(dataset_path):
    """
    Loads the dataset, preprocesses it, and trains a
    Random Forest Classifier.
    """
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f"Error: '{dataset_path}' not found.")
        st.error("Please make sure 'Dataset - Updated.csv' is in the same folder as 'app.py'.")
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

# --- 2. Load the Trained Model ---
model, feature_names = load_and_train_model('Dataset - Updated.csv')


# --- 3. Streamlit App Interface ---
st.set_page_config(page_title="Maternal Risk Assessor", layout="wide")
st.title("👩‍⚕️ Maternal Health Risk Assessment Tool")
st.write("""
This tool uses a Machine Learning model to predict pregnancy risk based on clinical data.
Enter the patient's information below.
""")

# --- 4. Input Form (NEW 3-COLUMN LAYOUT) ---
patient_input = {}

# We use st.form to batch inputs. The app won't re-run on every slider change.
with st.form(key='patient_form'):
    st.subheader("Please Enter Patient Data")
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    # --- Column 1: Patient Profile ---
    with col1:
        st.header("Profile")
        patient_input['age'] = st.slider("1. Age", 15, 60, 30)
        
        # BMI Calculation Inputs
        height_cm = st.number_input("2. Height (in cm)", min_value=100.0, max_value=250.0, value=160.0, step=0.1)
        weight_kg = st.number_input("3. Weight (in kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1)

    # --- Column 2: Vitals ---
    with col2:
        st.header("Vitals")
        patient_input['systolic_bp'] = st.slider("4. Systolic BP (Upper)", 80, 180, 120)
        patient_input['diastolic'] = st.slider("5. Diastolic BP (Lower)", 50, 120, 80)
        patient_input['bs'] = st.number_input(
            "6. Blood Sugar (BS)", 
            min_value=5.0, max_value=20.0, value=7.2, step=0.1,
            help="This dataset uses a non-standard scale for Blood Sugar (e.g., 7.2 or 11.0)."
        )
        patient_input['body_temp'] = st.slider("7. Body Temp (F)", 95.0, 105.0, 98.6, step=0.1)
        patient_input['heart_rate'] = st.slider("8. Heart Rate (bpm)", 60, 100, 75)

    # --- Column 3: Medical History ---
    with col3:
        st.header("History")
        patient_input['previous_complications'] = st.selectbox(
            "9. Previous Complications?", (0, 1), 
            format_func=lambda x: 'Yes' if x == 1 else 'No',
            help="Whether the patient has had complications in a previous pregnancy."
        )
        patient_input['preexisting_diabetes'] = st.selectbox("10. Preexisting Diabetes?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
        patient_input['gestational_diabetes'] = st.selectbox("11. Gestational Diabetes?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
        patient_input['mental_health'] = st.selectbox("12. Mental Health Issues?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')

    # --- Submit Button for the Form ---
    # The button must be *inside* the 'with st.form' block
    st.write("") # Add a little space
    submit_button = st.form_submit_button(label='Assess Risk', use_container_width=True)
    st.write("") # Add a little space


# --- 5. Prediction Logic (This runs *after* the button is pressed) ---
if submit_button:
    
    # --- Calculate BMI and add it to the input dictionary ---
    height_m = height_cm / 100.0
    patient_input['bmi'] = weight_kg / (height_m ** 2)
    # ---

    # 1. Convert the input dictionary to a DataFrame
    try:
        patient_df = pd.DataFrame([patient_input], columns=feature_names)
    except Exception as e:
        st.error(f"Error creating DataFrame: {e}")
        st.stop()

    # 2. Make predictions
    prediction = model.predict(patient_df)
    probability = model.predict_proba(patient_df)
    pred_label = "High-Risk" if prediction[0] == 1 else "Low-Risk"
    pred_prob = probability[0][prediction[0]] * 100

    # 3. Display Actionable Feedback
    st.subheader("--- Assessment Result ---")
    
    if pred_label == "High-Risk":
        st.error(f"**Prediction: {pred_label}** (Confidence: {pred_prob:.2f}%)")
        st.warning(
            "**Recommendation:** The model indicates a high-risk profile. "
            "A specialist consultation and increased monitoring are advised."
        )
    else:
        st.success(f"**Prediction: {pred_label}** (Confidence: {pred_prob:.2f}%)")
        st.info(
            "**Recommendation:** The model indicates a low-risk profile. "
            "Continue with routine prenatal care."
        )
    
    # 4. Show confidence breakdown
    st.write("--- Model Confidence Breakdown ---")
    prob_low = probability[0][0] * 100
    prob_high = probability[0][1] * 100
    st.write(f"Probability of Low-Risk: {prob_low:.2f}%")
    st.write(f"Probability of High-Risk: {prob_high:.2f}%")

    # 5. Display the input data that was used
    st.subheader("Input Data Used for this Assessment:")
    display_df = patient_df.copy()
    display_df['bmi'] = round(display_df['bmi'], 2) # Round BMI
    st.dataframe(display_df)
