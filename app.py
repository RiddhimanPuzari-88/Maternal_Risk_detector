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
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f"Error: '{dataset_path}' not found.")
        st.error("Please make sure 'Dataset - Updated.csv' is in the same folder as 'app.py'.")
        st.stop()
    
    # --- Data Cleaning (from our Colab notebook) ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Fill missing BMI
    if 'bmi' in df.columns and df['bmi'].isnull().sum() > 0:
        median_bmi = df['bmi'].median()
        df['bmi'].fillna(median_bmi, inplace=True)
    
    # Map and clean target variable 'risk_level'
    df['risk_level'] = df['risk_level'].map({'High': 1, 'Low': 0})
    
    # Drop any rows that couldn't be mapped
    df.dropna(subset=['risk_level'], inplace=True)
    
    df['risk_level'] = df['risk_level'].astype(int)

    # --- Define Features (X) and Target (y) ---
    y = df['risk_level']
    X = df.drop('risk_level', axis=1)
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    
    # --- Train the Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model trained successfully.")
    
    return model, feature_names

# --- 2. Load the Trained Model ---
# This line tells the function where to find your data file
model, feature_names = load_and_train_model('Dataset - Updated.csv')


# --- 3. Streamlit App Interface ---
st.set_page_config(page_title="Maternal Risk Assessor", layout="wide")
st.title("👩‍⚕️ Maternal Health Risk Assessment Tool")
st.write("""
This tool uses a Machine Learning model to predict whether a pregnancy
is **High-Risk** or **Low-Risk** based on clinical health indicators.
Please enter the patient's data in the sidebar.
""")

# --- 4. Input Sidebar (with Improvements) ---
st.sidebar.header("Enter Patient Data")

# This dictionary will hold all the inputs
patient_input = {}

patient_input['age'] = st.sidebar.slider("1. Age", 15, 60, 30)

st.sidebar.subheader("Vitals")
patient_input['systolic_bp'] = st.sidebar.slider("2. Systolic BP (Upper)", 80, 180, 120)
patient_input['diastolic'] = st.sidebar.slider("3. Diastolic BP (Lower)", 50, 120, 80)
patient_input['bs'] = st.sidebar.number_input(
    "4. Blood Sugar (BS)", 
    min_value=5.0, max_value=20.0, value=7.2, step=0.1,
    help="This dataset uses a non-standard scale for Blood Sugar (e.g., 7.2 or 11.0)."
)
patient_input['body_temp'] = st.sidebar.slider("5. Body Temp (F)", 95.0, 105.0, 98.6, step=0.1)
patient_input['heart_rate'] = st.sidebar.slider("6. Heart Rate (bpm)", 60, 100, 75)

st.sidebar.subheader("Metrics & History")

# --- IMPROVEMENT 1: Calculate BMI ---
height_cm = st.sidebar.number_input("7. Height (in cm)", min_value=100.0, max_value=250.0, value=160.0, step=0.1)
weight_kg = st.sidebar.number_input("8. Weight (in kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1)
# --- End of BMI Improvement ---

patient_input['previous_complications'] = st.sidebar.selectbox(
    "9. Previous Complications?", (0, 1), 
    format_func=lambda x: 'Yes' if x == 1 else 'No',
    help="Whether the patient has had complications in a previous pregnancy."
)
patient_input['preexisting_diabetes'] = st.sidebar.selectbox("10. Preexisting Diabetes?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
patient_input['gestational_diabetes'] = st.sidebar.selectbox("11. Gestational Diabetes?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
patient_input['mental_health'] = st.sidebar.selectbox("12. Mental Health Issues?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')


# --- 5. Prediction Logic (with Improvements) ---

if st.sidebar.button("Assess Risk", use_container_width=True):
    
    # --- IMPROVEMENT 1 (Part 2): Add calculated BMI to the input dictionary ---
    height_m = height_cm / 100.0
    patient_input['bmi'] = weight_kg / (height_m ** 2)
    # --- End of Improvement ---

    # 1. Convert the input dictionary to a DataFrame
    #    We use `feature_names` to ensure the column order is correct
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

    # --- IMPROVEMENT 3: Actionable Feedback ---
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
    # --- End of Improvement ---

    
    st.write("--- Model Confidence Breakdown ---")
    prob_low = probability[0][0] * 100
    prob_high = probability[0][1] * 100
    st.write(f"Probability of Low-Risk: {prob_low:.2f}%")
    st.write(f"Probability of High-Risk: {prob_high:.2f}%")

    # Display the input data that was used
    st.subheader("Input Data Used for this Assessment:")
    # Re-order the display dataframe to show BMI
    display_df = patient_df.copy()
    display_df['bmi'] = round(display_df['bmi'], 2) # Round BMI for display
    st.dataframe(display_df)
