import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Model Training (Cached) ---
# We use @st.cache_data to load and train the model only once.
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
        # This will show a nice error in the Streamlit app if the file is missing
        st.error(f"Error: '{dataset_path}' not found.")
        st.error("Please make sure 'Dataset - Updated.csv' is in the same folder as 'app.py'.")
        st.stop() # Stop the app
    
    # --- Data Cleaning (from our Colab notebook) ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Fill missing BMI
    if 'bmi' in df.columns and df['bmi'].isnull().sum() > 0:
        median_bmi = df['bmi'].median()
        df['bmi'].fillna(median_bmi, inplace=True)
    
    # Map and clean target variable 'risk_level'
    df['risk_level'] = df['risk_level'].map({'High': 1, 'Low': 0})
    
    # Drop any rows that couldn't be mapped (e.g., 'Medium' or NaN)
    df.dropna(subset=['risk_level'], inplace=True)
    
    df['risk_level'] = df['risk_level'].astype(int)

    # --- Define Features (X) and Target (y) ---
    y = df['risk_level']
    X = df.drop('risk_level', axis=1)
    
    # Save feature names for later use (very important!)
    feature_names = X.columns.tolist()
    
    # --- Train the Model ---
    # We train on ALL available data for the final app
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

# --- 4. Input Sidebar ---
st.sidebar.header("Enter Patient Data")

# Create a dictionary to hold the user's input
patient_input = {}

# Create input fields for each feature
patient_input['age'] = st.sidebar.slider("1. Age", 15, 60, 30)

st.sidebar.subheader("Vitals")
patient_input['systolic_bp'] = st.sidebar.slider("2. Systolic BP (Upper)", 80, 180, 120)
patient_input['diastolic'] = st.sidebar.slider("3. Diastolic BP (Lower)", 50, 120, 80)
patient_input['bs'] = st.sidebar.number_input("4. Blood Sugar (BS) (mg/dL-like)", min_value=5.0, max_value=20.0, value=7.2, step=0.1)
patient_input['body_temp'] = st.sidebar.slider("5. Body Temp (F)", 95.0, 105.0, 98.6, step=0.1)
patient_input['heart_rate'] = st.sidebar.slider("6. Heart Rate (bpm)", 60, 100, 75)

st.sidebar.subheader("Metrics & History")
patient_input['bmi'] = st.sidebar.number_input("7. BMI", min_value=15.0, max_value=50.0, value=22.5, step=0.1)
patient_input['previous_complications'] = st.sidebar.selectbox("8. Previous Complications?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
patient_input['preexisting_diabetes'] = st.sidebar.selectbox("9. Preexisting Diabetes?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
patient_input['gestational_diabetes'] = st.sidebar.selectbox("10. Gestational Diabetes?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
patient_input['mental_health'] = st.sidebar.selectbox("11. Mental Health Issues?", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')


# --- 5. Prediction Logic ---

# Create a button to trigger the prediction
if st.sidebar.button("Assess Risk", use_container_width=True):
    # 1. Convert the input dictionary to a DataFrame
    patient_df = pd.DataFrame([patient_input], columns=feature_names)

    # 2. Make predictions
    prediction = model.predict(patient_df)
    probability = model.predict_proba(patient_df)

    # 3. Get the result
    pred_label = "High-Risk" if prediction[0] == 1 else "Low-Risk"
    pred_prob = probability[0][prediction[0]] * 100

    # 4. Display the result
    st.subheader("--- Assessment Result ---")
    
    if pred_label == "High-Risk":
        st.error(f"**Prediction: {pred_label}**")
    else:
        st.success(f"**Prediction: {pred_label}**")
    
    st.write(f"**Confidence:** {pred_prob:.2f}%")
    
    st.write("--- Model Confidence Breakdown ---")
    prob_low = probability[0][0] * 100
    prob_high = probability[0][1] * 100
    st.write(f"Probability of Low-Risk: {prob_low:.2f}%")
    st.write(f"Probability of High-Risk: {prob_high:.2f}%")

    # Display the input data that was used
    st.subheader("Input Data Used for this Assessment:")
    st.dataframe(patient_df)
