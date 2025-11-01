# ---------------------------------------------------------------
# 1. SETUP: IMPORT LIBRARIES & UPLOAD DATA
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import io
from google.colab import files # For uploading files in Colab
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

print("--- Step 1: Setup Complete ---")
print("Please upload your 'Dataset - Updated.csv' file.")

# This command will open a file upload dialog
uploaded = files.upload()

# Get the filename (should be 'Dataset - Updated.csv')
filename = list(uploaded.keys())[0]

print(f"\nSuccessfully uploaded: {filename}")


# ---------------------------------------------------------------
# 2. LOAD & PREPROCESS THE DATA (WITH FIX)
# ---------------------------------------------------------------
print("\n--- Step 2: Loading and Preprocessing Data ---")

df = pd.read_csv(io.BytesIO(uploaded[filename]))

# A. Clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
print("Cleaned column names.")

# B. Handle missing values in features (X)
if 'bmi' in df.columns:
    missing_bmi_count = df['bmi'].isnull().sum()
    if missing_bmi_count > 0:
        print(f"\nFound {missing_bmi_count} missing BMI values.")
        median_bmi = df['bmi'].median()
        df['bmi'].fillna(median_bmi, inplace=True)
        print(f"Filled missing BMI values with median: {median_bmi}")
else:
    print("\n'bmi' column not found, skipping missing value fill.")
    
# C. Handle target variable (y)
# Map text labels to numbers
df['risk_level'] = df['risk_level'].map({'High': 1, 'Low': 0})

# Check for and drop any rows with unmapped/missing 'risk_level'
nan_in_target = df['risk_level'].isnull().sum()
if nan_in_target > 0:
    print(f"\nFound {nan_in_target} rows with missing 'risk_level'. Dropping them.")
    df.dropna(subset=['risk_level'], inplace=True)
else:
    print("\nNo missing 'risk_level' values found after mapping.")

# Convert target to integer type
df['risk_level'] = df['risk_level'].astype(int)

print("\nData preprocessing complete.")


# ---------------------------------------------------------------
# 3. DEFINE FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------------
print("\n--- Step 3: Defining Features and Target ---")

y = df['risk_level']
X = df.drop('risk_level', axis=1)

# Save the feature names for later, it's very important!
feature_names = X.columns.tolist()

print(f"Target (y): 'risk_level'")
print(f"Features (X): {feature_names}")


# ---------------------------------------------------------------
# 4. SPLIT DATA AND TRAIN THE MODEL
# ---------------------------------------------------------------
print("\n--- Step 4: Splitting Data and Training Model ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel training complete!")


# ---------------------------------------------------------------
# 5. EVALUATE THE MODEL (ON TEST DATA)
# ---------------------------------------------------------------
print("\n--- Step 5: Evaluating Model Performance ---")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low (0)', 'High (1)']))


# ---------------------------------------------------------------
# 6. (NEW) PREDICT ON YOUR OWN INPUT
# ---------------------------------------------------------------
print("\n--- Step 6: Practical Use Case - Predict on Your Input ---")

# Helper function to get a valid number from the user
def get_numeric_input(prompt):
    while True:
        try:
            value_str = input(prompt)
            return float(value_str)
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 22.5 or 140).")

# Helper function to get a valid binary (0 or 1) from the user
def get_binary_input(prompt):
    while True:
        try:
            value_str = input(prompt).lower().strip()
            if value_str in ['1', 'yes', 'y']:
                return 1.0
            elif value_str in ['0', 'no', 'n']:
                return 0.0
            else:
                # Try to convert to number just in case
                num = float(value_str)
                if num == 1.0:
                    return 1.0
                elif num == 0.0:
                    return 0.0
            print("Invalid input. Please enter 1 (Yes) or 0 (No).")
        except ValueError:
            print("Invalid input. Please enter 1 (Yes) or 0 (No).")

# --- Collect Data from You ---
print("\n--- Please Enter Patient Data ---")
print("Enter a number for each prompt. For Yes/No questions, enter 1 or 0.")

patient_input = {}
patient_input['age'] = get_numeric_input("1. Age (e.g., 32): ")
patient_input['systolic_bp'] = get_numeric_input("2. Systolic BP (Upper, e.g., 145): ")
patient_input['diastolic'] = get_numeric_input("3. Diastolic BP (Lower, e.g., 95): ")
patient_input['bs'] = get_numeric_input("4. Blood Sugar (BS) (e.g., 7.2 or 11.0): ")
patient_input['body_temp'] = get_numeric_input("5. Body Temp (e.g., 98): ")
patient_input['bmi'] = get_numeric_input("6. BMI (e.g., 22.5 or 32.0): ")
patient_input['previous_complications'] = get_binary_input("7. Previous Complications? (1=Yes, 0=No): ")
patient_input['preexisting_diabetes'] = get_binary_input("8. Preexisting Diabetes? (1=Yes, 0=No): ")
patient_input['gestational_diabetes'] = get_binary_input("9. Gestational Diabetes? (1=Yes, 0=No): ")
patient_input['mental_health'] = get_binary_input("10. Mental Health Issues? (1=Yes, 0=No): ")
patient_input['heart_rate'] = get_numeric_input("11. Heart Rate (e.g., 88): ")

# --- Format the data for the model ---
# Convert your dictionary into a DataFrame
# We use `feature_names` to ensure the columns are in the
# *exact* order the model was trained on.
patient_df = pd.DataFrame([patient_input], columns=feature_names)

print("\nInput data for this patient:")
print(patient_df.to_markdown(index=False)) # Prints a clean table

# --- Make the predictions ---
new_prediction = model.predict(patient_df)
new_probability = model.predict_proba(patient_df)

# --- Show the final results in a friendly way ---
labels = {1: 'High-Risk', 0: 'Low-Risk'}
pred_label = labels[new_prediction[0]]
pred_prob = new_probability[0][new_prediction[0]] * 100

print("\n--- PREDICTION RESULT ---")
print(f"\nPrediction: {pred_label}")
print(f"Confidence: {pred_prob:.2f}%")