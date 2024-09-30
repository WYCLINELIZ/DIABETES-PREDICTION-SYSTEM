import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the trained model and scaler
model = joblib.load('diabetes_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create the app title
st.title('Health Diagnosis Prediction App')

# Create input fields for user interaction
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', ['Male', 'Female'])
weight = st.number_input('Weight (kg)', min_value=0, max_value=200, value=70)
bp = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=300, value=120)
fbs = st.number_input('Blood Sugar (FBS)', min_value=0.0, max_value=500.0, value=100.0)
visit_months = st.number_input('Visit for the months', min_value=0, max_value=12, value=1)
treatment = st.number_input('Treatment', min_value=0, max_value=12, value=1)  # Added treatment input
diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'AGE': [age],
    'Sex Indicate': [1 if sex == 'Male' else 0],
    'Weight': [weight],
    'Visit for the months': [visit_months],
    'Bp(mmHg)': [bp],
    'Blood sugar(FBS)': [fbs],
    'Diabetes': [1 if diabetes == 'Yes' else 0],
    'Hypertension': [1 if hypertension == 'Yes' else 0],
})

# Impute missing values if necessary (though not required for user input here)
imputer = SimpleImputer(strategy='mean')
input_data = imputer.fit_transform(input_data)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions
if st.button("Predict Outcome"):
    # Predict the probability of both classes
    prediction_proba = model.predict_proba(input_data_scaled)
    
    # Output the probability of the positive class (e.g., index 1 for 'positive')
    positive_prob = prediction_proba[0][1]
    
    # Display the predicted outcome
    if positive_prob > 0.5:
        st.success(f"Predicted Outcome: **POSITIVE** with {positive_prob:.2f} probability")
    else:
        st.error(f"Predicted Outcome: **NEGATIVE** with {positive_prob:.2f} probability")
