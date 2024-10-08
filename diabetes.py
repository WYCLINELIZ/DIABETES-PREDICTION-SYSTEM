import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer

# Background image CSS
page_bg_img = '''
<style>
body {
    background-image: url("https://www.yourimageurl.com/TB.jpeg");
    background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Create the app title
st.title('Health Diagnosis Prediction App')

# Input fields for user interaction
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', ['Male', 'Female'])
weight = st.number_input('Weight (kg)', min_value=0, max_value=200, value=70)
bp = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=300, value=120)
fbs = st.number_input('Blood Sugar (FBS)', min_value=0.0, max_value=500.0, value=100.0)
visit_months = st.number_input('Visit for the months', min_value=0, max_value=12, value=1)
treatment = st.number_input('Treatment', min_value=0, max_value=12, value=1)
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

# Impute missing values if necessary
imputer = SimpleImputer(strategy='mean')
input_data = imputer.fit_transform(input_data)

# Make predictions based on diabetes and hypertension
if st.button("Predict Outcome"):
    if diabetes == 'Yes' and hypertension == 'Yes':
        st.success("Predicted Outcome: **POSITIVE** ")
    elif diabetes == 'No' and hypertension == 'No':
        st.error("Predicted Outcome: **NEGATIVE** ")
    else:
        st.warning("Predicted Outcome: **BOTH** ")
