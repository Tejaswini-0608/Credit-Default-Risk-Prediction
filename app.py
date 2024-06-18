import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the trained model using joblib
joblib_file = r"C:\Users\Tejaswini\Downloads\CreditCard\best_model_randomforest.joblib"
model = joblib.load(joblib_file)


# Define the column attributes
columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# CSS to inject contained in a string
page_style = """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f2f2f2;
            color: #333;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #4CAF50;
        }
        label {
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
            transition-duration: 0.4s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
"""

# Inject CSS with Markdown
st.markdown(page_style, unsafe_allow_html=True)


st.title('Credit Card Default Payment Prediction')

st.write('Enter the values for the attributes to predict the default payment (default or not default):')

# Create input fields for all the attributes
user_input = []
for col in columns:
    if col == 'SEX':
        value = st.selectbox(f'Select {col}', [1, 2])
    elif col == ['EDUCATION', 'MARRIAGE']:
        value = st.selectbox(f'Select {col}', [1, 2, 3, 4, 5, 6, 7, 8])
    elif col == ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
        value = st.selectbox(f'Select {col}', [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    else:
        value = st.number_input(f'Enter {col}', value=0, step=1, format='%d')
    user_input.append(value)

# Convert user input to a numpy array and reshape for prediction
user_input_array = np.array(user_input).reshape(1, -1)

# Predict the class
if st.button('Predict'):
    prediction = model.predict(user_input_array)
    if prediction[0] == 1:
        st.write('The customer is predicted to **default** on their payment.')
    else:
        st.write('The customer is predicted to **not default** on their payment.')

