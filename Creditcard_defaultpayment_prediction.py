# -*- coding: utf-8 -*-
"""Creditcard Fraud.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZrKLDhQc1ME9aN4YwE9isEGUN_Bj0SFD

# **Load and Explore the Dataset**
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pandas as pd

# Load the dataset
file_path = '/content/CreditCard_default_payment_prediction.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print(data.info())
print(data.describe())
print(data.head())

# Check for class imbalance
print(data['default payment next month'].value_counts())

data.info()

data.head()

data.tail()

# Number of rows and columns
num_rows, num_columns = data.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Column attributes
column_attributes = data.columns.tolist()
print(f"Column attributes: {column_attributes}")

# Count of 0s and 1s in the 'default payment next month' column
default_counts = data['default payment next month'].value_counts()
num_non_default = default_counts[0]
num_default = default_counts[1]
print(f"Number of non-default transactions (0): {num_non_default}")
print(f"Number of default transactions (1): {num_default}")

"""# **Data Preprocessing**"""

from sklearn.preprocessing import StandardScaler

# Check for missing values
print(data.isnull().sum())

# Assuming 'default payment next month' is the target column
X = data.drop('default payment next month', axis=1)
y = data['default payment next month']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

data.isnull().values.any()

# Check for null values
if data.isnull().values.any():
    print("Null values are present in the dataset.")

    # Display columns with null values and their count
    null_values = data.isnull().sum()
    null_columns = null_values[null_values > 0]
    print(f"Columns with null values:\n{null_columns}")

    # Remove rows with null values
    data_cleaned = data.dropna()
    print("Rows with null values have been removed.")

    # Verify that null values have been removed
    print(f"Number of rows after removing null values: {data_cleaned.shape[0]}")
    print(f"Number of columns after removing null values: {data_cleaned.shape[1]}")
else:
    print("There are no null values in the dataset.")

"""# **Handling Imbalanced Data**"""

from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Check if the dataset is balanced now
print(y_balanced.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

"""# **Model Implementation and Evaluation**"""

import numpy as np
# List of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(),
    'Isolation Forest': IsolationForest(contamination=0.1),  # Adjust contamination accordingly
    'Local Outlier Factor': LocalOutlierFactor(novelty=True)  # Set novelty to True for prediction
}

# Initialize a DataFrame to store the results
results = []

# Evaluate each model
for name, model in models.items():
    if name in ['Isolation Forest', 'Local Outlier Factor']:
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

results_df = pd.DataFrame(results)

# Display the results
print(results_df)

"""# **Visualize Model Performance**"""

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example results_df for demonstration purposes
data = {
    'Model': ['LogisticRegression', 'DecisionTree', 'RandomForest', 'SVM',  'K-Nearest Neighbors', 'Naive Bayes', 'XGBoost',  'Isolation Forest','Local Outlier Factor' ],
    'Accuracy': [0.682540, 0.784127, 0.882540, 0.749206, 0.746032 , 0.558730, 0.869841, 0.523810 , 0.485714],
    'Precision': [0.683871,0.787097,0.870370,0.767123,0.684211,0.534091,0.886667,0.629630,0.454545],
    'Recall': [0.675159,0.777070,0.898089,0.713376,0.910828,0.898089,0.847134,0.108280,0.159236],
    'F1 Score': [0.679487,0.782051,0.884013,0.739274,0.781421,0.669834,0.866450,0.184783,0.235849]
}

results_df = pd.DataFrame(data)

# List of metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Plot each metric
for metric in metrics:
    plt.figure(figsize=(4, 4))
    sns.barplot(x='Model', y=metric, data=results_df, color='blue')
    plt.title(f'Model Comparison by {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.show()

"""# **Saving and testing the best model with an input array**"""

# Assuming Random Forest is the best model based on previous evaluations
# Train the best model
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

import joblib
# Save the model using joblib
joblib_file = "best_model_randomforest.joblib"
joblib.dump(best_model, joblib_file)

# Code for loading and using the saved model for prediction
def load_model_and_predict(input_data):
    loaded_model = joblib.load(joblib_file)
    prediction = loaded_model.predict(input_data)
    return prediction


# Map binary predictions to labels for default payments
def map_predictions_to_labels(predictions):
    labels = ["will not default" if pred == 0 else "will default" for pred in predictions]
    return labels

# Example input data for testing the model (with 23 features)
input_data = np.array([
    [50000, 1, 2, 1, 35, 0, 0, 0, 0, 0, 0, 15000, 14000, 13000, 12000, 11000, 10000, 2000, 2100, 2200, 2300, 2400, 2500],  # Sample 1: Should predict 'will not default'
    [200000, 2, 1, 2, 40, 1, 2, 3, 4, 5, 6, 50000, 49000, 48000, 47000, 46000, 45000, 4000, 4100, 4200, 4300, 4400, 4500],  # Sample 2: Should predict 'will default'
    [150000, 1, 3, 1, 30, -1, -1, -1, -1, -1, -1, 30000, 29000, 28000, 27000, 26000, 25000, 3000, 3100, 3200, 3300, 3400, 3500]  # Sample 3: Should predict 'will not default'
])

# Call the function to make predictions for example data
example_prediction = load_model_and_predict(input_data)
example_prediction_labels = map_predictions_to_labels(example_prediction)
print("Example Predictions:", example_prediction_labels)

# Code for loading and using the saved model for prediction
def load_model_and_predict(input_data):
    loaded_model = joblib.load(joblib_file)
    prediction = loaded_model.predict(input_data)
    return prediction

# Example input data for testing the model (with 23 features)
input_data = np.array([
    [50000, 1, 2, 1, 35, 0, 0, 0, 0, 0, 0, 15000, 14000, 13000, 12000, 11000, 10000, 2000, 2100, 2200, 2300, 2400, 2500],  # Sample 1: Should predict 'will not default'
    [200000, 2, 1, 2, 40, 1, 2, 3, 4, 5, 6, 50000, 49000, 48000, 47000, 46000, 45000, 4000, 4100, 4200, 4300, 4400, 4500],  # Sample 2: Should predict 'will default'
    [150000, 1, 3, 1, 30, -1, -1, -1, -1, -1, -1, 30000, 29000, 28000, 27000, 26000, 25000, 3000, 3100, 3200, 3300, 3400, 3500]  # Sample 3: Should predict 'will not default'
])

# Call the function to make predictions for example data
example_prediction = load_model_and_predict(input_data)
print("Example Predictions:", example_prediction)

"""# **Deployment of the model**"""

#Run this following code using streamlit
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