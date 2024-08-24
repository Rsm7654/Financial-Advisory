
import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize Faker
fake = Faker()

# Number of samples
n_samples = 1000

# ... (rest of the code for data generation remains the same)

# Train the model (same as before)
# ...

# Streamlit app
st.title("Client Satisfaction Prediction")

# Collect client data through Streamlit input widgets
client_data = {}
for column in synthetic_data.columns:
    if column != 'Client Satisfaction':
        if column in ['Gender', 'Marital Status', 'Employment Status', 'Education Level', 'Residential Status',
                       'Loan Type', 'Risk Tolerance', 'Spending Habits', 'Saving Habits', 'Investment Activity']:
            # Dropdown for categorical features
            unique_values = synthetic_data[column].unique()
            client_data[column] = [st.selectbox(f"Select {column}", ['skip'] + list(unique_values))]
        else:
            # Text input for numerical features
            client_data[column] = [st.text_input(f"Enter {column} (or type 'skip')", "")]

# Create DataFrame and preprocess
new_client_df = pd.DataFrame(client_data)
for col in new_client_df.select_dtypes(include=['object']):
    if col in label_encoders:
        new_client_df[col] = label_encoders[col].transform(new_client_df[col])

numerical_features = new_client_df.select_dtypes(include=['number'])
new_client_df[numerical_features.columns] = scaler.transform(numerical_features)

# Make prediction and display
if st.button("Predict"):
    prediction = model.predict(new_client_df)
    st.write(f"Predicted Client Satisfaction: {prediction[0]}")

