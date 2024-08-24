!pip install faker
import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Number of samples
n_samples = 1000

# Generate Personal Information
data = {
    'Client ID': [fake.uuid4() for _ in range(n_samples)],
    'Age': np.random.randint(18, 80, n_samples),
    'Gender': np.random.choice(['Male', 'Female', 'Non-binary', 'Other'], n_samples),
    'Marital Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
    'Dependents': np.random.randint(0, 5, n_samples),
    'Occupation': [fake.job() for _ in range(n_samples)],
    'Annual Income': np.random.randint(20000, 150000, n_samples),
    'Employment Status': np.random.choice(['Employed', 'Unemployed', 'Retired', 'Student'], n_samples),
    'Education Level': np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], n_samples),
    'Residential Status': np.random.choice(['Renting', 'Own Home', 'Mortgage'], n_samples),
}

# Generate Financial Information
financial_data = {
    'Savings Account Balance': np.random.uniform(500, 100000, n_samples),
    'Checking Account Balance': np.random.uniform(100, 50000, n_samples),
    'Credit Score': np.random.randint(300, 850, n_samples),
    'Monthly Expenses': np.random.uniform(500, 5000, n_samples),
    'Loan Type': np.random.choice(['Mortgage', 'Car Loan', 'Personal Loan', 'Student Loan'], n_samples),
    'Loan Balance': np.random.uniform(1000, 300000, n_samples),
    'Monthly Loan Payment': np.random.uniform(50, 1500, n_samples),
    'Stocks': np.random.uniform(0, 100000, n_samples),
    'Bonds': np.random.uniform(0, 50000, n_samples),
    'Mutual Funds': np.random.uniform(0, 75000, n_samples),
    'Real Estate Investments': np.random.uniform(0, 500000, n_samples),
    'Life Insurance': np.random.uniform(0, 1000000, n_samples),
    'Health Insurance': np.random.uniform(0, 50000, n_samples),
    'Home/Auto Insurance': np.random.uniform(0, 100000, n_samples),
    '401(k)/Pension Plan Balance': np.random.uniform(0, 1000000, n_samples),
    'IRA Account Balance': np.random.uniform(0, 500000, n_samples),
    'Tax Bracket': np.random.choice(['10%', '12%', '22%', '24%', '32%', '35%', '37%'], n_samples),
    'Deductions and Credits': np.random.uniform(0, 20000, n_samples),
}

# Generate Goals and Preferences
goals_preferences = {
    'Short-term Goals': [fake.sentence(nb_words=6) for _ in range(n_samples)],
    'Long-term Goals': [fake.sentence(nb_words=8) for _ in range(n_samples)],
    'Risk Tolerance': np.random.choice(['Conservative', 'Moderate', 'Aggressive'], n_samples),
    'Investment Preferences': [fake.bs() for _ in range(n_samples)],
    'Retirement Age Goal': np.random.randint(55, 70, n_samples),
}

# Generate Behavioral Data
behavioral_data = {
    'Spending Habits': np.random.choice(['Luxury', 'Essential', 'Balanced'], n_samples),
    'Saving Habits': np.random.choice(['Frequent', 'Occasional', 'Rare'], n_samples),
    'Investment Activity': np.random.choice(['Active', 'Passive', 'None'], n_samples),
}

# Generate Advising Data
advising_data = {
    'Previous Advising History': [fake.text(max_nb_chars=100) for _ in range(n_samples)],
    'Adviser Notes': [fake.text(max_nb_chars=150) for _ in range(n_samples)],
    'Client Satisfaction': np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied'], n_samples),
}

# Combine all data into a single DataFrame
synthetic_data = pd.DataFrame(data)
synthetic_data = synthetic_data.assign(**financial_data, **goals_preferences, **behavioral_data, **advising_data)


# prompt: building a model by using previous data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you want to predict 'Client Satisfaction' based on other features
# Select features and target variable
features = synthetic_data.drop('Client Satisfaction', axis=1)
target = synthetic_data['Client Satisfaction']

# Encode categorical features
label_encoders = {}
for col in features.select_dtypes(include=['object']):
  le = LabelEncoder()
  features[col] = le.fit_transform(features[col])
  label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numerical_features = features.select_dtypes(include=['number'])
features[numerical_features.columns] = scaler.fit_transform(numerical_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


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

