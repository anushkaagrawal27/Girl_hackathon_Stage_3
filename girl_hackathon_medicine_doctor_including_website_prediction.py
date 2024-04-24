import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

data = pd.read_csv('DATA_MODIFIED.csv', encoding='utf-8')
data2 = pd.read_csv('Doctors.csv', encoding='utf-8')

data.isnull().sum()
missing_values = data.isnull().sum()
data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'], errors='coerce', dayfirst=True)

median_date = data['DateOfBirth'].dropna().median()

data['DateOfBirth'].fillna(median_date, inplace=True)

# Categorical Columns (Gender, Symptoms, Causes, Disease, Medicine):I impute missing values with the mode (most frequent value).
categorical_columns = ['Gender', 'Symptoms', 'Causes', 'Disease', 'Medicine','Doctor Specialist']
for column in categorical_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Check for missing values in each column
missing_values = data.isnull().sum()

# Display columns with missing values
#print(missing_values[missing_values > 0])

# Handle the missing values in column Name by imputation method
data['Name'].fillna('Unknown', inplace=True)

data.isnull().sum()

missing_values = data.isnull().sum()

# Display columns with missing values
#print(missing_values[missing_values > 0])

# Since most of my data is categorical, i want to encode it into a numerical format suitable for machine learning models.
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Exploratory Data Analysis (EDA)
# Class distribution for Medicine
# sns.countplot(data=data, x='Medicine')
# plt.title('Distribution of Medicine')
#plt.show()

# Correlation heatmap for numerical columns
# numeric_columns = ['DateOfBirth']
# sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# #plt.show()

# Data Processing : We need to encode categorical variables for the machine learning models.
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# splitting the datset: We divide the dataset into training and testing sets.
from sklearn.model_selection import train_test_split

X = data.drop(['Name', 'Medicine', 'Disease', 'Doctor Specialist', 'DateOfBirth'], axis=1)
y_disease = data['Disease']
y_medicine = data['Medicine']
y_doctor = data['Doctor Specialist']
X_train, X_test, y_disease_train, y_disease_test, y_medicine_train, y_medicine_test, y_doctor_train, y_doctor_test = train_test_split(X, y_disease, y_medicine, y_doctor, test_size=0.5, random_state=42)

dt_classifier_disease = DecisionTreeClassifier(random_state=42)
dt_classifier_disease.fit(X_train, y_disease_train)
rf_classifier_medicine = RandomForestClassifier(random_state=42)
rf_classifier_medicine.fit(X_train, y_medicine_train)

rf_classifier_doctor = RandomForestClassifier(random_state=42)
rf_classifier_doctor.fit(X_train, y_doctor_train)
joblib.dump(dt_classifier_disease, 'dt_classifier_disease.joblib')
joblib.dump(rf_classifier_medicine, 'rf_classifier_medicine.joblib')
joblib.dump(rf_classifier_doctor, 'rf_classifier_doctor.joblib')

# Load the models
dt_classifier_disease = joblib.load('dt_classifier_disease.joblib')
rf_classifier_medicine = joblib.load('rf_classifier_medicine.joblib')
rf_classifier_doctor = joblib.load('rf_classifier_doctor.joblib')
new_data = {
    'Name': ['Alice'],
    'Gender': ['2'],
    'DateOfBirth': ['01-01-1980'],
    'Symptoms': ['23'],
    'Causes': ['52']
}


new_data = pd.DataFrame(new_data)

# Predictions on new_data
# Assuming 'new_data' is your new dataset
# Preprocess 'new_data' similar to how you preprocessed the training data

# Remove 'DateOfBirth' column
new_data = new_data.drop('Name', axis=1)
new_data = new_data.drop('DateOfBirth', axis=1)

predicted_disease = dt_classifier_disease.predict(new_data)

# Predict Medicine
predicted_medicine = rf_classifier_medicine.predict(new_data)

# Predict Doctor Specialist
predicted_doctor = rf_classifier_doctor.predict(new_data)

disease = label_encoders['Disease'].inverse_transform(predicted_doctor)
medicine = label_encoders['Medicine'].inverse_transform(predicted_doctor)
doctor = label_encoders['Doctor Specialist'].inverse_transform(predicted_doctor)

print("You are probably suffering from ", disease[0])
print("You should take ", medicine[0])
print("Probably consult a ", doctor[0],"\n")

DoctorRatings = data2[data2['Specialization'] == doctor[0]]

print("Here are some ", doctor[0], ":-\n")

print(DoctorRatings)








import streamlit as st
import pandas as pd
import pickle

st.title("Disease Prediction App")
def get_user_input():
    """
    Collects user input for the prediction.
    """
    input_data = {}
    input_data["Name"] = st.text_input("Enter Name:")  # Add this line to collect the name
    input_data["DateOfBirth"] = st.text_input("Enter Date of Birth:")
    input_data["Symptoms"] = st.text_input("Describe Symptoms:")
    input_data["Causes"] = st.text_input("Causes:")
    input_data["Doctor Specialist"] = st.text_input("Doctor Specialist:")

    # Convert empty strings to None
    for key, value in input_data.items():
        if value == '':
            input_data[key] = None

    return input_data

def prepare_input(user_input):
    input_df = pd.DataFrame([user_input])


    for column in categorical_columns:
        if column in input_df.columns:
            input_df[column] = label_encoders[column].transform(input_df[column])

    return input_df

user_input = get_user_input()
input_df = pd.DataFrame([user_input]) 


input_df = input_df.drop('Name', axis=1)
input_df = input_df.drop('DateOfBirth', axis=1)
# input_df = input_df.drop('Disease', axis=1)
input_df = input_df.drop('Doctor Specialist', axis=1)

predicted_medicine = rf_classifier_medicine.predict(input_df)
target_column_name = 'Medicine'
if target_column_name in label_encoders:
    prediction_label = label_encoders[target_column_name].inverse_transform(predicted_disease)  
    st.write("Medicine Prediction:", prediction_label[0])
else:
    st.write("Prediction label encoder not found.")

predicted_doctor = rf_classifier_doctor.predict(input_df)
target_column_name = 'Doctor Specialist'
if target_column_name in label_encoders:
    prediction_label = label_encoders[target_column_name].inverse_transform(predicted_disease)  
    st.write("Doctor Specialist Prediction:", prediction_label[0])
else:
    st.write("Prediction label encoder not found.")


