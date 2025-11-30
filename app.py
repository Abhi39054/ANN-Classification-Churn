import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model(Path("models") / "model.h5")

# Load the scaler and encoder

with open(Path("models") / "label_encoder_gender.pkl", "rb") as file:
    label_enocder_gender = pickle.load(file)

with open(Path("models") / "ohe_encoder_geography.pkl", "rb") as file:
    ohe_encoder_geography = pickle.load(file)

with open(Path("models") / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Define the Streamlit app
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox("Geography", ohe_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_enocder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (years)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1]) 


input_data = {
    "CreditScore":credit_score,
    "Geography":geography,
    "Gender":gender,
    "Age":age,
    "Tenure":tenure,
    "Balance":balance,
    "NumOfProducts":num_of_products,
    "HasCrCard":has_cr_card,
    "IsActiveMember":is_active_member,
    "EstimatedSalary":estimated_salary
}

input_df = pd.DataFrame([input_data])

# Preprocess the input data

def preprocess_input(data):
    
    # Apply label encoder to Gender
    data["Gender"] = label_enocder_gender.transform(data["Gender"])

    # Apply OneHotEncoder on Geography

    geo_df = pd.DataFrame(ohe_encoder_geography.transform([[input_data["Geography"]]]), columns=ohe_encoder_geography.get_feature_names_out())
    data = pd.concat([data.drop("Geography", axis=1), geo_df], axis=1)


    # Apply Standard Scaler
    X = scaler.transform(data)

    return X

if st.button("Predict Churn"):
    output = model.predict(preprocess_input(input_df))
    st.write("Churn Prediction:", "Yes" if output[0][0] > 0.5 else "No")
    st.write("Churn Prediction (Probability):", output[0][0])