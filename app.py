import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("xgboost_churn_model.pkl", "rb"))

st.title("🔍 Customer Churn Prediction")

# User Inputs
credit_score = st.slider("Credit Score", 300, 850, 600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0)
products = st.slider("Number of Products", 1, 4, 1)
has_card = st.radio("Has Credit Card?", ["Yes", "No"])
is_active = st.radio("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0)

# Preprocess inputs
geo_dict = {"France": 0, "Spain": 2, "Germany": 1}
gender_dict = {"Female": 0, "Male": 1}

input_data = np.array([[
    credit_score,
    geo_dict[geography],
    gender_dict[gender],
    age,
    tenure,
    balance,
    products,
    1 if has_card == "Yes" else 0,
    1 if is_active == "Yes" else 0,
    salary
]])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("✅ This customer is likely to stay.")
