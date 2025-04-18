import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
model = pickle.load(open("xgboost_churn_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# Title
st.title("🔍 Customer Churn Prediction App")
st.markdown("Use this tool to predict if a customer will churn based on their profile.")

# Layout split into 2 columns
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)

with col2:
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

# Initialize session state to store predictions
if "history" not in st.session_state:
    st.session_state.history = []

# Predict + Save
if st.button("🔎 Predict Churn"):
    prediction = model.predict(input_data)
    st.subheader("📊 Prediction Result:")
    
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    # Save result into history
    result = {
        "Credit Score": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "Products": products,
        "Has Card": has_card,
        "Is Active": is_active,
        "Salary": salary,
        "Prediction": "Churn" if prediction[0] == 1 else "Stay"
    }
    st.session_state.history.append(result)

# Show prediction history and download option
if st.session_state.history:
    st.subheader("📝 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
