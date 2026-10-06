import streamlit as st
import os
import joblib
import pandas as pd


st.set_page_config(page_title="Credit Risk Prediction", page_icon=":money_with_wings:", layout="wide")

MODEL_PATH = "models\\model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please train the model first.")

model = joblib.load(MODEL_PATH)

st.title("Credit Risk Prediction")
st.write("Enter the below details of the applicant to predict the credit risk.")

annual_income = st.number_input("Annual Income", min_value=5000, step=10, key="annual_income")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1, key="credit_score")
loan_amount = st.number_input("Loan Amount", min_value=1000, step=10, key="loan_amount")
existing_debt = st.number_input("Existing Debt", min_value=0, step=10, key="existing_debt")
late_payments = st.number_input("Late Payments", min_value=0, step=1, key="late_payments")
previous_defaults = st.number_input("Previous Defaults", min_value=0, step=1, key="previous_defaults")

if st.button("Predict", type="primary"):
    new_customer = pd.DataFrame({
        "annual_income": [annual_income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "existing_debt": [existing_debt],
        "late_payments": [late_payments],
        "previous_defaults": [previous_defaults]
    })
    prediction = model.predict(new_customer)[0]
    st.success(f"The predicted credit risk is: {prediction}")