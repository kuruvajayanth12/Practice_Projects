import streamlit as st
import numpy as np
import joblib

# load model
model = joblib.load("loan_model.pkl")

st.title("Loan Approval Prediction")

st.write("Enter Applicant Details")

income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")
credit_history = st.selectbox("Credit History", [0,1])
dependents = st.number_input("Dependents")

if st.button("Predict Loan Status"):
    
    input_data = np.array([[income, loan_amount, credit_history, dependents]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")
