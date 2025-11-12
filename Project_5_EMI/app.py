import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    reg_model = joblib.load("best_model.pkl")
    clf_model = joblib.load("best_classification_model.pkl")
    return reg_model, clf_model

reg_model, clf_model = load_models()

# ------------------------------
# Realistic default values for placeholders
# ------------------------------
# You can replace these with actual median values from your training dataset
numeric_defaults = {
    'current_emi_amount': 10000,
    'school_fees': 5000,
    'college_fees': 10000,
    'travel_expenses': 3000,
    'groceries_utilities': 7000,
    'other_monthly_expenses': 2000,
    'emergency_fund': 20000,
    'debt_to_income': 0.3,
    'expenses_to_income': 0.5,
    'dependents': 0,
    'max_monthly_emi': 0  # ignored by regression but included in pipeline
}

categorical_defaults = {
    'gender': 'Male',
    'marital_status': 'Single',
    'education': 'Bachelors',
    'employment_type': 'Salaried',
    'house_type': 'Owned',
    'company_type': 'Private',
    'emi_scenario': 'Standard',
    'high_risk': 'No'
}

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="EMI Dashboard", layout="wide")
st.title("🏦 EMI Eligibility & Max Monthly EMI Predictor")
st.markdown("This app predicts EMI eligibility and maximum monthly EMI using key financial and employment details.")

# ------------------------------
# User Input: Key Features Only
# ------------------------------
st.header("📋 Enter Applicant Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD", "Other"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Contract", "Unemployed"])
    years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=40, value=5)
    house_type = st.selectbox("House Type", ["Owned", "Rented", "Company Provided", "Other"])
    monthly_rent = st.number_input("Monthly Rent", min_value=0, value=10000, step=1000)

with col2:
    monthly_salary = st.number_input("Monthly Salary", min_value=0, value=50000, step=500)
    family_size = st.number_input("Family Size", min_value=1, max_value=15, value=3)
    existing_loans = st.number_input("Existing Loans (Total Amount)", min_value=0, value=100000, step=5000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    bank_balance = st.number_input("Current Bank Balance", min_value=0, value=20000, step=1000)
    requested_amount = st.number_input("Requested Loan Amount", min_value=0, value=200000, step=5000)
    requested_tenure = st.number_input("Requested Tenure (Months)", min_value=6, max_value=120, value=24)

# ------------------------------
# Build Input DataFrame
# ------------------------------
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'marital_status': [marital_status],
    'education': [education],
    'monthly_salary': [monthly_salary],
    'employment_type': [employment_type],
    'years_of_employment': [years_of_employment],
    'house_type': [house_type],
    'monthly_rent': [monthly_rent],
    'family_size': [family_size],
    'existing_loans': [existing_loans],
    'credit_score': [credit_score],
    'bank_balance': [bank_balance],
    'requested_amount': [requested_amount],
    'requested_tenure': [requested_tenure]
})

st.subheader("🔍 Input Summary")
st.dataframe(input_data)

# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Eligibility & EMI"):
    try:
        # Fill missing columns with realistic defaults
        required_columns = reg_model.feature_names_in_
        for col in required_columns:
            if col not in input_data.columns:
                if col in numeric_defaults:
                    input_data[col] = numeric_defaults[col]
                elif col in categorical_defaults:
                    input_data[col] = categorical_defaults[col]
                else:
                    input_data[col] = 0  # fallback

        # Reorder columns
        input_data = input_data[required_columns]

        # ------------------------------
        # Predictions
        # ------------------------------
        eligibility_pred = clf_model.predict(input_data)[0]
        eligibility_proba = clf_model.predict_proba(input_data)[0][1] if hasattr(clf_model, "predict_proba") else None
        max_emi_pred = reg_model.predict(input_data)[0]

        # ------------------------------
        # Display Results
        # ------------------------------
        st.markdown("---")
        st.header("🧭 Prediction Results")

        if eligibility_pred == 1:
            st.success("✅ Eligible for EMI")
        else:
            st.error("❌ Not Eligible for EMI")

        if eligibility_proba is not None:
            st.info(f"**Eligibility Probability:** {eligibility_proba:.2%}")

        st.metric(label="Predicted Maximum Monthly EMI", value=f"₹ {max_emi_pred:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and your trained ML models.")
