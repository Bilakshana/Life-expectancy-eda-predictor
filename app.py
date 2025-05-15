import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Title
st.title("🌍 Life Expectancy Predictor")

# Load trained model
model = joblib.load("life_expectancy_model.pkl")

# Load feature column names
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Input widgets
st.header("Enter Health and Economic Indicators")

country = st.selectbox("Country (for demo - does not affect prediction)", ["Demo"])  # Dummy, because encoded
status = st.selectbox("Development Status", ["Developing", "Developed"])
year = st.slider("Year", 2000, 2015, 2010)
adult_mortality = st.number_input("Adult Mortality", 0.0, 1000.0, 150.0)
infant_deaths = st.number_input("Infant Deaths", 0.0, 500.0, 10.0)
alcohol = st.number_input("Alcohol Consumption", 0.0, 20.0, 5.0)
percentage_expenditure = st.number_input("Percentage Expenditure", 0.0, 100000.0, 1000.0)
hepatitis_b = st.number_input("Hepatitis B (%)", 0.0, 100.0, 80.0)
measles = st.number_input("Measles", 0.0, 10000.0, 500.0)
bmi = st.number_input("BMI", 10.0, 50.0, 20.0)
under_five_deaths = st.number_input("Under-Five Deaths", 0.0, 1000.0, 20.0)
polio = st.number_input("Polio (%)", 0.0, 100.0, 90.0)
total_expenditure = st.number_input("Total Expenditure", 0.0, 20.0, 5.0)
diphtheria = st.number_input("Diphtheria (%)", 0.0, 100.0, 85.0)
hiv_aids = st.number_input("HIV/AIDS", 0.0, 100.0, 0.1)
gdp = st.number_input("GDP", 0.0, 200000.0, 1000.0)
population = st.number_input("Population", 0.0, 1e10, 1e6)
thinness_1_19 = st.number_input("Thinness 1-19 Years", 0.0, 50.0, 5.0)
thinness_5_9 = st.number_input("Thinness 5-9 Years", 0.0, 50.0, 5.0)
income_composition = st.number_input("Income Composition of Resources", 0.0, 1.0, 0.6)
schooling = st.number_input("Schooling (years)", 0.0, 20.0, 12.0)

# Build input dict
input_data = {
    'year': year,
    'adult mortality': adult_mortality,
    'infant deaths': infant_deaths,
    'alcohol': alcohol,
    'percentage expenditure': percentage_expenditure,
    'hepatitis b': hepatitis_b,
    'measles': measles,
    'bmi': bmi,
    'under-five deaths': under_five_deaths,
    'polio': polio,
    'total expenditure': total_expenditure,
    'diphtheria': diphtheria,
    'hiv/aids': hiv_aids,
    'gdp': gdp,
    'population': population,
    'thinness 1-19 years': thinness_1_19,
    'thinness 5-9 years': thinness_5_9,
    'income composition of resources': income_composition,
    'schooling': schooling,
    'status_Developed': 1 if status == 'Developed' else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Align with training columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Predict
if st.button("Predict Life Expectancy"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Life Expectancy: {prediction:.2f} years")
