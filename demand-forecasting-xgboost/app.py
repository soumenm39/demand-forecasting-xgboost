import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_resource

def load_model():
    with open('Xgboost_demand_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, label_encoder

model, label_encoder = load_model()

st.title("Demand Prediction App")

st.divider()

st.header("Input Features")

price = st.number_input("Price", min_value=0.0, value = 50.0)
discount = st.number_input("Discount %", min_value=0.0, max_value = 100.0, value = 10.0)
inventory_level = st.number_input("Inventory Level", min_value=0, value = 100)
promotion = st.selectbox("Promotion", [0,1])
competitor_pricing = st.number_input("Competitor Price", min_value=0.0, value = 50.0)
category = st.selectbox("Category", 
                        label_encoder["Category"].classes_.tolist()
                        )

input_data = pd.DataFrame({
    'Price': [price],
    'Discount': [discount],
    'Inventory Level': [inventory_level],
    'Promotion': [promotion],
    'Competitor Pricing': [competitor_pricing],
    'Category': [category]
})

for col, encoder in label_encoder.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

st.divider()

if st.button("Predict Demand"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Demand: {prediction[0]:.2f} Units")

