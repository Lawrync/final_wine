#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open("best_random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the dataset for reference (optional)
@st.cache_data
def load_data():
    return pd.read_csv("wine_quality_no_outliers.csv")

# Main function to run the app
def main():
    # Page title
    st.title("Wine Quality Prediction")

    # Load model
    model = load_model()

    # Sidebar with user inputs for wine characteristics
    st.sidebar.title("Enter Wine Characteristics")
    fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value=0.0, max_value=15.0, step=0.1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.01)
    citric_acid = st.sidebar.slider("Citric Acid", min_value=0.0, max_value=1.0, step=0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar", min_value=0.0, max_value=20.0, step=0.1)
    chlorides = st.sidebar.slider("Chlorides", min_value=0.0, max_value=1.0, step=0.01)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", min_value=0, max_value=100, step=1)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", min_value=0, max_value=300, step=1)
    density = st.sidebar.slider("Density", min_value=0.990, max_value=1.005, step=0.0001)
    pH = st.sidebar.slider("pH", min_value=2.0, max_value=4.0, step=0.01)
    sulphates = st.sidebar.slider("Sulphates", min_value=0.0, max_value=2.0, step=0.01)
    alcohol = st.sidebar.slider("Alcohol", min_value=0.0, max_value=20.0, step=0.1)

    # Make prediction based on user inputs
    wine_features = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol
    ]]

    # Predict quality using the loaded model
    prediction = model.predict(wine_features)

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"The predicted wine quality is: {prediction[0]} (range: 3 to 9)")

if __name__ == "__main__":
    main()

