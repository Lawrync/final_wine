#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("xgboost_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Main function to run the app
def main():
    # Page title
    st.title("Wine Quality Prediction")

    # Load model and scaler
    model = load_model()
    

    # Sidebar with user inputs for wine characteristics
    st.sidebar.title("Enter Wine Characteristics")
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 0.0, 15.0, 7.0)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 2.0, 0.5)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.5)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 20.0, 5.0)
    chlorides = st.sidebar.slider("Chlorides", 0.0, 1.0, 0.1)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 0, 100, 10)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0, 300, 50)
    density = st.sidebar.slider("Density", 0.990, 1.005, 0.995)
    pH = st.sidebar.slider("pH", 2.0, 4.0, 3.2)
    sulphates = st.sidebar.slider("Sulphates", 0.0, 2.0, 0.5)
    alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 10.0)

    # Create a button to make predictions
    if st.sidebar.button("Predict Quality"):
        # Make prediction based on user inputs
        wine_features = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]])


        # Predict quality using the loaded model
        try:
            prediction = model.predict(wine_features)
            # Display prediction
            st.subheader("Prediction:")
            st.write(f"The predicted wine quality is: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()