import streamlit as st
from predictor_model import load_and_prepare_data, train_models, evaluate_models, predict

# Load data
X_train, X_test, y_train, y_test, df = load_and_prepare_data("RELIANCE.NS_data_updated.csv")

# Train models
lr_model, rf_model = train_models(X_train, y_train)

# Prediction
if st.button("Predict Today's Close Price"):
    predicted_price = predict(lr_model, open_val, high_val, low_val, volume_val, prev_close_val)
    st.success(f"Predicted Close Price: ₹{predicted_price:.2f}")

