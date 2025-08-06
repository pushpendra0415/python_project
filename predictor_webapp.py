# predictor_webapp.py

import streamlit as st
from predictor_model import load_and_prepare_data, train_models, evaluate_models, predict

# Load and split data
X_train, X_test, y_train, y_test = load_and_prepare_data("RELIANCE.NS_data.csv")

# Train models
lr_model, rf_model = train_models(X_train, y_train)

# Evaluate models
lr_mse, lr_r2 = evaluate_models(lr_model, X_test, y_test)
rf_mse, rf_r2 = evaluate_models(rf_model, X_test, y_test)

# Streamlit App UI
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Predictor (RELIANCE.NS)")
st.markdown("Enter Open, High, Low, and Volume to predict the Close price.")

# User input
open_val = st.number_input("Open Price", min_value=0.0, value=1000.0)
high_val = st.number_input("High Price", min_value=0.0, value=1020.0)
low_val = st.number_input("Low Price", min_value=0.0, value=980.0)
volume_val = st.number_input("Volume", min_value=0.0, value=5000000.0)

model_choice = st.selectbox("Choose model", ["Linear Regression", "Random Forest"])

if st.button("Predict"):
    if model_choice == "Linear Regression":
        pred = predict(lr_model, open_val, high_val, low_val, volume_val)
    else:
        pred = predict(rf_model, open_val, high_val, low_val, volume_val)

    st.success(f"📊 Predicted Close Price: ₹{pred:.2f}")

# Show evaluation metrics
with st.expander("🔍 Model Performance"):
    st.write("### Linear Regression")
    st.write(f"• Mean Squared Error (MSE): {lr_mse:.2f}")
    st.write(f"• R² Score: {lr_r2:.4f}")
    st.write("### Random Forest")
    st.write(f"• Mean Squared Error (MSE): {rf_mse:.2f}")
    st.write(f"• R² Score: {rf_r2:.4f}")
