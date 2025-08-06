import streamlit as st
from predictor_model import load_and_prepare_data, train_models, evaluate_models, predict

# Load and prepare the data
X_train, X_test, y_train, y_test = load_and_prepare_data("RELIANCE.NS_data.csv")

# Train the models
lr_model, rf_model = train_models(X_train, y_train)

# Evaluate models
lr_mse, lr_r2, _ = evaluate_models(lr_model, X_test, y_test)
rf_mse, rf_r2, _ = evaluate_models(rf_model, X_test, y_test)

# Streamlit UI
st.title("Stock Close Price Predictor")
st.write("Enter stock features to predict the Close price")

open_val = st.number_input("Open Price", format="%.2f")
high_val = st.number_input("High Price", format="%.2f")
low_val = st.number_input("Low Price", format="%.2f")
volume_val = st.number_input("Volume", format="%d")

model_choice = st.selectbox("Choose model", ["Linear Regression", "Random Forest"])

if st.button("Predict"):
    if model_choice == "Linear Regression":
        result = predict(lr_model, open_val, high_val, low_val, volume_val)
    else:
        result = predict(rf_model, open_val, high_val, low_val, volume_val)
    st.success(f"📈 Predicted Close Price: {result:.2f}")

# Show evaluation metrics
st.write("### Model Evaluation")
st.write(f"Linear Regression - MSE: {lr_mse:.2f}, R²: {lr_r2:.2f}")
st.write(f"Random Forest - MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}")
