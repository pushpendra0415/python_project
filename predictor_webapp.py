import streamlit as st
from predictor_model import load_and_prepare_data, train_models, evaluate_models, predict

# Load data
X_train, X_test, y_train, y_test, df = load_and_prepare_data("RELIANCE.NS_data_updated.csv")

# Train models
lr_model, rf_model = train_models(X_train, y_train)
# Evaluate
results = evaluate_models({"Linear Regression": lr_model, "Random Forest": rf_model}, X_test, y_test)

st.title("Stock Price Prediction (Today/Next Day)")
st.write("This app predicts **today's closing price** using today's or tomorrow's data.")

# Show model scores
st.subheader("Model Performance")
st.write(results)

# Input form
st.subheader("Enter Stock Data")

# Initialize session state (so defaults load only once)
if "open_val" not in st.session_state:
    st.session_state.open_val = float(df['Open'].iloc[-1])
if "high_val" not in st.session_state:
    st.session_state.high_val = float(df['High'].iloc[-1])
if "low_val" not in st.session_state:
    st.session_state.low_val = float(df['Low'].iloc[-1])
if "volume_val" not in st.session_state:
    st.session_state.volume_val = float(df['Volume'].iloc[-1])
if "prev_close_val" not in st.session_state:
    st.session_state.prev_close_val = float(df['Previous Close'].iloc[-1])

# Editable inputs
open_val = st.number_input("Open Price", value=st.session_state.open_val, step=0.01, format="%.2f")
high_val = st.number_input("High Price", value=st.session_state.high_val, step=0.01, format="%.2f")
low_val = st.number_input("Low Price", value=st.session_state.low_val, step=0.01, format="%.2f")
volume_val = st.number_input("Volume", value=st.session_state.volume_val, step=1000.0, format="%.2f")
prev_close_val = st.number_input("Previous Close Price", value=st.session_state.prev_close_val, step=0.01, format="%.2f")

# Prediction
if st.button("Predict Today's Close Price"):
    predicted_price = predict(lr_model, open_val, high_val, low_val, volume_val, prev_close_val)
    st.success(f"Predicted Close Price: ₹{predicted_price:.2f}")

