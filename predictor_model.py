# predictor_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Fix column names if needed
    df.columns = [col.strip() for col in df.columns]

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only needed columns

    # Remove missing values
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf

def evaluate_models(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def predict(model, open_val, high_val, low_val, volume_val):
    input_data = [[open_val, high_val, low_val, volume_val]]
    return model.predict(input_data)[0]
