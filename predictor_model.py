import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_and_prepare_data(csv_file):
    # Load cleaned data
    df = pd.read_csv(csv_file)

    # Ensure correct dtypes
    for col in ['Close', 'High', 'Low', 'Open', 'Volume', 'Previous Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaNs
    df.dropna(inplace=True)

    # Features & target
    X = df[['Open', 'High', 'Low', 'Volume', 'Previous Close']]
    y = df['Close']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df

def train_models(X_train, y_train):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    return lr_model, rf_model

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {"MSE": mse, "R2": r2}
    return results

def predict(model, open_val, high_val, low_val, volume_val, prev_close_val):
    input_features = np.array([[open_val, high_val, low_val, volume_val, prev_close_val]])
    return model.predict(input_features)[0]
