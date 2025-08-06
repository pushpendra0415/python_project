import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, skiprows=1)  # Skip first metadata row
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']  # Set correct column names
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only relevant columns
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(random_state=42)
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    return lr_model, rf_model

def evaluate_models(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def predict(model, open_price, high_price, low_price, volume):
    features = [[open_price, high_price, low_price, volume]]
    return model.predict(features)[0]
