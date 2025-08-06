import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ✅ Load your CSV file
df = pd.read_csv("RELIANCE.NS_data.csv")

# ✅ Show available columns
print("✅ Columns:", df.columns.tolist())

# ✅ Drop rows where 'Price' or any important column is missing
df = df.dropna(subset=['Price', 'Close', 'High', 'Low', 'Open', 'Volume'])

# ✅ Feature columns and target column
X = df[['Close', 'High', 'Low', 'Open', 'Volume']]
y = df['Price']

# ✅ Check if data is empty
if len(X) == 0 or len(y) == 0:
    raise ValueError("🚫 No data available for training. Please check your CSV file.")

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Make predictions
y_pred = model.predict(X_test)

# ✅ Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Mean Squared Error: {mse:.2f}")

# ✅ Predict the next price (optional example)
latest_data = df[['Close', 'High', 'Low', 'Open', 'Volume']].iloc[-1:]
predicted_price = model.predict(latest_data)
print(f"📈 Predicted next price: {predicted_price[0]:.2f}")
