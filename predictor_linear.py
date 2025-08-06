import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read and clean the CSV
df = pd.read_csv("RELIANCE.NS_data.csv", skiprows=1)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Check for missing values
print("\n🔍 Missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and check accuracy
predictions = lr_model.predict(X_test)

# Print sample predictions
print("\n🔮 Sample predictions vs actual:")
for i in range(5):
    print(f"Predicted: {predictions[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")
