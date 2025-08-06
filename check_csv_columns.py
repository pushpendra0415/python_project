import pandas as pd

data = pd.read_csv("RELIANCE.NS_data.csv")
print("📄 Columns in your CSV:")
print(data.columns.tolist())

print("\n🧾 First 5 rows:")
print(data.head())
