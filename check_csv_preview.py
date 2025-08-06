import pandas as pd

# Read without header
df = pd.read_csv("RELIANCE.NS_data.csv", header=None)

# Show first 5 rows
print("🧾 First 5 rows of raw file:")
print(df.head(5).to_string(index=False, header=False))
