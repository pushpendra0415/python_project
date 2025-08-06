import yfinance as yf

stock_symbol = 'RELIANCE.NS'
start_date = '2020-01-01'
end_date = '2025-08-01'

data = yf.download(stock_symbol, start=start_date, end=end_date)
data.to_csv(f"{stock_symbol}_data.csv")
print(f"Data saved to {stock_symbol}_data.csv")
