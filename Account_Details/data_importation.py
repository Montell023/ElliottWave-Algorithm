import yfinance as yf
from datetime import datetime, timedelta

def get_data(symbol):
    # Calculate the start date (60 days ago from today)
    start_date = datetime.now() - timedelta(days=7)
    
    # Download 1-minute interval data from Yahoo Finance
    data = yf.download(symbol, interval='1m', start=start_date.strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))

    return data

# Define symbol for Bitcoin
symbol = 'BTC-USD'

# Get data for BTC-USD
data = get_data(symbol)
print(f"{symbol} 1-minute data:")
print(data)

# Save data to CSV file
filename = f"{symbol}_1minute_data.csv"
data.to_csv(filename)
print(f"Data saved to {filename}")
