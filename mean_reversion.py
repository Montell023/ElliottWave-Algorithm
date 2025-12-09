import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('BTC-USD_5minute_data.csv', parse_dates=True, index_col=0)

# Select last day of data
data_last_day = data.tail(24 * 12)  # Assuming 24 data points per day (5-min intervals)

# Input parameters
length = 20
mult = 2.0

# Calculate mean and standard deviation
mean_price = data_last_day['Close'].rolling(window=length).mean()
std_price = data_last_day['Close'].rolling(window=length).std()

# Identify points where the price reverts to the mean
data_last_day['AboveMean'] = np.where(data_last_day['Close'] > mean_price + mult * std_price, data_last_day['Close'], np.nan)
data_last_day['BelowMean'] = np.where(data_last_day['Close'] < mean_price - mult * std_price, data_last_day['Close'], np.nan)

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(data_last_day.index, data_last_day['Close'], label='Close Price', color='blue')
plt.plot(mean_price.index, mean_price, label='Mean', color='red')
plt.plot(mean_price.index, mean_price + mult * std_price, label='Upper Band', color='green', linestyle='--')
plt.plot(mean_price.index, mean_price - mult * std_price, label='Lower Band', color='green', linestyle='--')
plt.scatter(data_last_day.index, data_last_day['AboveMean'], color='red', marker='o', label='Above Mean')
plt.scatter(data_last_day.index, data_last_day['BelowMean'], color='green', marker='o', label='Below Mean')

plt.title('BTC/USD Close Price with Mean and Standard Deviation Bands (Last Day)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
