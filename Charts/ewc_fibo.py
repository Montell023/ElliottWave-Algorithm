import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

# Identify peaks and valleys
peaks, _ = find_peaks(data_last_day['Close'], prominence=0.1)  # Adjust prominence threshold as needed
valleys, _ = find_peaks(-data_last_day['Close'], prominence=0.1)  # Adjust prominence threshold as needed

# Plotting
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(data_last_day.index, data_last_day['Close'], label='Close Price', color='blue')
ax.plot(mean_price.index, mean_price, label='Mean', color='red')
ax.plot(mean_price.index, mean_price + mult * std_price, label='Upper Band', color='green', linestyle='--')
ax.plot(mean_price.index, mean_price - mult * std_price, label='Lower Band', color='green', linestyle='--')
ax.scatter(data_last_day.index, data_last_day['AboveMean'], color='red', marker='o', label='Above Mean')
ax.scatter(data_last_day.index, data_last_day['BelowMean'], color='green', marker='o', label='Below Mean')
ax.scatter(data_last_day.index[peaks], data_last_day['Close'][peaks], color='red', marker='^', label='Peaks')
ax.scatter(data_last_day.index[valleys], data_last_day['Close'][valleys], color='green', marker='v', label='Valleys')

# Fibonacci retracement levels
high_price = 69010  # High price for Fibonacci retracement
low_price = 68508  # Low price for Fibonacci retracement
range_price = high_price - low_price

fibonacci_levels = [0.618, 1.618, 2.618, 4.236]  # Fibonacci retracement levels
fibonacci_colors = ['orange', 'red', 'green', 'blue']  # Corresponding colors for levels

for level, color in zip(fibonacci_levels, fibonacci_colors):
    retracement_price = low_price + level * range_price
    ax.axhline(retracement_price, color=color, linestyle='--')
    ax.text(data_last_day.index[0], retracement_price, f'{level * 100:.3f}%', color=color, fontsize=8, va='center')

# Plot the range
ax.axhline(high_price, color='blue', linestyle='--')
ax.axhline(low_price, color='blue', linestyle='--')
ax.text(data_last_day.index[0], high_price, f'Highest High', color='blue', fontsize=8, va='center')
ax.text(data_last_day.index[0], low_price, f'Lowest Low', color='blue', fontsize=8, va='center')

plt.title('BTC/USD Close Price with Mean, Standard Deviation Bands, Peaks, and Valleys (Last Day)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
#Take Note of below mean as starting point#