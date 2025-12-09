import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import coint

# Read the BTC/USD and ETH/USD data from CSV files
btc_data = pd.read_csv('BTC-USD_5minute_data.csv', parse_dates=True, index_col=0)
eth_data = pd.read_csv('ETH-USD_5minute_data.csv', parse_dates=True, index_col=0)

# Extract the 'Close' prices
btc_close = btc_data['Close']
eth_close = eth_data['Close']

# Perform the CADF test
cadf_result = ts.coint(btc_close, eth_close)

# Print the results
print('CADF Test Statistic:', cadf_result[0])
print('P-value:', cadf_result[1])
print('Critical Values:')
critical_values = {key: value for key, value in zip(['1%', '5%', '10%'], cadf_result[2])}
for key, value in critical_values.items():
    print(f'{key}: {value}')

# Plot the price series
plt.figure(figsize=(12, 6))
plt.plot(btc_close, label='BTC/USD', color='blue')
plt.plot(eth_close, label='ETH/USD', color='red')
plt.title('BTC/USD and ETH/USD Price Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# CADF Test Statistic: -0.7609206983154803
# P-value: 0.9396492675170164
# Critical Values:
# 1%: -3.897074931797408
# 5%: -3.3364841905570586
# 10%: -3.0446958473025085
