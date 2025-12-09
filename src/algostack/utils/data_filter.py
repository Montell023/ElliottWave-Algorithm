import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, parse_dates=['Datetime'])

    # Ensure 'Datetime' column is present
    if 'Datetime' not in df.columns:
        raise KeyError("The 'Datetime' column is missing from the CSV file.")

    # Set Datetime column as index
    df.set_index('Datetime', inplace=True)

    # Ensure the index is timezone-aware and in UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')

    # Convert timezone to America/New_York for Lumibot
    df.index = df.index.tz_convert('America/New_York')

    # Rename columns to match expected format
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Remove the 'Adj Close' column if it exists
    if 'Adj Close' in df.columns:
        df.drop('Adj Close', axis=1, inplace=True)

    # Ensure all data columns are float type
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Remove rows with NaN values
    df.dropna(inplace=True)

    # Remove rows with zero volume (if you want to keep only traded periods)
    df = df[df['volume'] > 0]

    # Ensure logical order of OHLC prices
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Sort by datetime
    df.sort_index(inplace=True)

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Rename index to lowercase 'datetime'
    df.index.name = 'datetime'

    # Save the cleaned data to a new CSV file
    df.to_csv(output_file)

    print(f"Cleaned data saved to {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(df.head())
    print(f"Index name: {df.index.name}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Column types: {df.dtypes}")

# Usage
input_file = 'BTC-USD_1minute_data.csv'  # Your original CSV file
output_file = 'BTC-USD_1minute_data_cleaned.csv'  # The new, cleaned CSV file

clean_csv_data(input_file, output_file)