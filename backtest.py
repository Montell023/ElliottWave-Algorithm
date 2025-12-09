import pandas as pd
from scipy.signal import find_peaks
from lumibot.backtesting import PandasDataBacktesting, BacktestingBroker
from lumibot.entities import Asset, Data
from lumibot.strategies import Strategy
from lumibot.traders import Trader

# Import your Elliott Wave algorithms & Fibonacci Levels
from Trends.correction_inverse import correction_inverse, calculate_fibonacci_levels as fib_ci, calculate_extension_levels as ext_ci
from Trends.correction import correction, calculate_fibonacci_levels as fib_c, calculate_extension_levels as ext_c
from Trends.impulse_inverse import downtrend, calculate_fibonacci_levels as fib_di
from Trends.impulse import uptrend, calculate_fibonacci_levels as fib_u

print("Imported Elliott Wave algorithms and Fibonacci level functions.")

def load_data(file_path):
    """
    Load data from the cleaned CSV file and return a pandas DataFrame.
    """
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert('America/New_York')
    
    print(f"Loaded data from {file_path}:")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(df.head())
    
    return df

class ElliottWaveStrategy(Strategy):
    def initialize(self):
        self.position_size = 0.01  # 1% of total equity per trade
        self.symbol = 'BTC-USD'
        self.set_market('24/7')
        self.logs = []
        self.log(f"Strategy initialized with symbol: {self.symbol}")

    def on_trading_iteration(self):
        try:
            asset = Asset(symbol="BTC-USD", asset_type=Asset.AssetType.CRYPTO)
            self.process_data(asset)
        except Exception as e:
            self.log(f"Error in trading iteration: {str(e)}")

    def process_data(self, asset):
        try:
            self.log(f"Processing data for {asset.symbol}")
            self.log(f"Current datetime: {self.get_datetime()}")
            bars = self.get_historical_prices(asset, 100)
            self.log(f"Bars object: {bars}")
            self.log(f"Bars type: {type(bars)}")

            if bars is None or bars.df.empty:
                self.log(f"No data for {asset.symbol}")
                return

            data = bars.df

            self.log(f"Data for {asset.symbol}:")
            self.log(f"Shape: {data.shape}")
            self.log(f"Date range: {data.index.min()} to {data.index.max()}")

            if len(data) < 20:
                self.log(f"Insufficient data for {asset.symbol}")
                return

            close_prices = data['close']
            peaks, _ = find_peaks(data['high'], prominence=1)
            valleys, _ = find_peaks(-data['low'], prominence=1)

            mean_price = close_prices.rolling(window=20).mean()
            upper_band = mean_price + 2 * close_prices.rolling(window=20).std()
            lower_band = mean_price - 2 * close_prices.rolling(window=20).std()
            above_mean = close_prices > mean_price
            below_mean = close_prices < mean_price

            # Check if data processing reduced usable data significantly
            usable_data_percentage = (len(mean_price.dropna()) / len(data)) * 100
            self.log(f"Usable data after processing: {usable_data_percentage:.2f}%")
            if usable_data_percentage < 80:
                self.log("Warning: Data processing reduced usable data significantly")

            # Ensure all inputs have the same length
            min_length = min(len(close_prices), len(mean_price.dropna()), len(upper_band.dropna()), len(lower_band.dropna()))
            close_prices = close_prices.iloc[-min_length:]
            mean_price = mean_price.dropna().iloc[-min_length:]
            upper_band = upper_band.dropna().iloc[-min_length:]
            lower_band = lower_band.dropna().iloc[-min_length:]
            above_mean = above_mean.iloc[-min_length:]
            below_mean = below_mean.iloc[-min_length:]

            # Adjust peaks and valleys to match the new length
            peaks = peaks[peaks < min_length]
            valleys = valleys[valleys < min_length]

            # Log the lengths of all relevant values
            self.log(f"Lengths after adjustment:")
            self.log(f"close_prices: {len(close_prices)}")
            self.log(f"mean_price: {len(mean_price)}")
            self.log(f"upper_band: {len(upper_band)}")
            self.log(f"lower_band: {len(lower_band)}")
            self.log(f"above_mean: {len(above_mean)}")
            self.log(f"below_mean: {len(below_mean)}")
            self.log(f"peaks: {len(peaks)}")
            self.log(f"valleys: {len(valleys)}")

            try:
                correction_inv_results = correction_inverse(close_prices, peaks, valleys)
            except Exception as e:
                self.log(f"Error in correction_inverse: {str(e)}")
                correction_inv_results = None

            try:
                correction_results = correction(close_prices, peaks, valleys)
            except Exception as e:
                self.log(f"Error in correction: {str(e)}")
                correction_results = None

            try:
                impulse_inv_results = downtrend(close_prices, mean_price, upper_band, below_mean, peaks, valleys)
            except Exception as e:
                self.log(f"Error in downtrend: {str(e)}")
                impulse_inv_results = None

            try:
                impulse_results = uptrend(close_prices, mean_price, lower_band, above_mean, peaks, valleys)
            except Exception as e:
                self.log(f"Error in uptrend: {str(e)}")
                impulse_results = None

            self.log(f"Correction Inverse Results: {correction_inv_results}")
            self.log(f"Correction Results: {correction_results}")
            self.log(f"Impulse Inverse Results: {impulse_inv_results}")
            self.log(f"Impulse Results: {impulse_results}")

            if all(result is not None for result in [correction_inv_results, correction_results, impulse_inv_results, impulse_results]):
                self.execute_trades(asset, correction_inv_results, correction_results, impulse_inv_results, impulse_results)
            else:
                self.log(f"Warning: One of the results for {asset.symbol} is None.")

        except Exception as e:
            self.log(f"Error processing data for {asset.symbol}: {str(e)}")

    def execute_trades(self, asset, correction_inv, correction, impulse_inv, impulse):
        try:
            # Correction Inverse Logic
            if correction_inv.get('Wave B'):
                self.sell_positions(asset, 3)
            if correction_inv.get('Wave C'):
                self.take_profit(asset, 3)

            # Correction Logic
            if correction.get('Wave B'):
                self.buy_positions(asset, 3)
            if correction.get('Wave C'):
                self.take_profit(asset, 3)

            # Impulse Inverse Logic
            if impulse_inv:
                self.sell_positions(asset, 3)
                self.take_profit(asset, 3)
                self.sell_positions(asset, 2)
                self.take_profit(asset, 2)

            # Impulse Logic
            if impulse:
                self.buy_positions(asset, 3)
                self.take_profit(asset, 3)
                self.buy_positions(asset, 2)
                self.take_profit(asset, 2)
        except Exception as e:
            self.log(f"Error executing trades: {str(e)}")

    def buy_positions(self, asset, num_positions):
        try:
            position_size = self.portfolio_value * self.position_size * num_positions
            order = self.create_order(asset, position_size, "buy")
            self.submit_order(order)
        except Exception as e:
            self.log(f"Error buying positions: {str(e)}")

    def sell_positions(self, asset, num_positions):
        try:
            position_size = self.portfolio_value * self.position_size * num_positions
            order = self.create_order(asset, position_size, "sell")
            self.submit_order(order)
        except Exception as e:
            self.log(f"Error selling positions: {str(e)}")

    def take_profit(self, asset, num_positions):
        try:
            position_size = self.portfolio_value * self.position_size * num_positions
            order = self.create_order(asset, position_size, "sell")
            self.submit_order(order)
        except Exception as e:
            self.log(f"Error taking profit: {str(e)}")

    def log(self, message):
        self.logs.append(message)

# Main execution
if __name__ == "__main__":
    try:
        # Read the data from the cleaned CSV file
        file_path = 'BTC-USD_1minute_data_cleaned.csv'
        df = load_data(file_path)

        # Create the Asset object for BTC-USD
        base_asset = Asset(
            symbol="BTC-USD",
            asset_type=Asset.AssetType.CRYPTO,
        )
        quote_asset = Asset(
            symbol="USD",
            asset_type=Asset.AssetType.FOREX,
        )

        # Create the dictionary to be passed into Lumibot
        pandas_data = {
            base_asset: Data(
                base_asset,
                df,
                timestep="minute",
                quote=quote_asset
            )
        }

        # Print data info for verification
        print("Pandas Data:")
        for asset, data in pandas_data.items():
           print(f"Asset: {asset}")
           print(f"Quote: {data.quote}")
           print(f"Data shape: {data.df.shape}")
           print(f"Data range: {data.df.index.min()} to {data.df.index.max()}")

        # Use the full date range from the data
        backtesting_start = df.index.min()
        backtesting_end = df.index.max()

        print(f"Backtesting from {backtesting_start} to {backtesting_end}")

        # Run the backtesting
        trader = Trader(backtest=True)
        data_source = PandasDataBacktesting(
            pandas_data=pandas_data,
            datetime_start=backtesting_start,
            datetime_end=backtesting_end
        )
        print(f"Data source created: {data_source}")
        broker = BacktestingBroker(data_source)
        strat = ElliottWaveStrategy(
            broker=broker,
            budget=100000,
            timestamp=backtesting_start
        )

        trader.add_strategy(strat)
        trader.run_all(show_plot=True, show_tearsheet=True)

        print("\nBacktest completed.")
        print("\nStrategy Logs:")
        all_logs = strat.logs
        
        # Filter out the length logs
        length_logs = [log for log in all_logs if log.startswith("Lengths after adjustment:") or 
                       any(key in log for key in ['close_prices:', 'mean_price:', 'upper_band:', 
                                                  'lower_band:', 'above_mean:', 'below_mean:', 
                                                  'peaks:', 'valleys:'])]
        
        # Print all logs except length logs
        for log in all_logs:
            if log not in length_logs:
                print(log)
        
        # Print length logs at the very end
        print("\nFinal Length Information:")
        for log in length_logs:
            print(log)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")