# alpaca_trade.py
import alpaca_trade_api as tradeapi

# Replace these with your Alpaca API credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def place_order(symbol, qty, side, order_type='market', time_in_force='gtc'):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None
