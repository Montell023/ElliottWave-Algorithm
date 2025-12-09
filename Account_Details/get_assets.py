from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# Assign your Alpaca API key and secret to variables
APCA_API_KEY_ID = 'PK9AKCI5VTB983095JKH'
APCA_API_SECRET_KEY = 'BRpsT5XAxP086ffLNhQqVoQKemohnessY1U9h37w'

trading_client = TradingClient('PK9AKCI5VTB983095JKH', 'BRpsT5XAxP086ffLNhQqVoQKemohnessY1U9h37w')

# search for US equities
search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)

assets = trading_client.get_all_assets(search_params)

for asset in assets:
    print(asset)

#The above code lets us retrieve tradable assets (you’ll receive a list of US equities)

#Import config.py to use by API key and secret ...