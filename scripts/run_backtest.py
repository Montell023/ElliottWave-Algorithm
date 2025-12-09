import pandas as pd
import logging
from scipy.signal import find_peaks
from collections import deque
from typing import cast, Optional, Dict, Any

from lumibot.backtesting import PandasDataBacktesting
from lumibot.entities import Asset, Data
from lumibot.strategies import Strategy

from algostack.Trends.correction_inverse import correction_inverse
from algostack.Waves.corrective_abc import corrective
from algostack.Waves.motive_abc import motive

# ═════════ GLOBAL SETUP ═════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("root")

raw = pd.read_csv("data/BTC-USD_1minute_data_cleaned.csv", index_col=0)

# ✅ Safe: force DatetimeIndex
if not isinstance(raw.index, pd.DatetimeIndex):
    raw.index = pd.to_datetime(raw.index)
logger.info(f"Loaded CSV: {len(raw)} rows, {raw.shape[1]} columns")

# ✅ Safe tz attributes - Fixed type checking
if isinstance(raw.index, pd.DatetimeIndex):
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("Africa/Johannesburg")
    else:
        raw.index = raw.index.tz_convert("Africa/Johannesburg")
    raw = raw.tz_convert("UTC").tz_convert("America/New_York")
else:
    raise ValueError("Expected DatetimeIndex for timestamp operations.")

asset = Asset(symbol="BTC-USD", asset_type="crypto")  # Fixed Asset.AssetType reference
quote = Asset(symbol="USD", asset_type="forex")      # Fixed Asset.AssetType reference

raw = raw.resample("1min").last().ffill()

start_time = raw.index[0]
end_time = start_time + pd.Timedelta(days=1)
raw = raw.loc[start_time:end_time]
logger.info(f"Using data window {start_time} to {end_time} ({len(raw)} bars)")

data_obj = Data(asset, raw, timestep="minute", quote=quote)
pandas_data: Dict[str, Data] = {"BTC-USD": data_obj}  # Added type annotation


# ═════════ STRATEGY CLASS ═════════
class ElliottWaveStrategy(Strategy):
    base_asset: Asset
    quote_asset: Asset
    warmup_bars: int
    analysis_window_size: int
    bar_buffer: deque
    all_waves: pd.DataFrame
    last_wave_b_time: Optional[pd.Timestamp]
    last_wave_c_time: Optional[pd.Timestamp]

    def initialize(self, parameters=None):
        # self.set_market("24/7") # ← remove this line
        logger.info("Initializing strategy...")
        self.base_asset = asset
        self.quote_asset = quote
        logger.info(f"Assets initialized — Base: {self.base_asset}, Quote: {self.quote_asset}")
        self.warmup_bars = 50
        self.analysis_window_size = 1440
        logger.info(f"Warmup bars: {self.warmup_bars}, Analysis window size: {self.analysis_window_size}")
        self.bar_buffer = deque(maxlen=self.analysis_window_size)
        self.all_waves = pd.DataFrame(
            columns=pd.Index(["Starting_Point", "Wave_A", "Wave_B", "Wave_C"])
        )
        logger.info("Deque and all_waves DataFrame initialized.")
        self.last_wave_b_time = None
        self.last_wave_c_time = None
        logger.info("Wave time trackers set to None.")
        
        logger.info("Strategy initialization complete ✅")

    def on_trading_iteration(self):
        logger.info(f"🔄 on_trading_iteration called; buffer size={len(self.bar_buffer)}")
        latest = self.get_bars([self.base_asset.symbol], length=1, timestep="minute")
        if latest is None:
            logger.info(f"No bars returned for {self.base_asset.symbol}. Skipping iteration.")
            return

        # Fixed: Check if latest is a dictionary before calling .get()
        if not isinstance(latest, dict):
            logger.info(f"Expected dictionary from get_bars, got {type(latest)}. Skipping iteration.")
            return

        asset_bars = latest.get(self.base_asset)
        if asset_bars is None:
            logger.info(f"No data for {self.base_asset.symbol}. Skipping iteration.")
            return

        latest_bar_df = getattr(asset_bars, "df", None)
        if latest_bar_df is None or latest_bar_df.empty:
            logger.info(f"Empty DataFrame for {self.base_asset.symbol}. Skipping iteration.")
            return

        bar_data = latest_bar_df.iloc[0].to_dict()
        bar_data['timestamp'] = latest_bar_df.index[0]
        self.bar_buffer.append(bar_data)

        if len(self.bar_buffer) < self.warmup_bars:
            logger.info(f"Warming up: {len(self.bar_buffer)}/{self.warmup_bars} bars.")
            return

        self.run_wave_analysis()

    def run_wave_analysis(self):
        logger.info(f"▶️ Entered run_wave_analysis at {self.get_datetime()} — buffer size {len(self.bar_buffer)}")
        # Fixed: Ensure we have data in buffer before converting to DataFrame
        if not self.bar_buffer:
            logger.info("No data in buffer for analysis.")
            return
            
        window_df = pd.DataFrame(list(self.bar_buffer)).set_index('timestamp').sort_index()
        logger.info(f"Analyzing {len(window_df)} bars up to {window_df.index[-1]}")

        close = window_df["close"]
        mean_price = close.rolling(20).mean()
        std_price = close.rolling(20).std()

        if mean_price.dropna().empty:
            logger.info("Rolling mean still warming up.")
            return

        upper_band = mean_price + 1.5 * std_price
        lower_band = mean_price - 1.5 * std_price
        above_mean = close > upper_band
        below_mean = close < lower_band

        p_idx, _ = find_peaks(close, prominence=0.01)
        v_idx, _ = find_peaks(-close, prominence=0.01)
        peak_mask = pd.Series(False, index=window_df.index)
        valley_mask = pd.Series(False, index=window_df.index)
        peak_mask.iloc[p_idx] = True
        valley_mask.iloc[v_idx] = True

        corr_mask = corrective(window_df, peak_mask, valley_mask)
        mot_mask = motive(window_df, peak_mask, valley_mask)

        logger.info(
            f"About to call correction_inverse() — close pts={len(close)}, peaks={peak_mask.sum()}, valleys={valley_mask.sum()}"
        )
        waves_ci = correction_inverse(
            data=cast(pd.Series, close),
            mean=mean_price,
            upper_band=upper_band,
            peak_mask=peak_mask,
            valley_mask=valley_mask,
            above_mean=above_mean,
            below_mean=below_mean,
            corrective_mask=corr_mask,
            motive_mask=mot_mask,
        )
        logger.info(f"correction_inverse returned {len(waves_ci)} wave rows")
        if waves_ci.empty:
            logger.info("No waves detected this iteration.")
            return

        self._process_and_store_waves(waves_ci)

    def _process_and_store_waves(self, waves_df: pd.DataFrame):
        for col in ["Starting_Point", "Wave_A", "Wave_B", "Wave_C"]:
            if col not in waves_df.columns:
                waves_df[col] = pd.NaT

        now = self.get_datetime().replace(tzinfo=None)
        waves_df["Wave_B_naive"] = pd.to_datetime(waves_df["Wave_B"]).dt.tz_localize(None)
        waves_df["Wave_C_naive"] = pd.to_datetime(waves_df["Wave_C"]).dt.tz_localize(None)

        filtered = waves_df[
            (waves_df["Wave_B_naive"].notna() & (waves_df["Wave_B_naive"] <= now)) |
            (waves_df["Wave_C_naive"].notna() & (waves_df["Wave_C_naive"] <= now))
        ].copy()
        filtered.drop(columns=["Wave_B_naive", "Wave_C_naive"], inplace=True)
        if filtered.empty:
            return

        self.all_waves = cast(pd.DataFrame, pd.concat([self.all_waves, filtered], ignore_index=True))
        self.all_waves = cast(
            pd.DataFrame,
            self.all_waves.drop_duplicates(subset=["Starting_Point", "Wave_A", "Wave_B", "Wave_C"])
        )
        self._act_on_new_waves()

    def _act_on_new_waves(self):
        wave_b_times = pd.to_datetime(self.all_waves["Wave_B"].dropna())
        wave_c_times = pd.to_datetime(self.all_waves["Wave_C"].dropna())

        new_bs = cast(pd.Series, wave_b_times if self.last_wave_b_time is None
                      else wave_b_times[wave_b_times > self.last_wave_b_time])
        if new_bs.size > 0:
            t_min = new_bs.min()
            # Create timestamp and explicitly handle NaT case
            timestamp = pd.Timestamp(t_min)
            if pd.isna(timestamp):
                t = None
            else:
                t = timestamp
            
            if t is not None:
                logger.info(f"Signal: New Wave_B at {t}. SELLING.")
                for _ in range(2):
                    self.submit_order(self.create_order(self.base_asset, quantity=0.01, side="sell"))
                self.last_wave_b_time = t

        new_cs = cast(pd.Series, wave_c_times if self.last_wave_c_time is None
                      else wave_c_times[wave_c_times > self.last_wave_c_time])
        if new_cs.size > 0:
            pos = self.get_position(self.base_asset) or 0
            if pos > 0:
                t_min = new_cs.min()
                # Create timestamp and explicitly handle NaT case
                timestamp = pd.Timestamp(t_min)
                if pd.isna(timestamp):
                    t = None
                else:
                    t = timestamp
                
                if t is not None:
                    logger.info(f"Signal: New Wave_C at {t}. Closing position.")
                    self.close_position(self.base_asset)
                    self.last_wave_c_time = t

    def close_position(self, asset: Asset) -> None:
        position = self.get_position(asset) or 0
        if position > 0:
            self.submit_order(self.create_order(asset, quantity=position, side="buy"))


if __name__ == "__main__":
    # Fixed: Added proper type handling for backtest result
    backtest_result = ElliottWaveStrategy.run_backtest(
        PandasDataBacktesting,
        data_obj.datetime_start,
        data_obj.datetime_end,
        pandas_data=pandas_data,
    )
    
    # Handle the result properly - it might be a tuple or a single object
    if isinstance(backtest_result, tuple):
        result, _ = backtest_result
    else:
        result = backtest_result
        
    logger.info("Backtest complete.")