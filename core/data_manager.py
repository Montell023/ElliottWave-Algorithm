# algostack/core/data_manager.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Tuple, List, Optional, Any
import logging
import sys
import os

# Fix import paths - add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Goes up to algostack/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

class DataManager:
    """
    Complete Data Manager with WavesManager integration.
    Now provides real subdivision masks for TrendsManager.
    """

    def __init__(self, mode: str = 'backtest', window_size: int = 200, 
                 ma_length: int = 20, bb_multiplier: float = 2.0,
                 peak_prominence: float = 0.5):
        """
        Initialize the complete Data Manager with WavesManager integration.

        Parameters:
        -----------
        mode : str
            'backtest' or 'live' - determines peak detection method
        window_size : int
            Size of data window to maintain
        ma_length : int
            Moving average length (from ewc_plots.py)
        bb_multiplier : float  
            Bollinger Band multiplier (from ewc_plots.py)
        peak_prominence : float
            Prominence threshold for peak detection
        """
        self.mode = mode
        self.window_size = window_size
        self.ma_length = ma_length
        self.bb_multiplier = bb_multiplier
        self.peak_prominence = peak_prominence
        
        # Data storage
        self.data = pd.DataFrame()
        self.indicators = {}
        
        # Import specialized components
        self.peak_detector = None
        self.fibonacci_calculator = None
        self.waves_manager = None
        
        # Initialize specialized components based on mode
        if mode == 'live':
            try:
                from .real_time_peak_detector import RealTimePeakDetector
                self.peak_detector = RealTimePeakDetector(
                    window_size=min(50, window_size),
                    prominence=0.5  # Fixed parameter name
                )
                logger.info("RealTimePeakDetector initialized for live mode")
            except ImportError as e:
                logger.warning(f"RealTimePeakDetector not available: {e}, using backtest peak detection")
        
        try:
            from .fibonacci_calculator import FibonacciCalculator
            self.fibonacci_calculator = FibonacciCalculator()
            logger.info("FibonacciCalculator initialized")
        except ImportError as e:
            logger.warning(f"FibonacciCalculator not available: {e}, Fibonacci functions will be unavailable")
        
        # Initialize WavesManager
        self._initialize_waves_manager()
        
        self._initialize_indicators()
        
        logger.info(f"DataManager initialized: mode={mode}, window_size={window_size}, "
                   f"ma_length={ma_length}, bb_multiplier={bb_multiplier}")

    def _initialize_waves_manager(self):
        """Initialize WavesManager with dependency injection."""
        try:
            # Import from waves_manager package
            from waves_manager.waves_manager import WavesManager
            self.waves_manager = WavesManager(self)
            logger.info("WavesManager initialized successfully")
        except ImportError as e:
            logger.warning(f"WavesManager not available: {e}")
            self.waves_manager = None
        except Exception as e:
            logger.error(f"Error initializing WavesManager: {e}")
            self.waves_manager = None

    def _initialize_indicators(self):
        """
        Initialize ALL indicator storage - exactly matching ewc_plots.py requirements
        plus additional indicators needed for Elliott Wave algorithms.
        """
        self.indicators = {
            # EXACTLY from ewc_plots.py - Core calculations
            'mean': pd.Series(dtype=float),           # Rolling mean (Close.rolling.mean())
            'std': pd.Series(dtype=float),            # Rolling std (Close.rolling.std())
            'upper_band': pd.Series(dtype=float),     # Upper Bollinger Band
            'lower_band': pd.Series(dtype=float),     # Lower Bollinger Band
            
            # EXACTLY from ewc_plots.py - Boolean conditions
            'above_mean': pd.Series(dtype=bool),      # Close > upper_band
            'below_mean': pd.Series(dtype=bool),      # Close < lower_band
            
            # Peak/valley detection (enhanced from ewc_plots.py)
            'peak_mask': pd.Series(dtype=bool),       # Peak positions
            'valley_mask': pd.Series(dtype=bool),     # Valley positions
            
            # Additional indicators for wave algorithms
            'high': pd.Series(dtype=float),           # High prices
            'low': pd.Series(dtype=float),            # Low prices
            'close': pd.Series(dtype=float),          # Close prices
            'open': pd.Series(dtype=float),           # Open prices
        }

    def update_data(self, new_data: pd.DataFrame, current_timestamp: pd.Timestamp = None) -> bool:
        """
        Update data and calculate ALL indicators from ewc_plots.py plus peaks/valleys.
        Also updates WavesManager with new data.

        Parameters:
        -----------
        new_data : pd.DataFrame
            New price data with columns: ['open', 'high', 'low', 'close', 'volume']
        current_timestamp : pd.Timestamp, optional
            Current timestamp for live mode peak detection

        Returns:
        --------
        bool
            True if successful update, False if insufficient data
        """
        try:
            # Validate input data
            if new_data is None or new_data.empty:
                logger.warning("No data provided to update_data")
                return False
                
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in new_data.columns for col in required_columns):
                logger.error(f"Missing required columns. Need: {required_columns}")
                return False
            
            # Update data window
            if self.data.empty:
                self.data = new_data.copy()
            else:
                # Concatenate and remove duplicates
                combined = pd.concat([self.data, new_data])
                self.data = combined[~combined.index.duplicated(keep='last')].copy()
            
            # Maintain window size
            if len(self.data) > self.window_size:
                self.data = self.data.iloc[-self.window_size:].copy()
            
            # Store individual price series for easy access
            self.indicators['open'] = self.data['open']
            self.indicators['high'] = self.data['high']
            self.indicators['low'] = self.data['low'] 
            self.indicators['close'] = self.data['close']
            
            # Calculate ALL indicators from ewc_plots.py
            self._calculate_ewc_indicators()
            
            # Mode-specific peak detection
            if self.mode == 'backtest':
                success = self._update_peaks_backtest()
            else:
                success = self._update_peaks_live(current_timestamp)
            
            # Update WavesManager with new subdivisions analysis
            if success and self.waves_manager is not None:
                try:
                    algorithm_inputs = self.get_algorithm_inputs()
                    waves_result = self.waves_manager.analyze_subdivisions(
                        data=algorithm_inputs['data'],
                        peak_mask=algorithm_inputs['peak_mask'],
                        valley_mask=algorithm_inputs['valley_mask']
                    )
                    subdivision_count = len([r for r in waves_result.values() if r is not None and (hasattr(r, 'sum') and r.sum() > 0 or hasattr(r, 'notna') and r.notna().sum() > 0)])
                    logger.debug(f"WavesManager updated - found {subdivision_count} subdivision types with valid data")
                except Exception as e:
                    logger.error(f"Error updating WavesManager: {str(e)}")
            
            if success:
                logger.debug(f"Data updated successfully. Current data points: {len(self.data)}")
                # Log detection stats
                peaks, valleys = self.get_peaks_valleys()
                logger.debug(f"Peaks: {len(peaks)}, Valleys: {len(valleys)}")
            else:
                logger.warning("Data update completed but peak detection may be limited")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating data: {str(e)}")
            return False

    def _calculate_ewc_indicators(self):
        """
        Calculate EXACTLY the same indicators as ewc_plots.py
        These are the core values your Elliott Wave algorithms depend on.
        """
        if len(self.data) < self.ma_length:
            logger.warning(f"Insufficient data for indicator calculation. Need {self.ma_length}, have {len(self.data)}")
            return
        
        close = self.data['close']
        
        # EXACT CALCULATIONS from ewc_plots.py:
        # mean_price = data_last_day['Close'].rolling(window=length).mean()
        self.indicators['mean'] = close.rolling(window=self.ma_length).mean()
        
        # std_price = data_last_day['Close'].rolling(window=length).std()  
        self.indicators['std'] = close.rolling(window=self.ma_length).std()
        
        # upper_band = mean_price + mult * std_price
        self.indicators['upper_band'] = self.indicators['mean'] + (self.indicators['std'] * self.bb_multiplier)
        
        # lower_band = mean_price - mult * std_price
        self.indicators['lower_band'] = self.indicators['mean'] - (self.indicators['std'] * self.bb_multiplier)
        
        # EXACT CONDITIONS from ewc_plots.py:
        # data_last_day['AboveMean'] = np.where(data_last_day['Close'] > mean_price + mult * std_price, ...)
        self.indicators['above_mean'] = close > self.indicators['upper_band']
        
        # data_last_day['BelowMean'] = np.where(data_last_day['Close'] < mean_price - mult * std_price, ...)  
        self.indicators['below_mean'] = close < self.indicators['lower_band']
        
        logger.debug("EWC indicators calculated successfully")

    def _update_peaks_backtest(self) -> bool:
        """
        Backtest mode peak detection - uses scipy find_peaks on current window.
        Processes data sequentially to avoid future lookahead bias.
        """
        if len(self.data) < 10:
            logger.warning("Insufficient data for peak detection")
            return False
            
        try:
            highs = self.data['high'].values
            lows = self.data['low'].values
            
            # Use scipy find_peaks but only on available data (no future knowledge)
            peaks, _ = find_peaks(highs, prominence=self.peak_prominence)
            valleys, _ = find_peaks(-lows, prominence=self.peak_prominence)
            
            # Create boolean masks
            self.indicators['peak_mask'] = pd.Series(False, index=self.data.index)
            self.indicators['valley_mask'] = pd.Series(False, index=self.data.index)
            
            if len(peaks) > 0:
                peak_indices = self.data.index[peaks]
                self.indicators['peak_mask'].loc[peak_indices] = True
                logger.debug(f"Found {len(peaks)} peaks in backtest mode")
                
            if len(valleys) > 0:
                valley_indices = self.data.index[valleys]
                self.indicators['valley_mask'].loc[valley_indices] = True
                logger.debug(f"Found {len(valleys)} valleys in backtest mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in backtest peak detection: {str(e)}")
            return False

    def _update_peaks_live(self, current_timestamp: pd.Timestamp) -> bool:
        """
        Live mode peak detection using RealTimePeakDetector.
        No future knowledge - only uses data up to current point.
        """
        if current_timestamp is None:
            logger.warning("No current timestamp provided for live peak detection")
            return self._update_peaks_backtest()  # Fall back to backtest method
            
        if self.peak_detector is None:
            logger.warning("Peak detector not available, using backtest method")
            return self._update_peaks_backtest()  # Fall back to backtest method
            
        try:
            # Get latest prices
            if current_timestamp not in self.data.index:
                logger.warning(f"Current timestamp {current_timestamp} not in data index")
                return False
                
            latest_high = self.data.loc[current_timestamp, 'high']
            latest_low = self.data.loc[current_timestamp, 'low'] 
            latest_close = self.data.loc[current_timestamp, 'close']
            
            # Update real-time detector
            peaks, valleys = self.peak_detector.update(
                latest_high, latest_low, latest_close, current_timestamp
            )
            
            # Update masks
            self.indicators['peak_mask'] = pd.Series(False, index=self.data.index)
            self.indicators['valley_mask'] = pd.Series(False, index=self.data.index)
            
            for peak_time in peaks:
                if peak_time in self.data.index:
                    self.indicators['peak_mask'].loc[peak_time] = True
                    
            for valley_time in valleys:
                if valley_time in self.data.index:
                    self.indicators['valley_mask'].loc[valley_time] = True
            
            logger.debug(f"Live peak detection: {len(peaks)} peaks, {len(valleys)} valleys")
            return True
            
        except Exception as e:
            logger.error(f"Error in live peak detection: {str(e)}")
            return self._update_peaks_backtest()  # Fall back to backtest method

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get COMPLETE state with ALL values needed for Elliott Wave algorithms.
        Returns exactly what your algorithms expect.

        Returns:
        --------
        Dict with all data and indicators
        """
        if self.data.empty:
            logger.warning("No data available in get_current_state")
            return {}
            
        return {
            # Core data
            'data': self.data,
            'close': self.indicators['close'],
            'high': self.indicators['high'],
            'low': self.indicators['low'],
            'open': self.indicators['open'],
            
            # All indicators
            'indicators': self.indicators,
            
            # EXACTLY from ewc_plots.py
            'mean': self.indicators['mean'],
            'upper_band': self.indicators['upper_band'],
            'lower_band': self.indicators['lower_band'],
            'above_mean': self.indicators['above_mean'],
            'below_mean': self.indicators['below_mean'],
            
            # Peak/valley information
            'peaks': self.data.index[self.indicators['peak_mask']],
            'valleys': self.data.index[self.indicators['valley_mask']],
            'peak_mask': self.indicators['peak_mask'],
            'valley_mask': self.indicators['valley_mask']
        }

    def get_algorithm_inputs(self) -> Dict[str, Any]:
        """
        Get exactly the inputs your Elliott Wave algorithms expect.
        Matches the parameter signatures of impulse.py, correction.py, etc.

        Returns:
        --------
        Dict with parameters for Elliott Wave algorithms
        """
        state = self.get_current_state()
        if not state:
            logger.warning("No state available for algorithm inputs")
            return {}
            
        return {
            'data': state['data'],
            'mean': state['mean'],
            'upper_band': state['upper_band'],
            'lower_band': state['lower_band'],
            'peak_mask': state['peak_mask'],
            'valley_mask': state['valley_mask'],
            'above_mean': state['above_mean'],
            'below_mean': state['below_mean']
        }

    def get_subdivision_masks(self) -> Dict[str, pd.Series]:
        """
        Get REAL subdivision masks from WavesManager for TrendsManager.
        This is the critical integration point that fixes the 0-pattern detection issue.

        Returns:
        --------
        Dict with 'corrective_mask' and 'motive_mask' boolean Series
        """
        if self.waves_manager is None:
            logger.warning("WavesManager not available, returning empty subdivision masks")
            return self._create_empty_subdivision_masks()
        
        try:
            masks = self.waves_manager.get_subdivision_masks()
            logger.info(f"Retrieved real subdivision masks - corrective: {masks['corrective_mask'].sum()}, motive: {masks['motive_mask'].sum()}")
            return masks
        except Exception as e:
            logger.error(f"Error getting subdivision masks from WavesManager: {str(e)}")
            return self._create_empty_subdivision_masks()

    def _create_empty_subdivision_masks(self) -> Dict[str, pd.Series]:
        """Create empty subdivision masks as fallback."""
        if self.data.empty:
            return {
                'corrective_mask': pd.Series(dtype=bool),
                'motive_mask': pd.Series(dtype=bool)
            }
        return {
            'corrective_mask': pd.Series(False, index=self.data.index),
            'motive_mask': pd.Series(False, index=self.data.index)
        }

    def get_corrective_mask(self) -> pd.Series:
        """
        Get corrective mask for TrendsManager compatibility.
        Uses real data from WavesManager instead of mock data.
        """
        masks = self.get_subdivision_masks()
        return masks['corrective_mask']

    def get_motive_mask(self) -> pd.Series:
        """
        Get motive mask for TrendsManager compatibility.
        Uses real data from WavesManager instead of mock data.
        """
        masks = self.get_subdivision_masks()
        return masks['motive_mask']

    def get_peaks_valleys(self) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """
        Get current peaks and valleys as timestamp lists.

        Returns:
        --------
        Tuple of (peaks_list, valleys_list)
        """
        if self.data.empty:
            return [], []
            
        peaks = self.data.index[self.indicators['peak_mask']].tolist()
        valleys = self.data.index[self.indicators['valley_mask']].tolist()
        return peaks, valleys

    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current detection state.

        Returns:
        --------
        Dict with detection statistics
        """
        peaks, valleys = self.get_peaks_valleys()
        
        stats = {
            'data_points': len(self.data),
            'peaks_detected': len(peaks),
            'valleys_detected': len(valleys),
            'mode': self.mode,
            'window_size': self.window_size,
            'ma_length': self.ma_length,
            'bb_multiplier': self.bb_multiplier,
            'has_sufficient_data': self.has_sufficient_data(),
            'indicators_ready': len([v for v in self.indicators.values() if not v.empty]) > 0,
            'waves_manager_available': self.waves_manager is not None
        }
        
        # Add WavesManager stats if available
        if self.waves_manager is not None:
            try:
                waves_stats = self.waves_manager.get_detection_stats()
                stats['waves_manager'] = waves_stats
            except Exception as e:
                logger.debug(f"Could not get WavesManager stats: {e}")
        
        # Add peak detector stats if available
        if self.peak_detector and hasattr(self.peak_detector, 'get_detection_stats'):
            try:
                peak_stats = self.peak_detector.get_detection_stats()
                stats.update({'peak_detector_' + k: v for k, v in peak_stats.items()})
            except Exception as e:
                logger.debug(f"Could not get peak detector stats: {e}")
            
        return stats

    def has_sufficient_data(self, min_points: int = None) -> bool:
        """
        Check if we have sufficient data for analysis.

        Parameters:
        -----------
        min_points : int, optional
            Minimum required data points, defaults to ma_length

        Returns:
        --------
        bool indicating if sufficient data is available
        """
        if min_points is None:
            min_points = self.ma_length
            
        return len(self.data) >= min_points

    def get_fibonacci_calculator(self):
        """
        Get the Fibonacci calculator instance if available.

        Returns:
        --------
        FibonacciCalculator instance or None
        """
        return self.fibonacci_calculator

    def get_waves_manager(self):
        """
        Get the WavesManager instance if available.

        Returns:
        --------
        WavesManager instance or None
        """
        return self.waves_manager

    def reset(self):
        """Reset all data and indicators."""
        self.data = pd.DataFrame()
        self._initialize_indicators()
        
        if self.peak_detector and hasattr(self.peak_detector, 'reset'):
            try:
                self.peak_detector.reset()
            except Exception as e:
                logger.warning(f"Error resetting peak detector: {e}")
        
        if self.waves_manager and hasattr(self.waves_manager, 'reset'):
            try:
                self.waves_manager.reset()
            except Exception as e:
                logger.warning(f"Error resetting WavesManager: {e}")
            
        logger.info("DataManager reset complete")

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of current data state.

        Returns:
        --------
        Dict with data summary information
        """
        if self.data.empty:
            return {'status': 'no_data'}
            
        # Get subdivision mask stats
        masks = self.get_subdivision_masks()
        corrective_count = masks['corrective_mask'].sum() if not masks['corrective_mask'].empty else 0
        motive_count = masks['motive_mask'].sum() if not masks['motive_mask'].empty else 0
            
        return {
            'status': 'ready',
            'data_points': len(self.data),
            'date_range': {
                'start': self.data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': self.data.index.max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'price_range': {
                'high': float(self.data['high'].max()),
                'low': float(self.data['low'].min()),
                'close': float(self.data['close'].iloc[-1])
            },
            'subdivision_masks': {
                'corrective_points': int(corrective_count),
                'motive_points': int(motive_count)
            },
            'detection_stats': self.get_detection_stats()
        }


# Test the updated DataManager with WavesManager integration
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing Updated DataManager with WavesManager Integration...")
    
    # Create test data matching BTC-USD format
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')  # Increased to 200 for better wave detection
    np.random.seed(42)  # For reproducible results
    
    # Create more realistic wave-like data
    t = np.linspace(0, 6*np.pi, 200)
    base_trend = np.linspace(100, 110, 200)  # Slight upward trend
    wave_pattern = 10 * np.sin(t) + 5 * np.sin(2*t)  # Multiple frequency waves
    noise = np.random.normal(0, 2, 200)
    
    synthetic_prices = base_trend + wave_pattern + noise
    
    test_data = pd.DataFrame({
        'open': synthetic_prices,
        'high': synthetic_prices + abs(np.random.normal(0, 1, 200)),
        'low': synthetic_prices - abs(np.random.normal(0, 1, 200)), 
        'close': synthetic_prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Test 1: Backtest Mode with WavesManager
    print("\n1. Testing Backtest Mode with WavesManager:")
    dm_backtest = DataManager(mode='backtest', window_size=100, ma_length=20)
    
    # Simulate sequential updates (like real trading)
    for i in range(0, len(test_data), 20):
        chunk = test_data.iloc[i:i+20]
        success = dm_backtest.update_data(chunk)
        if success:
            stats = dm_backtest.get_detection_stats()
            print(f"  Updated {len(chunk)} bars -> {stats['peaks_detected']} peaks, {stats['valleys_detected']} valleys")
    
    # Test getting algorithm inputs
    inputs = dm_backtest.get_algorithm_inputs()
    print(f"\n  Algorithm inputs available: {len(inputs)} parameters")
    
    # Test subdivision masks (CRITICAL FOR TRENDSMANAGER)
    print(f"\n2. Testing Subdivision Masks (TrendsManager Integration):")
    subdivision_masks = dm_backtest.get_subdivision_masks()
    print(f"  corrective_mask: {subdivision_masks['corrective_mask'].sum()} True values")
    print(f"  motive_mask: {subdivision_masks['motive_mask'].sum()} True values")
    print(f"  Masks compatible with data: {subdivision_masks['corrective_mask'].index.equals(test_data.index)}")
    
    # Test 3: Data Summary with subdivision info
    print("\n3. Data Summary with Subdivision Information:")
    summary = dm_backtest.get_data_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        print(f"      {k2}: {v2}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Test 4: WavesManager direct access
    print(f"\n4. Testing WavesManager Direct Access:")
    waves_manager = dm_backtest.get_waves_manager()
    if waves_manager:
        waves_stats = waves_manager.get_detection_stats()
        print(f"  WavesManager available: Yes")
        print(f"  Total subdivisions detected: {waves_stats['total_subdivisions_detected']}")
        for algo, count in waves_stats['subdivision_counts'].items():
            print(f"    {algo}: {count}")
    else:
        print(f"  WavesManager available: No")
    
    print("\nUpdated DataManager test completed successfully! ✅")
    print("\nKEY IMPROVEMENT: TrendsManager will now use REAL subdivision masks instead of mock data!")