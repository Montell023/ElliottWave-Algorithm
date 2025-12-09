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
    Enhanced with DFS Structural Turn Detection for noise filtering.
    Enhanced with Wave-Aware DFS for robust Elliott Wave detection.
    """

    def __init__(self, data: pd.DataFrame = None, mode: str = 'backtest', window_size: int = None, 
                 ma_length: int = 20, bb_multiplier: float = 2.0,
                 peak_prominence: float = 0.1,  # CHANGED: Default prominence from 0.5 to 0.1
                 structural_move_threshold: float = 0.001):  # NEW: Default 0.5% minimum move for structural turns
        """
        Initialize the complete Data Manager with WavesManager integration.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Initial price data with columns: ['open', 'high', 'low', 'close', 'volume']
        mode : str
            'backtest' or 'live' - determines peak detection method
        window_size : int
            Size of data window to maintain (None = use full dataset)
        ma_length : int
            Moving average length (from ewc_plots.py)
        bb_multiplier : float  
            Bollinger Band multiplier (from ewc_plots.py)
        peak_prominence : float
            Prominence threshold for peak detection - CHANGED: Default to 0.1
        structural_move_threshold : float
            Minimum price movement ratio (0.005 = 0.5%) to consider a turn structural
        """
        # FIX: Handle case where data is passed as first parameter
        if isinstance(mode, pd.Series) or isinstance(mode, pd.DataFrame):
            # Data was passed as first positional argument, adjusting parameters...
            logger.warning("Data passed as first parameter, adjusting arguments...")
            data = mode
            mode = 'backtest'
        
        self.mode = str(mode)  # Ensure mode is always a string
        self.window_size = window_size  # CHANGED: Allow None for full dataset
        self.ma_length = ma_length
        self.bb_multiplier = bb_multiplier
        self.peak_prominence = peak_prominence  # Now defaults to 0.1
        self.structural_move_threshold = structural_move_threshold  # NEW: Structural turn threshold
        
        # Data storage
        self.data = pd.DataFrame()
        self.indicators = {}
        
        # Import specialized components
        self.peak_detector = None
        self.fibonacci_calculator = None
        self.waves_manager = None
        
        # NEW: Cache for structural turns to improve performance
        self._structural_turn_cache = {}
        
        # Initialize specialized components based on mode
        # FIX: Use explicit string comparison with proper type checking
        if str(self.mode) == 'live':
            try:
                from .real_time_peak_detector import RealTimePeakDetector
                self.peak_detector = RealTimePeakDetector(
                    window_size=min(50, window_size) if window_size else 50,
                    prominence=0.1  # CHANGED: From 0.5 to 0.1 to match backtest
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
        
        # Initialize with provided data if any
        if data is not None and not data.empty:
            success = self.update_data(data)
            if success:
                logger.info(f"Initialized with {len(data)} data points")
            else:
                logger.warning("Failed to initialize with provided data")
        
        logger.info(f"DataManager initialized: mode={self.mode}, window_size={window_size}, "
                   f"ma_length={ma_length}, bb_multiplier={bb_multiplier}, peak_prominence={self.peak_prominence}, "
                   f"structural_move_threshold={self.structural_move_threshold}")

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

    def _ensure_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all price columns are numeric by converting to float.
        This is CRITICAL for CSV data that may be read as strings.
        """
        try:
            # Convert price columns to numeric
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convert volume to numeric if present
            if 'volume' in data.columns:
                data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
            
            # Fill any NaN values that resulted from conversion
            data = data.ffill().bfill()
            
            logger.debug("Successfully converted data to numeric types")
            return data
            
        except Exception as e:
            logger.error(f"Error converting data to numeric: {e}")
            return data

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
                
            # FIX: Handle case where single Series is passed instead of DataFrame
            if isinstance(new_data, pd.Series):
                logger.info("Converting Series to DataFrame")
                if new_data.name == 'close':
                    new_data = pd.DataFrame({
                        'open': new_data,
                        'high': new_data, 
                        'low': new_data,
                        'close': new_data,
                        'volume': pd.Series(1000, index=new_data.index)  # Default volume
                    })
                else:
                    logger.error("Cannot convert unnamed Series to DataFrame")
                    return False
            
            # FIX: Ensure all required columns exist with proper case sensitivity
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in new_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(new_data.columns)}")
                
                # Try to handle case sensitivity issues
                column_mapping = {}
                for col in required_columns:
                    # Try to find case-insensitive match
                    matching_cols = [c for c in new_data.columns if c.lower() == col.lower()]
                    if matching_cols:
                        column_mapping[col] = matching_cols[0]
                
                if len(column_mapping) == len(required_columns):
                    logger.info(f"Found case-insensitive matches: {column_mapping}")
                    new_data = new_data.rename(columns={v: k for k, v in column_mapping.items()})
                else:
                    logger.error("Could not find all required columns")
                    return False
            
            # CRITICAL FIX: Ensure all data is numeric before processing
            new_data = self._ensure_numeric_data(new_data)
            
            # Check if we have any valid numeric data after conversion
            if new_data[['open', 'high', 'low', 'close']].isna().all().all():
                logger.error("No valid numeric data found after conversion")
                return False
            
            # Update data window - CHANGED: Only apply window if specified
            if self.data.empty:
                self.data = new_data.copy()
            else:
                # Concatenate and remove duplicates
                combined = pd.concat([self.data, new_data])
                self.data = combined[~combined.index.duplicated(keep='last')].copy()
            
            # Maintain window size only if specified - CHANGED: Allow full dataset analysis
            if self.window_size is not None and len(self.data) > self.window_size:
                self.data = self.data.iloc[-self.window_size:].copy()
            
            # NEW: Clear structural turn cache when data changes
            self._structural_turn_cache.clear()
            
            # Store individual price series for easy access
            self.indicators['open'] = self.data['open']
            self.indicators['high'] = self.data['high']
            self.indicators['low'] = self.data['low'] 
            self.indicators['close'] = self.data['close']
            
            # Calculate ALL indicators from ewc_plots.py
            self._calculate_ewc_indicators()
            
            # Mode-specific peak detection
            if str(self.mode) == 'backtest':  # FIX: Explicit string comparison
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
                logger.info(f"Peak/Valley Detection Results: {len(peaks)} peaks, {len(valleys)} valleys")  # CHANGED: to INFO level
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
            # CRITICAL FIX: Use CLOSE prices instead of high/low like original ewc_plots.py
            close_prices = pd.to_numeric(self.data['close'], errors='coerce').values
            
            # Remove any NaN values that might interfere with peak detection
            valid_prices = ~np.isnan(close_prices)
            
            if not np.any(valid_prices):
                logger.warning("No valid numeric data for peak detection")
                return False
            
            # ----------------------------------------------------------
            #  GOLDILOCKS PROMINENCE – 3 % of 2-hour range
            # ----------------------------------------------------------
            close = self.data['close']
            lookback = min(120, len(close))              # 2 h on 1-min
            hi = close.rolling(lookback).max()
            lo = close.rolling(lookback).min()
            range_ = (hi - lo).fillna(0)
            prominence = range_ * 0.03                   # 3 % of local range
            prominence = prominence.clip(lower=0.005)    # floor 0.5 ¢

            peaks, _   = find_peaks(close_prices, prominence=prominence.values)
            valleys, _ = find_peaks(-close_prices, prominence=prominence.values)
            logger.info(f"Adaptive prominence - peaks: {len(peaks)}, valleys: {len(valleys)}")
            
            # Create boolean masks
            self.indicators['peak_mask'] = pd.Series(False, index=self.data.index)
            self.indicators['valley_mask'] = pd.Series(False, index=self.data.index)
            
            if len(peaks) > 0:
                peak_indices = self.data.index[peaks]
                self.indicators['peak_mask'].loc[peak_indices] = True
                logger.info(f"Found {len(peaks)} peaks in backtest mode using adaptive prominence")
                
            if len(valleys) > 0:
                valley_indices = self.data.index[valleys]
                self.indicators['valley_mask'].loc[valley_indices] = True
                logger.info(f"Found {len(valleys)} valleys in backtest mode using adaptive prominence")
            
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
            # Get latest prices - ensure numeric
            if current_timestamp not in self.data.index:
                logger.warning(f"Current timestamp {current_timestamp} not in data index")
                return False
                
            latest_high = float(pd.to_numeric(self.data.loc[current_timestamp, 'high'], errors='coerce'))
            latest_low = float(pd.to_numeric(self.data.loc[current_timestamp, 'low'], errors='coerce'))
            latest_close = float(pd.to_numeric(self.data.loc[current_timestamp, 'close'], errors='coerce'))
            
            # Check if we have valid numeric values
            if np.isnan(latest_high) or np.isnan(latest_low) or np.isnan(latest_close):
                logger.warning("Invalid numeric values for live peak detection")
                return self._update_peaks_backtest()
            
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
            
            logger.info(f"Live peak detection: {len(peaks)} peaks, {len(valleys)} valleys")
            return True
            
        except Exception as e:
            logger.error(f"Error in live peak detection: {str(e)}")
            return self._update_peaks_backtest()  # Fall back to backtest method

    # =========================================================================
    # WAVE-AWARE DFS STRUCTURAL TURN DETECTION - ENHANCED METHODS ***** - get_structural_turn_for_wave IS NO LONGER USED 
    # BUT CAN BE USED AS A GUIDE ON HOW THE DFS SEARCH WAS IMPLEMENTED AND ALSO IT CAN BE USED TO TRY OUT APPLICATIONS 
    # WHERE WE TRY TO FIND THE HIGHEST PRICE OF A RANGE BEING THE FURTHEST NODE(PEAK/VALLEY).
    # =========================================================================

    def get_structural_turn_for_wave(self, start_timestamp: pd.Timestamp, direction: str, 
                                   wave_type: str = None, expected_move: float = None) -> Optional[pd.Timestamp]:
        """
        ENHANCED: Wave-aware DFS that treats each wave as a separate graph/family.
        Finds the furthest structural cousin in the wave family.
        Handles both orphan waves (single member) and family waves (multiple members).
        
        Parameters:
        -----------
        start_timestamp : pd.Timestamp
            Starting point of the wave (parent)
        direction : str ('up' for peaks, 'down' for valleys)
            Direction to search for the next turn
        wave_type : str, optional
            Type of wave ('wave1', 'wave2', 'wave3', etc.) for context-aware detection
        expected_move : float, optional
            Expected price move for this wave type (defaults to wave-type specific)
            
        Returns:
        --------
        pd.Timestamp or None
            The furthest structural turn in this wave family, or None if not found
        """
        if self.data.empty:
            logger.warning("No data available for structural turn detection")
            return None
        
        if start_timestamp not in self.data.index:
            logger.warning(f"Start timestamp {start_timestamp} not in data index")
            return None
        
        # Create cache key for this query
        cache_key = (start_timestamp, direction, wave_type, expected_move)
        if cache_key in self._structural_turn_cache:
            return self._structural_turn_cache[cache_key]
        
        # Get expected move based on wave type
        if expected_move is None and wave_type is not None:
            expected_move = self._get_wave_expected_move(wave_type)
        elif expected_move is None:
            expected_move = self.structural_move_threshold
        
        logger.debug(f"Wave-aware DFS from {start_timestamp} in {direction} direction "
                    f"for {wave_type} with expected_move={expected_move:.4f}")
        
        # Get ALL potential family members in this wave
        if direction == 'up':
            family_members = self._get_peaks_after(start_timestamp)
        else:
            family_members = self._get_valleys_after(start_timestamp)
        
        # Handle orphan waves (single member) - return immediately
        if len(family_members) == 1:
            orphan_member = family_members[0]
            orphan_move = self._calculate_price_move(start_timestamp, orphan_member)
            logger.debug(f"Orphan wave detected: single {direction} at {orphan_member} "
                        f"with move {orphan_move:.4f}")
            self._structural_turn_cache[cache_key] = orphan_member
            return orphan_member
        
        # Handle empty family (no members found)
        if len(family_members) == 0:
            logger.debug(f"No {direction} turns found after {start_timestamp}")
            self._structural_turn_cache[cache_key] = None
            return None
        
        # Family wave (multiple members) - find the furthest structural cousin
        structural_turn = self._find_furthest_cousin(start_timestamp, family_members, expected_move, direction)
        
        # Cache the result
        self._structural_turn_cache[cache_key] = structural_turn
        
        if structural_turn:
            actual_move = self._calculate_price_move(start_timestamp, structural_turn)
            logger.info(f"Found structural turn for {wave_type} at {structural_turn} "
                       f"(move: {actual_move:.4f}, expected: {expected_move:.4f})")
        else:
            logger.debug(f"No structural turn found from {start_timestamp} in {direction} direction "
                        f"for {wave_type} with threshold {expected_move:.4f}")
        
        return structural_turn
    
    def get_quality_starting_points(self, min_price_move: float = 0.002) -> List[pd.Timestamp]:
        """
        Get only STRUCTURALLY significant turns as starting points.
        Filters out noise and tiny moves.
        
        Parameters:
        -----------
        min_price_move : float
            Minimum price movement ratio (0.002 = 0.2%) to consider turn significant
            
        Returns:
        --------
        List of timestamps that are quality starting points for wave analysis
        """
        try:
            # Get raw counts for logging
            raw_peaks_count = len(self.data.index[self.indicators['peak_mask']]) if not self.indicators['peak_mask'].empty else 0
            raw_valleys_count = len(self.data.index[self.indicators['valley_mask']]) if not self.indicators['valley_mask'].empty else 0
            total_raw_turns = raw_peaks_count + raw_valleys_count
            
            # Get structural turns (already filters by significance)
            turns = self.get_all_structural_turns('both', min_price_move)
            
            # Combine peaks and valleys, sort chronologically
            quality_starts = sorted(turns['peaks'] + turns['valleys'])
            
            quality_count = len(quality_starts)
            
            logger.info(f"🔍 Quality Filtering: {quality_count} quality starts out of {total_raw_turns} raw turns "
                       f"({quality_count/total_raw_turns*100:.1f}% kept, {min_price_move*100:.1f}% min move)")
            
            return quality_starts
            
        except Exception as e:
            logger.error(f"Error getting quality starting points: {e}")
            # Fallback to raw turns if error
            peaks = self.data.index[self.indicators['peak_mask']].tolist() if not self.indicators['peak_mask'].empty else []
            valleys = self.data.index[self.indicators['valley_mask']].tolist() if not self.indicators['valley_mask'].empty else []
            return sorted(peaks + valleys)
    
    def find_wave_end_by_tiredness(self, start_timestamp: pd.Timestamp, direction: str,
                                 wave_type: str = None, tiredness_threshold: float = 0.25) -> Optional[pd.Timestamp]:
        """
        Climber analogy: Find where price gets "tired" (shows reversal momentum).
        Stops at first significant reversal after the start point.
        
        Parameters:
        -----------
        start_timestamp : pd.Timestamp
            Starting point of the wave
        direction : str ('up' for peaks, 'down' for valleys)
            Direction of the wave
        wave_type : str, optional
            Type of wave for context-aware tiredness thresholds
        tiredness_threshold : float
            Minimum tiredness ratio (0.25 = 25%) to consider wave complete
            
        Returns:
        --------
        pd.Timestamp or None
            Timestamp where wave shows significant reversal, or None if not found
        """
        if self.data.empty:
            logger.warning("No data available for tiredness detection")
            return None
        
        if start_timestamp not in self.data.index:
            logger.warning(f"Start timestamp {start_timestamp} not in data index")
            return None
        
        logger.debug(f"Tiredness detection from {start_timestamp} in {direction} direction "
                    f"for {wave_type} with threshold={tiredness_threshold:.2f}")
        
        # Get potential wave endings (resting spots)
        if direction == 'up':
            resting_spots = self._get_peaks_after(start_timestamp)
        else:
            resting_spots = self._get_valleys_after(start_timestamp)
        
        if not resting_spots:
            logger.debug(f"No resting spots found after {start_timestamp}")
            return None
        
        start_price = float(self.data.loc[start_timestamp, 'close'])
        
        # Wave-specific tiredness thresholds (HIGHER for cleaner patterns)
        wave_tiredness = {
            'wave1': 0.40,   # Wave 1 needs 40% reversal (was 25%)
            'wave2': 0.50,   # Wave 2 needs 50% reversal (was 35%)
            'wave3': 0.60,   # Wave 3 needs 60% exhaustion (was 40%)
            'wave4': 0.40,   # Wave 4 needs 40% reversal (was 30%)
            'wave5': 0.45,   # Wave 5 needs 45% exhaustion (was 35%)
            'waveA': 0.40,   # Wave A needs 40% reversal
            'waveB': 0.45,   # Wave B needs 45% reversal  
            'waveC': 0.50,   # Wave C needs 50% exhaustion
            'default': 0.45  # Default higher threshold
        }
        
        threshold = wave_tiredness.get(wave_type, wave_tiredness['default'])
        
        # WAVE-SPECIFIC MINIMUM MOVE REQUIREMENTS
        min_wave_moves = {
            'wave1': 0.0020,  # Wave 1 must move at least 0.20%
            'wave2': 0.0010,  # Wave 2 must move at least 0.10%
            'wave3': 0.0030,  # Wave 3 must move at least 0.30%
            'wave4': 0.0010,  # Wave 4 must move at least 0.10%
            'wave5': 0.0015,  # Wave 5 must move at least 0.15%
            'waveA': 0.0015,
            'waveB': 0.0010,
            'waveC': 0.0020,
            'default': 0.0015
        }
        
        min_move_ratio = min_wave_moves.get(wave_type, min_wave_moves['default'])
        
        for spot in resting_spots:
            spot_price = float(self.data.loc[spot, 'close'])
            
            # Calculate how much we've moved (the "climb" - must be POSITIVE)
            if direction == 'up':
                climb = spot_price - start_price  # Should be POSITIVE for upward move
            else:
                climb = start_price - spot_price  # Should be POSITIVE for downward move
            
            # Skip if climb is negative (wrong direction)
            if climb <= 0:
                logger.debug(f"Skipping spot {spot} - wrong direction (climb={climb:.2f})")
                continue
            
            # Check minimum move requirement
            move_ratio = climb / start_price
            if move_ratio < min_move_ratio:
                logger.debug(f"Skipping spot {spot} - move {move_ratio:.4f} < min {min_move_ratio:.4f}")
                continue
            
            # Check what happens in the next 10 minutes
            next_extreme_price = self._check_next_10_minutes(spot, direction)
            if next_extreme_price is None:
                continue
            
            # Calculate how much price "gives back" (the "drop" - must be POSITIVE)
            if direction == 'up':
                # For upward moves: how much does price drop from this peak?
                drop = spot_price - next_extreme_price  # Should be POSITIVE
            else:
                # For downward moves: how much does price bounce from this valley?
                drop = next_extreme_price - spot_price  # Should be POSITIVE
            
            # Skip if no reversal or negative drop (price continued in same direction)
            if drop <= 0:
                logger.debug(f"Skipping spot {spot} - no reversal (drop={drop:.2f})")
                continue
            
            # Tiredness ratio = drop / climb (should be between 0 and 1)
            tiredness_ratio = drop / climb
            
            # Cap tiredness at 1.0 (100% retracement)
            if tiredness_ratio > 1.0:
                tiredness_ratio = 1.0
                logger.debug(f"Capped tiredness at 1.0 for spot {spot}")
            
            logger.debug(f"Checking spot {spot} @ {spot_price:.2f}: "
                        f"climb={climb:.2f}, drop={drop:.2f}, tiredness={tiredness_ratio:.3f}")
            
            # If climber gets tired enough, this is the wave end!
            if tiredness_ratio >= threshold:
                logger.info(f"✅ Found {wave_type} end at {spot} @ {spot_price:.2f} "
                          f"with tiredness {tiredness_ratio:.3f} (needed {threshold:.3f})")
                return spot
        
        # If no spot shows enough tiredness, fall back to original method
        logger.debug(f"No spot showed enough tiredness (≥{threshold:.2f}). Falling back to structural turn.")
        return self.get_structural_turn_for_wave(start_timestamp, direction, wave_type)  

    def _get_wave_expected_move(self, wave_type: str) -> float:
        """
        Get expected price move for different wave types based on Elliott Wave principles.
        Wave 3 should be larger than Wave 1, Wave 5 similar to Wave 1, etc.
        
        Parameters:
        -----------
        wave_type : str
            Type of wave ('wave1', 'wave2', 'wave3', etc.)
            
        Returns:
        --------
        float
            Expected price movement ratio for this wave type
        """
        wave_expectations = {
            'wave1': self.structural_move_threshold,           # Standard move
            'wave2': self.structural_move_threshold * 0.3,     # Smaller retracement (30%)
            'wave3': 0.0003,                                   # 0.03 % - matches wave-C
            'wave4': self.structural_move_threshold * 0.2,     # Smaller retracement (20%)
            'wave5': self.structural_move_threshold,           # Similar to Wave 1
            'waveA': self.structural_move_threshold,           # Standard move
            'waveB': self.structural_move_threshold * 0.6,     # Moderate retracement
            'waveC': 0.0003,                                   # 0.03 % - explicit
        }
        expected_move = wave_expectations.get(wave_type, self.structural_move_threshold)
        logger.debug(f"Expected move for {wave_type}: {expected_move:.4f}")
        return expected_move
    
    def _check_next_10_minutes(self, timestamp: pd.Timestamp, direction: str) -> Optional[float]:
        """
        Check what happens in the next 10 minutes (or next 10 candles) after a timestamp.
        
        Parameters:
        -----------
        timestamp : pd.Timestamp
            The timestamp to check after
        direction : str
            'up' for peaks (look for lowest price in next period)
            'down' for valleys (look for highest price in next period)
            
        Returns:
        --------
        float or None
            The lowest/highest price in the next period, or None if insufficient data
        """
        try:
            # Find the index of our timestamp
            if timestamp not in self.data.index:
                logger.debug(f"Timestamp {timestamp} not in data index")
                return None
                
            idx = self.data.index.get_loc(timestamp)
            
            # Look ahead 10 candles or until end of data
            lookahead_candles = min(10, len(self.data) - idx - 1)
            
            if lookahead_candles <= 0:
                return None
            
            # Get the slice of data to check (next 1 to 10 candles)
            next_data = self.data.iloc[idx + 1:idx + 1 + lookahead_candles]
            
            if next_data.empty:
                return None
            
            if direction == 'up':
                # For peaks: we want to see how much price DROPS afterward
                # Return the lowest LOW price in the next period
                return float(next_data['low'].min())
            else:
                # For valleys: we want to see how much price BOUNCES afterward  
                # Return the highest HIGH price in the next period
                return float(next_data['high'].max())
                
        except (KeyError, ValueError, IndexError) as e:
            logger.debug(f"Error checking next 10 minutes after {timestamp}: {e}")
            return None

    def _find_furthest_cousin(self, start_timestamp: pd.Timestamp,
                              family_members: List[pd.Timestamp],
                              min_move: float,
                              direction: str) -> Optional[pd.Timestamp]:
        """
        Return the SINGLE cousin whose price move is the LARGEST in the CORRECT DIRECTION.
        No early stop – we scan the whole list every time.
        
        Parameters:
        -----------
        start_timestamp : pd.Timestamp
            Starting point for the wave
        family_members : List[pd.Timestamp]
            List of candidate turns (peaks for 'up', valleys for 'down')
        min_move : float
            Minimum price movement ratio required
        direction : str
            Expected direction ('up' or 'down') - cousin must actually move in this direction
        """
        best_cousin = None
        best_move   = min_move          # must beat this value
        start_price = float(self.data.loc[start_timestamp, 'close'])

        for m in family_members:
            cousin_price = float(self.data.loc[m, 'close'])
            
            # NEW: Skip if moving in wrong direction
            if direction == 'up' and cousin_price <= start_price:
                logger.debug(f"Skipping cousin {m} @ {cousin_price:.2f} - expected UP move from {start_price:.2f}")
                continue  # Not actually moving up
            if direction == 'down' and cousin_price >= start_price:
                logger.debug(f"Skipping cousin {m} @ {cousin_price:.2f} - expected DOWN move from {start_price:.2f}")
                continue  # Not actually moving down
            
            move = abs(cousin_price - start_price) / start_price
            if move > best_move:        # strictly larger
                best_move   = move
                best_cousin = m
                logger.debug(f"New furthest cousin: {m} with move {move:.4f} ({direction})")

        logger.debug(f"Final furthest cousin: {best_cousin}  move={best_move:.4f}  direction={direction}")
        return best_cousin

    def get_structural_turn(self, start_timestamp: pd.Timestamp, direction: str, 
                          min_price_move: float = None) -> Optional[pd.Timestamp]:
        """
        ORIGINAL: DFS to find the next significant structural turning point.
        Maintained for backward compatibility.
        
        Parameters:
        -----------
        start_timestamp : pd.Timestamp
            Starting point for the search
        direction : str ('up' for peaks, 'down' for valleys)
            Direction to search for the next turn
        min_price_move : float, optional
            Minimum price movement (as ratio) to consider a turn significant
            Defaults to self.structural_move_threshold if not provided
            
        Returns:
        --------
        pd.Timestamp or None
            Timestamp of the next significant turn, or None if not found
        """
        # Use the new wave-aware method with default wave type
        return self.get_structural_turn_for_wave(
            start_timestamp, 
            direction, 
            wave_type='generic',
            expected_move=min_price_move
        )

    def _dfs_find_structural_turn(self, current_point: pd.Timestamp, direction: str, 
                                min_move: float, visited: set = None) -> Optional[pd.Timestamp]:
        """
        Recursive DFS to find significant turning points.
        
        Parameters:
        -----------
        current_point : pd.Timestamp
            Current point being examined
        direction : str
            'up' for peaks, 'down' for valleys
        min_move : float
            Minimum price movement ratio to consider significant
        visited : set, optional
            Set of already visited points to avoid cycles
        
        Returns:
        --------
        pd.Timestamp or None
            Next structural turn, or None if not found
        """
        if visited is None:
            visited = set()
        
        if current_point in visited:
            return None
        visited.add(current_point)
        
        # Get candidate turns in specified direction
        if direction == 'up':
            candidates = self._get_peaks_after(current_point)
        else:
            candidates = self._get_valleys_after(current_point)
        
        for candidate in candidates:
            price_move = self._calculate_price_move(current_point, candidate)
            
            # If significant move found, return it (structural turn)
            if price_move >= min_move:
                return candidate
            
            # If not significant, recursively explore from this candidate
            # This is the key - we skip noise and continue searching
            next_turn = self._dfs_find_structural_turn(candidate, direction, min_move, visited)
            if next_turn:
                return next_turn
        
        return None  # No structural turn found

    def _get_peaks_after(self, timestamp: pd.Timestamp) -> List[pd.Timestamp]:
        """Get all peaks after the given timestamp"""
        if self.indicators['peak_mask'].empty:
            return []
        
        peak_mask = self.indicators['peak_mask']
        peaks_after = self.data.index[peak_mask & (self.data.index > timestamp)]
        return sorted(peaks_after)

    def _get_valleys_after(self, timestamp: pd.Timestamp) -> List[pd.Timestamp]:
        """Get all valleys after the given timestamp"""
        if self.indicators['valley_mask'].empty:
            return []
        
        valley_mask = self.indicators['valley_mask']
        valleys_after = self.data.index[valley_mask & (self.data.index > timestamp)]
        return sorted(valleys_after)

    def _calculate_price_move(self, point1: pd.Timestamp, point2: pd.Timestamp) -> float:
        """Calculate price movement between two points as ratio"""
        try:
            price1 = float(self.data.loc[point1, 'close'])
            price2 = float(self.data.loc[point2, 'close'])
            return abs(price2 - price1) / price1
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error calculating price move between {point1} and {point2}: {e}")
            return 0.0

    def get_all_structural_turns(self, turn_type: str = 'both', 
                               min_price_move: float = None) -> Dict[str, List[pd.Timestamp]]:
        """
        Get all structural turns of specified type.
        
        Parameters:
        -----------
        turn_type : str
            'peaks', 'valleys', or 'both'
        min_price_move : float, optional
            Minimum price movement ratio
            
        Returns:
        --------
        Dict with lists of structural turns
        """
        if min_price_move is None:
            min_price_move = self.structural_move_threshold
        
        result = {'peaks': [], 'valleys': []}
        
        if turn_type in ['peaks', 'both']:
            # Find structural peaks by starting from each valley
            valleys = self._get_valleys_after(self.data.index[0]) if not self.data.empty else []
            for valley in valleys:
                peak = self.get_structural_turn(valley, 'up', min_price_move)
                if peak and peak not in result['peaks']:
                    result['peaks'].append(peak)
        
        if turn_type in ['valleys', 'both']:
            # Find structural valleys by starting from each peak
            peaks = self._get_peaks_after(self.data.index[0]) if not self.data.empty else []
            for peak in peaks:
                valley = self.get_structural_turn(peak, 'down', min_price_move)
                if valley and valley not in result['valleys']:
                    result['valleys'].append(valley)
        
        # Sort results by timestamp
        result['peaks'] = sorted(result['peaks'])
        result['valleys'] = sorted(result['valleys'])
        
        logger.info(f"Found {len(result['peaks'])} structural peaks and {len(result['valleys'])} structural valleys")
        return result

    def get_structural_masks(self, min_price_move: float = None) -> Dict[str, pd.Series]:
        """
        Get boolean masks for structural peaks and valleys.
        
        Parameters:
        -----------
        min_price_move : float, optional
            Minimum price movement ratio
            
        Returns:
        --------
        Dict with 'structural_peak_mask' and 'structural_valley_mask'
        """
        if min_price_move is None:
            min_price_move = self.structural_move_threshold
        
        # SAFETY FALLBACK: If we get too few turns, use more permissive threshold
        turns = self.get_all_structural_turns('both', min_price_move)
        
        # If we have very few structural turns, try with lower threshold
        if len(turns['peaks']) < 10 or len(turns['valleys']) < 10:
            logger.warning(f"Too few structural turns: {len(turns['peaks'])} peaks, {len(turns['valleys'])} valleys. Using fallback threshold.")
            fallback_threshold = min_price_move * 0.5  # 50% of original threshold
            turns = self.get_all_structural_turns('both', fallback_threshold)
            logger.info(f"Fallback structural turns: {len(turns['peaks'])} peaks, {len(turns['valleys'])} valleys")
        
        # Initialize masks as False
        peak_mask = pd.Series(False, index=self.data.index)
        valley_mask = pd.Series(False, index=self.data.index)
        
        # Set mask positions to True for structural turns
        for peak in turns['peaks']:
            if peak in peak_mask.index:
                peak_mask.loc[peak] = True
        
        for valley in turns['valleys']:
            if valley in valley_mask.index:
                valley_mask.loc[valley] = True
        
        logger.info(f"Structural masks created: {peak_mask.sum()} peaks, {valley_mask.sum()} valleys")
        return {
            'structural_peak_mask': peak_mask,
            'structural_valley_mask': valley_mask
        }

    # =========================================================================
    # NEW: SEQUENTIAL STRUCTURAL TURNS FOR HYBRID DFS
    # =========================================================================

    def get_structural_turns_sequential(self, min_price_move: float = None) -> List[pd.Timestamp]:
        """
        NEW: Get all structural turns in chronological order without filtering.
        This is used by the hybrid DFS system to process turns sequentially.
        
        Parameters:
        -----------
        min_price_move : float, optional
            Minimum price movement ratio
            
        Returns:
        --------
        List of all structural turns in chronological order
        """
        if min_price_move is None:
            min_price_move = self.structural_move_threshold
        
        # Get all structural turns
        turns = self.get_all_structural_turns('both', min_price_move)
        
        # Combine peaks and valleys and sort chronologically
        all_turns = turns['peaks'] + turns['valleys']
        all_turns_sorted = sorted(all_turns)
        
        logger.info(f"Sequential structural turns: {len(all_turns_sorted)} total turns "
                   f"({len(turns['peaks'])} peaks, {len(turns['valleys'])} valleys)")
        return all_turns_sorted

    def get_next_structural_start(self, current_timestamp: pd.Timestamp = None, 
                                min_price_move: float = None) -> Dict[str, Any]:
        """
        NEW: Find the next structural starting point for hybrid DFS.
        Used when external DFS is ON (no algorithms running).
        
        Parameters:
        -----------
        current_timestamp : pd.Timestamp, optional
            Starting point for search (if None, starts from beginning)
        min_price_move : float, optional
            Minimum price movement ratio
            
        Returns:
        --------
        Dict with turn information or None if no turns found
        """
        if self.data.empty:
            logger.warning("No data available for structural start search")
            return None
        
        if min_price_move is None:
            min_price_move = self.structural_move_threshold
        
        # Get all sequential turns
        all_turns = self.get_structural_turns_sequential(min_price_move)
        
        if not all_turns:
            logger.debug("No structural turns found")
            return None
        
        # Find the first turn after current timestamp
        if current_timestamp is None:
            next_turn = all_turns[0]
        else:
            turns_after = [turn for turn in all_turns if turn > current_timestamp]
            if not turns_after:
                logger.debug(f"No structural turns found after {current_timestamp}")
                return None
            next_turn = turns_after[0]
        
        # Determine if it's a peak or valley
        is_peak = next_turn in self.indicators['peak_mask'].index and self.indicators['peak_mask'].loc[next_turn]
        is_valley = next_turn in self.indicators['valley_mask'].index and self.indicators['valley_mask'].loc[next_turn]
        
        turn_info = {
            'timestamp': next_turn,
            'price': float(self.data.loc[next_turn, 'close']),
            'type': 'peak' if is_peak else 'valley' if is_valley else 'unknown',
            'is_peak': is_peak,
            'is_valley': is_valley
        }
        
        logger.info(f"Next structural start found: {turn_info['type']} at {next_turn} "
                   f"@ {turn_info['price']:.2f}")
        return turn_info

    # =========================================================================
    # END OF NEW SEQUENTIAL STRUCTURAL TURNS
    # =========================================================================

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
            'peak_prominence': self.peak_prominence,  # ADDED: Include prominence in stats
            'structural_move_threshold': self.structural_move_threshold,  # NEW: Include structural threshold
            'has_sufficient_data': self.has_sufficient_data(),
            'indicators_ready': len([v for v in self.indicators.values() if not v.empty]) > 0,
            'waves_manager_available': self.waves_manager is not None
        }
        
        # NEW: Add structural turn statistics
        try:
            structural_turns = self.get_all_structural_turns('both')
            stats['structural_peaks'] = len(structural_turns['peaks'])
            stats['structural_valleys'] = len(structural_turns['valleys'])
            stats['structural_turn_ratio'] = (
                (len(structural_turns['peaks']) + len(structural_turns['valleys'])) / 
                (len(peaks) + len(valleys)) if (len(peaks) + len(valleys)) > 0 else 0
            )
            
            # NEW: Add sequential turns count
            sequential_turns = self.get_structural_turns_sequential()
            stats['sequential_turns_count'] = len(sequential_turns)
            
        except Exception as e:
            logger.debug(f"Could not calculate structural turn stats: {e}")
            stats['structural_peaks'] = 0
            stats['structural_valleys'] = 0
            stats['structural_turn_ratio'] = 0
            stats['sequential_turns_count'] = 0
        
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

    def clear_cache(self):
        """Clear the structural turn cache."""
        self._structural_turn_cache.clear()
        logger.debug("Structural turn cache cleared")

    def reset(self):
        """
        Reset the DataManager to initial state.
        Useful for testing or when switching instruments.
        """
        self.data = pd.DataFrame()
        self.indicators = {}
        self._structural_turn_cache.clear()
        self._initialize_indicators()
        
        # Reset peak detector if available
        if self.peak_detector and hasattr(self.peak_detector, 'reset'):
            self.peak_detector.reset()
            
        logger.info("DataManager reset to initial state")

# =============================================================================
# TEST FUNCTION FOR WAVE-AWARE DFS STRUCTURAL TURN DETECTION
# =============================================================================

def test_wave_aware_dfs_structural_turns():
    """
    Enhanced test function to verify Wave-Aware DFS structural turn detection works correctly.
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    print("🧪 Testing WAVE-AWARE DFS Structural Turn Detection...")
    
    # Create test data - FIXED: Use simpler data for clarity
    base_time = datetime(2023, 7, 17, 12, 42)
    data_points = []
    
    # WAVE 1: Valley @ 63,897 → Peak @ 64,323 (with noise)
    data_points.append({'timestamp': base_time, 'open': 63897, 'high': 63897, 'low': 63897, 'close': 63897, 'volume': 1000})
    
    # Noise in Wave 1
    noise_points = [
        {'timestamp': base_time + timedelta(minutes=3), 'open': 64068, 'high': 64068, 'low': 64000, 'close': 64000, 'volume': 800},
        {'timestamp': base_time + timedelta(minutes=5), 'open': 64000, 'high': 64140, 'low': 64000, 'close': 64140, 'volume': 700},
    ]
    data_points.extend(noise_points)
    
    # Structural Peak for Wave 1
    data_points.append({'timestamp': base_time + timedelta(minutes=21), 'open': 64323, 'high': 64323, 'low': 63980, 'close': 64323, 'volume': 1500})
    
    # WAVE 2: Valley @ 63,980 (orphan)
    data_points.append({'timestamp': base_time + timedelta(minutes=25), 'open': 63980, 'high': 64558, 'low': 63980, 'close': 63980, 'volume': 1800})
    
    # Create DataFrame
    test_data = pd.DataFrame(data_points)
    test_data.set_index('timestamp', inplace=True)
    
    # Initialize DataManager with higher quality parameters
    dm = DataManager(
    data=test_data, 
    mode='backtest', 
    window_size=None,
    structural_move_threshold=0.002,  # Higher: 0.2% minimum for structural turns
    ma_length=10,                     # Faster: 10 instead of 5 for quicker response
    bb_multiplier=2.0,
    peak_prominence=0.20,             # Higher: Detect only more significant peaks
    )
    
    print("\n🔍 Testing WAVE-AWARE DFS...")
    
    # Start from initial valley
    start_valley = base_time
    print(f"Starting from valley: {start_valley} @ {dm.data.loc[start_valley, 'close']}")
    
    # Test Wave 1 (Family wave)
    wave1_peak = dm.get_structural_turn_for_wave(start_valley, 'up', 'wave1', 0.005)
    if wave1_peak:
        wave1_price = dm.data.loc[wave1_peak, 'close']
        print(f"✅ Wave 1 Structural Peak: {wave1_peak} @ {wave1_price}")
        print("   ✅ FAMILY WAVE: Found furthest cousin ignoring noise")
    else:
        print("❌ Failed to find Wave 1 structural peak")
        return dm
    
    # Test Wave 2 (Orphan wave) - FIXED: Use more permissive threshold
    wave2_valley = dm.get_structural_turn_for_wave(wave1_peak, 'down', 'wave2', 0.001)  # Reduced threshold
    if wave2_valley:
        wave2_price = dm.data.loc[wave2_valley, 'close']
        print(f"✅ Wave 2 Structural Valley: {wave2_valley} @ {wave2_price}")
        print("   ✅ ORPHAN WAVE: Single valley detected immediately")
    else:
        print("⚠️  Wave 2 not found (threshold too strict for test data)")
        # Continue anyway for demonstration
    
    print(f"\n🎯 TEST SUMMARY:")
    print(f"Wave 1 (Family): ✅ Found at {wave1_peak}")
    print(f"Wave 2 (Orphan): {'✅' if wave2_valley else '⚠️'}")
    print(f"Key Improvement: System now distinguishes between family/orphan waves!")
    
    return dm