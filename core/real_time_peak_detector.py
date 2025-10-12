# algostack/core/real_time_peak_detector.py
import pandas as pd
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import logging
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class RealTimePeakDetector:
    """
    Real-time peak and valley detection using scipy.find_peaks.
    Uses same proven algorithm as DataManager for consistency.
    Designed to work with 1-minute crypto data and provide reliable
    peak/valley detection for Elliott Wave algorithms.
    """
    
    def __init__(self, window_size: int = 50, prominence: float = 0.5, 
                 min_peak_distance: int = 5):
        """
        Initialize the real-time peak detector.
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window for peak detection
        prominence : float
            Minimum prominence for peak/valley validation (same as DataManager)
        min_peak_distance : int
            Minimum bars between consecutive peaks/valleys
        """
        self.window_size = window_size
        self.prominence = prominence
        self.min_peak_distance = min_peak_distance
        
        # Data buffers - store (timestamp, high, low, close)
        self.highs_buffer = deque(maxlen=window_size)
        self.lows_buffer = deque(maxlen=window_size)
        self.closes_buffer = deque(maxlen=window_size)
        self.timestamps_buffer = deque(maxlen=window_size)
        
        # Detected peaks and valleys (confirmed)
        self.confirmed_peaks = []  # List of (timestamp, price)
        self.confirmed_valleys = []  # List of (timestamp, price)
        
        logger.info(f"RealTimePeakDetector initialized: "
                   f"window_size={window_size}, prominence={prominence}, "
                   f"min_peak_distance={min_peak_distance}")

    def update(self, high: float, low: float, close: float, timestamp: pd.Timestamp) -> Tuple[List, List]:
        """
        Update with new price data and return current confirmed peaks/valleys.
        
        Parameters:
        -----------
        high : float
            Current high price
        low : float
            Current low price  
        close : float
            Current close price
        timestamp : pd.Timestamp
            Current timestamp
            
        Returns:
        --------
        Tuple[List, List]
            (confirmed_peaks, confirmed_valleys) as lists of timestamps
        """
        try:
            # Add new data to buffers
            self.highs_buffer.append(high)
            self.lows_buffer.append(low)
            self.closes_buffer.append(close)
            self.timestamps_buffer.append(timestamp)
            
            # Need minimum data for detection
            if len(self.highs_buffer) < 10:
                return self._get_confirmed_timestamps()
            
            # Convert buffers to numpy arrays for scipy
            highs_array = np.array(self.highs_buffer)
            lows_array = np.array(self.lows_buffer)
            
            # Detect peaks using scipy.find_peaks (same as DataManager)
            peaks, _ = find_peaks(highs_array, prominence=self.prominence, distance=self.min_peak_distance)
            valleys, _ = find_peaks(-lows_array, prominence=self.prominence, distance=self.min_peak_distance)
            
            # Convert indices to timestamps and prices
            new_peaks = []
            for idx in peaks:
                timestamp_peak = self.timestamps_buffer[idx]
                price_peak = highs_array[idx]
                if not any(peak[0] == timestamp_peak for peak in self.confirmed_peaks):
                    self.confirmed_peaks.append((timestamp_peak, price_peak))
                    new_peaks.append(timestamp_peak)
                    logger.debug(f"Detected peak: {timestamp_peak} at {price_peak:.2f}")
            
            new_valleys = []
            for idx in valleys:
                timestamp_valley = self.timestamps_buffer[idx]
                price_valley = lows_array[idx]
                if not any(valley[0] == timestamp_valley for valley in self.confirmed_valleys):
                    self.confirmed_valleys.append((timestamp_valley, price_valley))
                    new_valleys.append(timestamp_valley)
                    logger.debug(f"Detected valley: {timestamp_valley} at {price_valley:.2f}")
            
            # Clean old extremes
            self._clean_old_extremes()
            
            return self._get_confirmed_timestamps()
            
        except Exception as e:
            logger.error(f"Error in peak detector update: {str(e)}")
            return self._get_confirmed_timestamps()

    def _clean_old_extremes(self):
        """Remove peaks/valleys that are too old (outside our effective window)."""
        if not self.timestamps_buffer:
            return
        
        current_time = self.timestamps_buffer[-1]
        effective_window_start = current_time - pd.Timedelta(minutes=self.window_size * 2)
        
        # Clean confirmed peaks/valleys
        self.confirmed_peaks = [
            (ts, price) for ts, price in self.confirmed_peaks 
            if ts >= effective_window_start
        ]
        self.confirmed_valleys = [
            (ts, price) for ts, price in self.confirmed_valleys 
            if ts >= effective_window_start
        ]

    def _get_confirmed_timestamps(self) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """Get confirmed peaks and valleys as timestamp lists only."""
        peak_timestamps = [ts for ts, _ in self.confirmed_peaks]
        valley_timestamps = [ts for ts, _ in self.confirmed_valleys]
        return peak_timestamps, valley_timestamps

    def get_current_peaks_valleys(self) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """Get current confirmed peaks and valleys as timestamps."""
        return self._get_confirmed_timestamps()

    def get_peak_valley_prices(self) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
        """Get current confirmed peaks and valleys with prices."""
        return self.confirmed_peaks, self.confirmed_valleys

    def get_peak_valley_masks(self, data_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
        """
        Convert confirmed peaks/valleys to boolean masks for a given index.
        
        Parameters:
        -----------
        data_index : pd.DatetimeIndex
            Index to create masks for
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (peak_mask, valley_mask) as boolean Series
        """
        peak_mask = pd.Series(False, index=data_index)
        valley_mask = pd.Series(False, index=data_index)
        
        for peak_time, _ in self.confirmed_peaks:
            if peak_time in data_index:
                peak_mask.loc[peak_time] = True
        
        for valley_time, _ in self.confirmed_valleys:
            if valley_time in data_index:
                valley_mask.loc[valley_time] = True
        
        return peak_mask, valley_mask

    def get_recent_extremes(self, lookback_bars: int = 10) -> Tuple[List, List]:
        """
        Get the most recent peaks and valleys.
        
        Parameters:
        -----------
        lookback_bars : int
            Number of most recent extremes to return
            
        Returns:
        --------
        Tuple[List, List]
            (recent_peaks, recent_valleys) as lists of (timestamp, price) tuples
        """
        # Return the most recent extremes
        recent_peaks = self.confirmed_peaks[-lookback_bars:] if self.confirmed_peaks else []
        recent_valleys = self.confirmed_valleys[-lookback_bars:] if self.confirmed_valleys else []
        
        return recent_peaks, recent_valleys

    def get_detection_stats(self) -> Dict[str, any]:
        """Get statistics about current detection state."""
        return {
            'total_peaks_detected': len(self.confirmed_peaks),
            'total_valleys_detected': len(self.confirmed_valleys),
            'buffer_size': len(self.highs_buffer),
            'window_size': self.window_size,
            'prominence': self.prominence,
            'min_peak_distance': self.min_peak_distance
        }

    def reset(self):
        """Reset detector state completely."""
        self.highs_buffer.clear()
        self.lows_buffer.clear()
        self.closes_buffer.clear()
        self.timestamps_buffer.clear()
        self.confirmed_peaks.clear()
        self.confirmed_valleys.clear()
        
        logger.info("RealTimePeakDetector reset complete")


# Test the RealTimePeakDetector
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing RealTimePeakDetector with scipy.find_peaks...")
    
    # Create test data (sine wave with noise to simulate price movements)
    detector = RealTimePeakDetector(
        window_size=100,  # 300 bars total
        prominence=0.5,   # Same as DataManager - PROVEN TO WORK
        min_peak_distance=3
    )
    
    # Generate test data with clear peaks and valleys
    timestamps = pd.date_range(start='2024-01-01', periods=300, freq='1min')
    
    # Create a sine wave with some noise - MORE VOLATILE for better detection
    t = np.linspace(0, 12*np.pi, 300)
    base_prices = 100 + 20 * np.sin(t) + np.random.normal(0, 3.0, 300)  # Increased volatility
    
    print("Simulating real-time price updates...")
    
    detected_peaks = []
    detected_valleys = []
    
    for i, (ts, price) in enumerate(zip(timestamps, base_prices)):
        # Create realistic high/low/close with some spread
        spread = 0.2
        high = price + abs(np.random.normal(0, spread))
        low = price - abs(np.random.normal(0, spread))
        close = price
        
        # Update detector
        peaks, valleys = detector.update(high, low, close, ts)
        
        # Track detections
        if ts in peaks:
            detected_peaks.append((ts, high))
        if ts in valleys:
            detected_valleys.append((ts, low))
        
        # Print status periodically
        if i % 30 == 0:
            stats = detector.get_detection_stats()
            print(f"\nBar {i}:")
            print(f"  Confirmed: {stats['total_peaks_detected']} peaks, {stats['total_valleys_detected']} valleys")
            print(f"  Buffer size: {stats['buffer_size']}")
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    
    stats = detector.get_detection_stats()
    print(f"\nDetection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal peaks detected: {len(detected_peaks)}")
    print(f"Total valleys detected: {len(detected_valleys)}")
    
    # EXPECTED: 8-15 peaks AND 8-15 valleys
    if len(detected_peaks) >= 8 and len(detected_valleys) >= 8:
        print("✅ SUCCESS: Realistic peak/valley detection achieved!")
    else:
        print("❌ WARNING: Detection counts still low - may need parameter adjustment")
    
    if detected_peaks:
        print(f"\nRecent peaks (timestamp, price):")
        for ts, price in detected_peaks[-5:]:
            print(f"  {ts}: {price:.2f}")
    
    if detected_valleys:
        print(f"\nRecent valleys (timestamp, price):")
        for ts, price in detected_valleys[-5:]:
            print(f"  {ts}: {price:.2f}")
    
    # Test mask generation
    print(f"\nTesting mask generation...")
    test_index = pd.date_range(start='2024-01-01', periods=300, freq='1min')
    peak_mask, valley_mask = detector.get_peak_valley_masks(test_index)
    
    print(f"Peak mask has {peak_mask.sum()} True values")
    print(f"Valley mask has {valley_mask.sum()} True values")
    
    print("\nRealTimePeakDetector test completed!")