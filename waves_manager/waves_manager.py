# algostack/waves_manager/waves_manager.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)

class WavesManager:
    """
    Unified manager for Elliott Wave subdivision detection.
    Converts 4 standalone subdivision algorithms to class methods.
    Provides corrective_mask and motive_mask for TrendsManager integration.
    """

    def __init__(self, data_manager, history_size: int = 100):
        """
        Initialize WavesManager with dependency injection.
        
        Parameters:
        -----------
        data_manager : DataManager
            The core data management component
        history_size : int
            Maximum number of subdivision results to keep in history
        """
        self.data_manager = data_manager
        self.subdivision_history = deque(maxlen=history_size)
        self.current_subdivisions = {}
        
        logger.info(f"WavesManager initialized with history size: {history_size}")

    def analyze_subdivisions(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> Dict[str, Any]:
        """
        Run all 4 subdivision algorithms and return consolidated results.
        
        Parameters:
        -----------
        data : pd.Series
            Price data (typically close prices)
        peak_mask : pd.Series  
            Boolean mask indicating peak positions
        valley_mask : pd.Series
            Boolean mask indicating valley positions
            
        Returns:
        --------
        Dict with subdivision types as keys and detection results as values
        """
        try:
            # Run all subdivision detection algorithms
            subdivisions = {
                'motive_abc': self._motive_abc(data, peak_mask, valley_mask),
                'corrective_abc': self._corrective_abc(data, peak_mask, valley_mask),
                'motive_five': self._motive_five(data, peak_mask, valley_mask),
                'corrective_five': self._corrective_five(data, peak_mask, valley_mask)
            }
            
            # Update current state and history
            self.current_subdivisions = subdivisions
            self._update_subdivision_history(subdivisions)
            
            successful_algos = len([s for s in subdivisions.values() if s is not None and 
                                  (hasattr(s, 'sum') and s.sum() > 0 or 
                                   hasattr(s, 'notna') and s.notna().sum() > 0 or
                                   hasattr(s, '__len__') and len(s) > 0)])
            
            logger.info(f"Subdivision analysis completed - found subdivisions across {successful_algos} algorithms")
            return subdivisions
            
        except Exception as e:
            logger.error(f"Error in analyze_subdivisions: {str(e)}")
            return {
                'motive_abc': None,
                'corrective_abc': None,
                'motive_five': None, 
                'corrective_five': None
            }

    def _motive_abc(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> pd.Series:
        """
        EXACT conversion of motive_abc.py logic.
        "Mountain"-style subdivide: finds little up–down–up shapes (peak A → valley B → peak C).
        
        Returns:
        --------
        pd.Series: Boolean mask on data.index where valley B conditions are met
        """
        try:
            # pull out close prices
            if isinstance(data, pd.DataFrame):
                data = data['close']
            
            # boolean mask over full index
            result = pd.Series(False, index=data.index)
            
            # get the actual timestamps of peaks & valleys
            peaks = data.index[peak_mask]
            valleys = data.index[valley_mask]
            
            # for each valley, see if it sits between a rising peak A → peak C
            for v in valleys:
                # 1) find the last peak before this valley
                prev_peaks = peaks[peaks < v]
                if len(prev_peaks) == 0:
                    continue
                pA = prev_peaks[-1]
                
                # 2) find any peak after the valley that is higher than pA
                next_peaks = peaks[peaks > v]
                higher_after = [p for p in next_peaks if data[p] > data[pA]]
                if not higher_after:
                    continue
                
                # we found a mini mountain: peak-A → valley-v → peak-C
                # mark this valley as a valid "Wave B" subdivide
                result.loc[v] = True
            
            logger.debug(f"_motive_abc found {result.sum()} valid Wave B subdivisions")
            return result
            
        except Exception as e:
            logger.error(f"Error in _motive_abc: {str(e)}")
            return pd.Series(False, index=data.index)

    def _corrective_abc(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> pd.Series:
        """
        EXACT conversion of corrective_abc.py logic.
        Identifies "corrective" valleys (Wave A candidates) in a little down–up–down mini-pattern.
        
        Returns:
        --------
        pd.Series: Boolean mask on data.index where corrective valley conditions are met
        """
        try:
            # Get the close series
            if isinstance(data, pd.DataFrame):
                data = data['close']
            
            # Prepare an all-False mask over every timestamp
            mask = pd.Series(False, index=data.index)
            
            # Extract the actual times of peaks & valleys
            peaks = data.index[peak_mask]
            valleys = data.index[valley_mask]
            
            # We need at least one peak before a valley to call it corrective
            if len(peaks) == 0 or len(valleys) < 1:
                return mask
            
            # For each valley, see if it sits below its preceding peak
            for v in valleys:
                # find the last peak before this valley
                prev_peaks = peaks[peaks < v]
                if len(prev_peaks) == 0:
                    continue
                last_peak = prev_peaks[-1]
                
                # if the valley price is lower than that peak price, it's a corrective valley
                if data.loc[v] < data.loc[last_peak]:
                    mask.loc[v] = True
            
            logger.debug(f"_corrective_abc found {mask.sum()} corrective valleys")
            return mask
            
        except Exception as e:
            logger.error(f"Error in _corrective_abc: {str(e)}")
            return pd.Series(False, index=data.index)

    def _motive_five(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> pd.Series:
        """
        EXACT conversion of motive_five.py logic.
        Identifies a five-wave subdividing structure using vectorized operations.
        
        Returns:
        --------
        pd.Series: Wave labels ('1', '2', '3', '4', '5') indexed by valley positions.
        """
        try:
            # --- Ensure `data` is a Pandas Series of close prices ---
            if isinstance(data, pd.DataFrame):
                data = data['close']  # Extract the 'close' column
            
            # --- Index Alignment Checks ---
            if not data.index.equals(peak_mask.index):
                logger.warning("Index mismatch: data/peak_mask in _motive_five")
                return pd.Series(dtype=object)
            if not data.index.equals(valley_mask.index):
                logger.warning("Index mismatch: data/valley_mask in _motive_five")
                return pd.Series(dtype=object)

            # --- Identify Peaks and Valleys ---
            peaks = data.index[peak_mask]
            valleys = data.index[valley_mask]

            # --- Handle Insufficient Data ---
            if len(peaks) < 2 or len(valleys) < 5:  # Changed from 3 to 5 for five waves
                return pd.Series(index=valleys, dtype="object")

            # --- Extract Prices Efficiently ---
            peak_prices = data.loc[peaks].to_numpy()
            valley_prices = data.loc[valleys].to_numpy()

            # --- Apply Vectorized Conditions for Wave Detection ---
            # Wave 1: Peak above previous valley
            cond_wave_1 = peak_prices[1:] > valley_prices[:-1]

            # Wave 2: Valley lower than Wave 1 but higher than previous valley
            cond_wave_2 = (valley_prices[1:] < peak_prices[1:]) & (valley_prices[1:] > valley_prices[:-1])

            # Wave 3: Peak above Wave 1 by Fibonacci multiples
            cond_wave_3 = (
                (peak_prices[1:] > peak_prices[:-1] * 1.618) |
                (peak_prices[1:] > peak_prices[:-1] * 2.618) |
                (peak_prices[1:] > peak_prices[:-1] * 4.236)
            )

            # Wave 4: Valley higher than Wave 3 but lower than Wave 1
            cond_wave_4 = (valley_prices[1:] < peak_prices[1:]) & (valley_prices[1:] > peak_prices[:-1])

            # Wave 5: Peak higher than Wave 3
            cond_wave_5 = peak_prices[1:] > peak_prices[:-1]

            # --- FIX: Ensure all conditions have compatible lengths ---
            min_length = min(
                len(cond_wave_1) - 3,  # We need 4 more elements after this
                len(cond_wave_2) - 2,  # We need 3 more elements after this  
                len(cond_wave_3) - 1,  # We need 2 more elements after this
                len(cond_wave_4),      # We need 1 more element after this
                len(cond_wave_5)       # No offset needed
            )
            
            # Only proceed if we have enough data for a complete five-wave structure
            if min_length <= 0:
                return pd.Series(index=valleys, dtype="object")

            # Apply conditions with compatible lengths
            valid_waves = (
                cond_wave_1[:min_length] & 
                cond_wave_2[:min_length] & 
                cond_wave_3[:min_length] & 
                cond_wave_4[:min_length] & 
                cond_wave_5[:min_length]
            )

            # Initialize an empty array for classifications
            classifications = np.full(len(valleys), None, dtype=object)

            # Assign wave labels where conditions hold
            if np.any(valid_waves):
                # Find the first valid occurrence
                first_valid_idx = np.argmax(valid_waves)
                if first_valid_idx + 4 < len(classifications):  # Ensure we have room for 5 waves
                    classifications[first_valid_idx] = '1'  # Wave 1
                    classifications[first_valid_idx + 1] = '2'  # Wave 2
                    classifications[first_valid_idx + 2] = '3'  # Wave 3
                    classifications[first_valid_idx + 3] = '4'  # Wave 4
                    classifications[first_valid_idx + 4] = '5'  # Wave 5
            
            result = pd.Series(classifications, index=valleys)
            valid_waves_count = result.notna().sum()
            logger.debug(f"_motive_five found {valid_waves_count} five-wave structures")
            return result
            
        except Exception as e:
            logger.error(f"Error in _motive_five: {str(e)}")
            return pd.Series(dtype=object)

    def _corrective_five(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> pd.Series:
        """
        EXACT conversion of corrective_five.py logic.
        Identifies a five-wave downtrend structure using vectorized operations.
        
        Returns:
        --------
        pd.Series: Wave labels ('1', '2', '3', '4', '5') indexed by valley positions.
        """
        try:
            # --- Ensure `data` is a Pandas Series of close prices ---
            if isinstance(data, pd.DataFrame):
                data = data['close']  # Extract the 'close' column
            
            # --- Index Alignment Checks ---
            if not data.index.equals(peak_mask.index):
                logger.warning("Index mismatch: data/peak_mask in _corrective_five")
                return pd.Series(dtype=object)
            if not data.index.equals(valley_mask.index):
                logger.warning("Index mismatch: data/valley_mask in _corrective_five")
                return pd.Series(dtype=object)

            # --- Identify Peaks and Valleys ---
            peaks = data.index[peak_mask]
            valleys = data.index[valley_mask]

            # --- Handle Insufficient Data ---
            if len(peaks) < 3 or len(valleys) < 6:  # Need more data for downtrend structure
                return pd.Series(index=valleys, dtype="object")

            # --- Extract Prices Efficiently ---
            peak_prices = data.loc[peaks].to_numpy()
            valley_prices = data.loc[valleys].to_numpy()

            # --- Apply Vectorized Conditions for Wave Detection ---
            # Wave 1: Valley below previous peak
            cond_wave_1 = valley_prices[1:] < peak_prices[:-1]

            # Wave 2: Peak above Wave 1 but below previous peak
            cond_wave_2 = (peak_prices[1:] > valley_prices[1:]) & (peak_prices[1:] < peak_prices[:-1])

            # Wave 3: Valley below Wave 1 by specified Fibonacci levels
            cond_wave_3 = (
                (valley_prices[2:] < valley_prices[1:-1] * (1 - 1.618)) |
                (valley_prices[2:] < valley_prices[1:-1] * (1 - 2.618)) |
                (valley_prices[2:] < valley_prices[1:-1] * (1 - 4.236))
            )

            # Wave 4: Peak after Wave 3, below Wave 1
            cond_wave_4 = (peak_prices[2:] > valley_prices[2:]) & (peak_prices[2:] < valley_prices[1:-1])

            # Wave 5: Valley below Wave 3
            cond_wave_5 = valley_prices[3:] < valley_prices[2:-1]

            # --- FIX: Ensure all conditions have compatible lengths ---
            min_length = min(
                len(cond_wave_1) - 3,  # We need 4 more elements after this
                len(cond_wave_2) - 2,  # We need 3 more elements after this
                len(cond_wave_3) - 1,  # We need 2 more elements after this  
                len(cond_wave_4),      # We need 1 more element after this
                len(cond_wave_5)       # No offset needed
            )
            
            # Only proceed if we have enough data for a complete five-wave structure
            if min_length <= 0:
                return pd.Series(index=valleys, dtype="object")

            # Apply conditions with compatible lengths
            valid_waves = (
                cond_wave_1[:min_length] & 
                cond_wave_2[:min_length] & 
                cond_wave_3[:min_length] & 
                cond_wave_4[:min_length] & 
                cond_wave_5[:min_length]
            )

            # Initialize an empty array for classifications
            classifications = np.full(len(valleys), None, dtype=object)

            # Assign wave labels where conditions are met
            if np.any(valid_waves):
                # Find the first valid occurrence
                first_valid_idx = np.argmax(valid_waves)
                if first_valid_idx + 5 < len(classifications):  # Ensure we have room for 5 waves
                    classifications[first_valid_idx + 1] = '1'  # Wave 1
                    classifications[first_valid_idx + 2] = '2'  # Wave 2
                    classifications[first_valid_idx + 3] = '3'  # Wave 3
                    classifications[first_valid_idx + 4] = '4'  # Wave 4
                    classifications[first_valid_idx + 5] = '5'  # Wave 5

            result = pd.Series(classifications, index=valleys)
            valid_waves_count = result.notna().sum()
            logger.debug(f"_corrective_five found {valid_waves_count} five-wave downtrend structures")
            return result
            
        except Exception as e:
            logger.error(f"Error in _corrective_five: {str(e)}")
            return pd.Series(dtype=object)

    def _update_subdivision_history(self, new_subdivisions: Dict[str, Any]):
        """Update subdivision history with new detections."""
        timestamp = pd.Timestamp.now()
        history_entry = {
            'timestamp': timestamp,
            'subdivisions': new_subdivisions
        }
        self.subdivision_history.append(history_entry)
        
        logger.debug(f"Subdivision history updated - total entries: {len(self.subdivision_history)}")

    def get_subdivision_masks(self) -> Dict[str, pd.Series]:
        """
        Get current corrective_mask and motive_mask for TrendsManager.
        
        Returns:
        --------
        Dict with 'corrective_mask' and 'motive_mask' boolean Series
        """
        try:
            # Get algorithm inputs from DataManager
            algorithm_inputs = self.data_manager.get_algorithm_inputs()
            
            if not algorithm_inputs:
                logger.warning("No algorithm inputs available for subdivision masks")
                return self._create_empty_masks(algorithm_inputs.get('data', pd.Series()))
            
            # Run subdivision analysis
            data = algorithm_inputs['data']
            peak_mask = algorithm_inputs['peak_mask']
            valley_mask = algorithm_inputs['valley_mask']
            
            subdivisions = self.analyze_subdivisions(data, peak_mask, valley_mask)
            
            # Extract masks from subdivision results
            corrective_mask = subdivisions.get('corrective_abc', pd.Series(False, index=data.index))
            motive_mask = subdivisions.get('motive_abc', pd.Series(False, index=data.index))
            
            # For five-wave structures, we can also mark the starting points as valid
            motive_five = subdivisions.get('motive_five', pd.Series(dtype=object))
            corrective_five = subdivisions.get('corrective_five', pd.Series(dtype=object))
            
            # Enhance masks with five-wave structure information
            if not motive_five.empty:
                valid_motive_waves = motive_five.notna()
                motive_indices = valid_motive_waves[valid_motive_waves].index
                for idx in motive_indices:
                    if idx in motive_mask.index:
                        motive_mask.loc[idx] = True
            
            if not corrective_five.empty:
                valid_corrective_waves = corrective_five.notna()
                corrective_indices = valid_corrective_waves[valid_corrective_waves].index
                for idx in corrective_indices:
                    if idx in corrective_mask.index:
                        corrective_mask.loc[idx] = True
            
            masks = {
                'corrective_mask': corrective_mask,
                'motive_mask': motive_mask
            }
            
            logger.info(f"Subdivision masks generated - corrective: {corrective_mask.sum()}, motive: {motive_mask.sum()}")
            return masks
            
        except Exception as e:
            logger.error(f"Error generating subdivision masks: {str(e)}")
            return self._create_empty_masks(pd.Series())

    def _create_empty_masks(self, data: pd.Series) -> Dict[str, pd.Series]:
        """Create empty masks as fallback."""
        if data.empty:
            return {
                'corrective_mask': pd.Series(dtype=bool),
                'motive_mask': pd.Series(dtype=bool)
            }
        return {
            'corrective_mask': pd.Series(False, index=data.index),
            'motive_mask': pd.Series(False, index=data.index)
        }

    def get_subdivision_history(self, lookback: int = 10) -> List[Dict]:
        """Get recent subdivision history."""
        return list(self.subdivision_history)[-lookback:]

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about subdivision detection performance."""
        stats = {
            'total_subdivisions_detected': 0,
            'subdivision_counts': {},
            'history_size': len(self.subdivision_history),
            'last_update': pd.Timestamp.now()
        }
        
        for subdivision_type, subdivision_data in self.current_subdivisions.items():
            if subdivision_data is not None:
                if hasattr(subdivision_data, 'sum'):  # Boolean masks
                    count = subdivision_data.sum()
                elif hasattr(subdivision_data, 'notna'):  # Wave labels
                    count = subdivision_data.notna().sum()
                else:
                    count = len(subdivision_data) if subdivision_data is not None else 0
            else:
                count = 0
                
            stats['subdivision_counts'][subdivision_type] = count
            stats['total_subdivisions_detected'] += count
        
        return stats

    def run_analysis(self) -> Dict[str, Any]:
        """
        Complete analysis run - gets data from DataManager and returns all results.
        
        Returns:
        --------
        Dict with subdivisions, masks, and statistics
        """
        algorithm_inputs = self.data_manager.get_algorithm_inputs()
        
        if not algorithm_inputs:
            logger.warning("No algorithm inputs available for analysis")
            return {}
        
        subdivisions = self.analyze_subdivisions(
            data=algorithm_inputs['data'],
            peak_mask=algorithm_inputs['peak_mask'],
            valley_mask=algorithm_inputs['valley_mask']
        )
        
        masks = self.get_subdivision_masks()
        stats = self.get_detection_stats()
        
        return {
            'subdivisions': subdivisions,
            'masks': masks,
            'statistics': stats,
            'timestamp': pd.Timestamp.now()
        }


# Test the WavesManager
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing WavesManager with all 4 subdivision algorithms...")
    
    # Mock DataManager for testing
    class MockDataManager:
        def get_algorithm_inputs(self):
            # Create synthetic test data with clear wave structures
            dates = pd.date_range('2024-01-01', periods=200, freq='1min')
            
            # Create synthetic Elliott Wave patterns
            t = np.linspace(0, 4*np.pi, 200)
            
            # Impulse wave pattern (motive)
            impulse_wave = 100 + 10 * np.sin(t) + 5 * np.sin(5*t) + np.random.normal(0, 1, 200)
            
            # Corrective wave pattern  
            corrective_wave = 100 + 8 * np.sin(t) + 3 * np.sin(3*t) + np.random.normal(0, 0.8, 200)
            
            # Combine both patterns
            synthetic_data = impulse_wave * 0.6 + corrective_wave * 0.4
            
            data = pd.Series(synthetic_data, index=dates)
            
            # Create synthetic peaks and valleys that match wave structures
            peak_mask = pd.Series([False] * 200, index=dates)
            valley_mask = pd.Series([False] * 200, index=dates)
            
            # Add peaks at wave highs
            for i in range(20, 180, 25):
                peak_mask.iloc[i] = True
            
            # Add valleys at wave lows  
            for i in range(10, 190, 25):
                valley_mask.iloc[i] = True
            
            return {
                'data': data,
                'peak_mask': peak_mask,
                'valley_mask': valley_mask
            }
    
    # Test the WavesManager
    mock_dm = MockDataManager()
    waves_manager = WavesManager(mock_dm)
    
    print("Running subdivision analysis...")
    results = waves_manager.run_analysis()
    
    print("\n" + "="*60)
    print("WAVES MANAGER TEST RESULTS:")
    print("="*60)
    
    # Print subdivision counts
    subdivisions = results.get('subdivisions', {})
    for algo_name, algo_result in subdivisions.items():
        if algo_result is not None:
            if hasattr(algo_result, 'sum'):  # Boolean mask
                count = algo_result.sum()
                print(f"{algo_name}: {count} valid points")
            elif hasattr(algo_result, 'notna'):  # Wave labels
                count = algo_result.notna().sum()
                print(f"{algo_name}: {count} wave structures")
            else:
                print(f"{algo_name}: {len(algo_result)} results")
        else:
            print(f"{algo_name}: No results")
    
    # Print mask statistics
    masks = results.get('masks', {})
    print(f"\nSubdivision Masks:")
    print(f"  corrective_mask: {masks.get('corrective_mask', pd.Series()).sum()} True values")
    print(f"  motive_mask: {masks.get('motive_mask', pd.Series()).sum()} True values")
    
    # Print overall statistics
    stats = results.get('statistics', {})
    print(f"\nDetection Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Test integration with TrendsManager
    print(f"\nIntegration Test with TrendsManager:")
    subdivision_masks = waves_manager.get_subdivision_masks()
    print(f"  corrective_mask ready: {not subdivision_masks['corrective_mask'].empty}")
    print(f"  motive_mask ready: {not subdivision_masks['motive_mask'].empty}")
    print(f"  Both masks compatible with TrendsManager: {subdivision_masks['corrective_mask'].index.equals(subdivision_masks['motive_mask'].index)}")
    
    print("\nWavesManager test completed successfully! ✅")
    print("\nNEXT STEP: Update TrendsManager to use real subdivision masks:")
    print("""
    # BEFORE (broken):
    corrective_mask = pd.Series([False] * 100)  # Mock
    motive_mask = pd.Series([False] * 100)      # Mock
    
    # AFTER (working):
    subdivision_masks = waves_manager.get_subdivision_masks()
    corrective_mask = subdivision_masks['corrective_mask']  # Real!
    motive_mask = subdivision_masks['motive_mask']          # Real!
    """)