# algostack/waves_manager/waves_manager.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class WavesManager:
    """
    Unified manager for Elliott Wave subdivision detection.
    Now implements DFS-only system with sequential processing and superposition.
    Provides corrective_mask and motive_mask for TrendsManager integration.
    """

    def __init__(self, data_manager, history_size: int = 100, use_structural_turns: bool = True):
        """
        Initialize WavesManager with dependency injection.
        
        Parameters:
        -----------
        data_manager : DataManager
            The core data management component
        history_size : int
            Maximum number of subdivision results to keep in history
        use_structural_turns : bool
            Whether to use DFS structural turns instead of raw peaks/valleys
        """
        self.data_manager = data_manager
        self.subdivision_history = deque(maxlen=history_size)
        self.current_subdivisions = {}
        self.use_structural_turns = use_structural_turns
        
        # DFS state management
        self.last_processed_turn = None
        
        logger.info(f"WavesManager initialized with history size: {history_size}, use_structural_turns: {use_structural_turns}")

    def analyze_subdivisions_sequential(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> Dict[str, Any]:
        """
        DFS sequential processing - the MAIN method.
        Processes structural turns sequentially instead of batch processing.
        
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
            logger.info("Starting DFS sequential analysis")
            
            # CRITICAL FIX: Ensure we extract close prices from DataFrame BEFORE any processing
            if isinstance(data, pd.DataFrame):
                data = data['close']  # Extract close prices as Series
                logger.info(f"Extracted close prices from DataFrame - length: {len(data)}")
            
            # Ensure data is numeric and masks are proper boolean Series
            data = self._ensure_numeric_series(data)
            peak_mask = self._ensure_boolean_series(peak_mask, data.index)
            valley_mask = self._ensure_boolean_series(valley_mask, data.index)
            
            # Initialize empty results
            subdivisions = {
                'motive_abc': pd.Series(False, index=data.index),
                'corrective_abc': pd.Series(False, index=data.index),
                'motive_five': pd.Series(dtype=object, index=data.index),
                'corrective_five': pd.Series(dtype=object, index=data.index)
            }
            
            # Use DFS sequential processing
            subdivisions = self._process_structural_turns_sequentially(data, peak_mask, valley_mask)
            
            # Update current state and history
            self.current_subdivisions = subdivisions
            self._update_subdivision_history(subdivisions)
            
            # Count successful algorithms
            successful_algos = 0
            for s in subdivisions.values():
                if s is not None and not s.empty:
                    if hasattr(s, 'dtype') and s.dtype == object:
                        if s.notna().sum() > 0:
                            successful_algos += 1
                    else:
                        if s.sum() > 0:
                            successful_algos += 1
            
            logger.info(f"DFS analysis completed - found subdivisions across {successful_algos} algorithms")
            return subdivisions
            
        except Exception as e:
            logger.error(f"Error in analyze_subdivisions_sequential: {str(e)}")
            return {
                'motive_abc': pd.Series(False, index=data.index if not data.empty else []),
                'corrective_abc': pd.Series(False, index=data.index if not data.empty else []),
                'motive_five': pd.Series(dtype=object, index=data.index if not data.empty else []),
                'corrective_five': pd.Series(dtype=object, index=data.index if not data.empty else [])
            }

    def _process_structural_turns_sequentially(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series) -> Dict[str, Any]:
        """
        Process structural turns sequentially using DFS.
        This is the core of the system.
        """
        logger.info("Starting sequential structural turn processing")
        
        # Initialize results
        results = {
            'motive_abc': pd.Series(False, index=data.index),
            'corrective_abc': pd.Series(False, index=data.index),
            'motive_five': pd.Series(dtype=object, index=data.index),
            'corrective_five': pd.Series(dtype=object, index=data.index)
        }
        
        # Get raw peaks and valleys for algorithms to use internally
        raw_peaks = data.index[peak_mask].tolist()
        raw_valleys = data.index[valley_mask].tolist()
        
        logger.info(f"Sequential processing with {len(raw_peaks)} raw peaks and {len(raw_valleys)} raw valleys")
        
        # Combine all raw peaks and valleys, sort chronologically
        # all_raw_turns = sorted(raw_peaks + raw_valleys)
        # logger.info(f"Total raw turns available: {len(all_raw_turns)}")
        # Get ONLY quality starting points (structural turns, not noise)
        quality_starts = self.data_manager.get_quality_starting_points(min_price_move=0.002)
        all_raw_turns = quality_starts  # Use quality starts instead of raw turns
        logger.info(f"Quality starting points available: {len(all_raw_turns)} (vs {len(raw_peaks) + len(raw_valleys)} raw turns)")
        
        # Process raw turns sequentially
        current_timestamp = self.last_processed_turn
        processed_turns = 0
        max_turns_to_process = min(200, len(all_raw_turns))  # Process up to 500 raw turns currently (NOW USES 200 MAX POINTS)
        
        # Find starting index based on last processed turn
        start_index = 0
        if current_timestamp:
            # Find the first turn after our last processed timestamp
            for i, turn_ts in enumerate(all_raw_turns):
                if turn_ts > current_timestamp:
                    start_index = i
                    break
        
        logger.info(f"Starting from index {start_index} of {len(all_raw_turns)} raw turns")
        
        for i in range(start_index, len(all_raw_turns)):
            if processed_turns >= max_turns_to_process:
                logger.info(f"Reached maximum turns to process: {max_turns_to_process}")
                break
                
            turn_timestamp = all_raw_turns[i]
            
            # Determine if this is a peak or valley
            is_peak = turn_timestamp in raw_peaks
            is_valley = turn_timestamp in raw_valleys
            turn_type = 'peak' if is_peak else 'valley' if is_valley else 'unknown'
            
            # Get price for logging
            try:
                turn_price = float(data.loc[turn_timestamp])
            except:
                turn_price = 0.0
            
            logger.info(f"Processing raw {turn_type} at {turn_timestamp} @ {turn_price:.2f}")
            
            if turn_type == 'valley':
                # SUPERPOSITION: Try both motive patterns from this valley
                self._build_motive_patterns_from_valley(turn_timestamp, raw_peaks, raw_valleys, data, results)
            elif turn_type == 'peak':
                # SUPERPOSITION: Try both corrective patterns from this peak
                self._build_corrective_patterns_from_peak(turn_timestamp, raw_peaks, raw_valleys, data, results)
            
            # Update for next iteration
            self.last_processed_turn = turn_timestamp
            processed_turns += 1
            
            logger.info(f"Completed processing turn {processed_turns}: {turn_type} at {turn_timestamp}")
        
        logger.info(f"Sequential processing completed - processed {processed_turns} raw turns out of {len(all_raw_turns)} available")
        return results

    def _build_motive_patterns_from_valley(self, start_valley: pd.Timestamp, raw_peaks: List, raw_valleys: List, 
                                         data: pd.Series, results: Dict[str, Any]):
        """
        Superposition logic for motive patterns from a structural valley.
        Tries both ABC and Five-wave patterns, resolves based on price action.
        """
        logger.info(f"Building motive patterns from valley at {start_valley}")
        
        # SUPERPOSITION: Try both pattern types in parallel
        abc_pattern = self._motive_abc(start_valley, raw_peaks, raw_valleys, data)
        five_wave_pattern = self._motive_five(start_valley, raw_peaks, raw_valleys, data)
        
        # FIXED: Five-wave patterns get priority
        if five_wave_pattern and five_wave_pattern['is_complete']:
            # Five-wave pattern confirmed (takes priority)
            self._label_five_wave_pattern(five_wave_pattern, results, 'motive_five')
            logger.info(f"Labelled motive_five pattern starting at {start_valley}")
            logger.info(f"Superposition RESOLVED: Five-wave motive pattern at {start_valley}")
        elif abc_pattern and abc_pattern['is_complete']:
            # ABC pattern confirmed (fallback)
            self._label_abc_pattern(abc_pattern, results, 'motive_abc')
            logger.info(f"Superposition RESOLVED: ABC motive pattern at {start_valley}")
        else:
            # Pattern incomplete or invalid
            logger.debug(f"Superposition: No valid motive pattern at {start_valley}")

    def _build_corrective_patterns_from_peak(self, start_peak: pd.Timestamp, raw_peaks: List, raw_valleys: List, 
                                           data: pd.Series, results: Dict[str, Any]):
        """
        Superposition logic for corrective patterns from a structural peak.
        Tries both ABC and Five-wave patterns, resolves based on price action.
        """
        logger.info(f"Building corrective patterns from peak at {start_peak}")
        
        # SUPERPOSITION: Try both pattern types in parallel
        abc_pattern = self._corrective_abc(start_peak, raw_peaks, raw_valleys, data)
        five_wave_pattern = self._corrective_five(start_peak, raw_peaks, raw_valleys, data)
        
        # FIXED: Five-wave patterns get priority
        if five_wave_pattern and five_wave_pattern['is_complete']:
            # Five-wave pattern confirmed (takes priority)
            self._label_five_wave_pattern(five_wave_pattern, results, 'corrective_five')
            logger.info(f"Labelled corrective_five pattern starting at {start_peak}")
            logger.info(f"Superposition RESOLVED: Five-wave corrective pattern at {start_peak}")
        elif abc_pattern and abc_pattern['is_complete']:
            # ABC pattern confirmed (fallback)
            self._label_abc_pattern(abc_pattern, results, 'corrective_abc')
            logger.info(f"Superposition RESOLVED: ABC corrective pattern at {start_peak}")
        else:
            # Pattern incomplete or invalid
            logger.debug(f"Superposition: No valid corrective pattern at {start_peak}")

    def _motive_five(self, start_valley: pd.Timestamp, raw_peaks: List, raw_valleys: List, 
                             data: pd.Series) -> Optional[Dict]:
        """
        Instrumented five-wave builder – returns None **and** prints the exact leg that failed.
        """
        logger.info(f"🛠️  MOTIVE-5 START  {start_valley}")
        try:
            # --------- wave 1 ---------
            w1 = self.data_manager.find_wave_end_by_tiredness(start_valley, 'up', 'wave1')
            if w1 is None:
                logger.warning("❌  wave-1  – no structural peak after valley")
                return None
            p1_price = float(data.loc[w1])
            logger.info(f"wave-1  peak  {w1}  @ {p1_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 1 (max 2.0 hours)
            time_gap_w1 = (w1 - start_valley).total_seconds() / 3600  # hours
            if time_gap_w1 > 2.0:
                logger.warning(f"❌  wave-1  – too long: {time_gap_w1:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave 1 duration: {time_gap_w1:.2f}h")

            # --------- wave 2 ---------
            w2 = self.data_manager.find_wave_end_by_tiredness(w1, 'down', 'wave2')
            if w2 is None:
                logger.warning("❌  wave-2  – no structural valley after wave-1")
                return None
            v1_price = float(data.loc[w2])
            logger.info(f"wave-2  valley  {w2}  @ {v1_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 2 (max 1.5 hours)
            time_gap_w2 = (w2 - w1).total_seconds() / 3600
            if time_gap_w2 > 1.5:
                logger.warning(f"❌  wave-2  – too long: {time_gap_w2:.1f}h > 1.5h")
                return None
            logger.debug(f"Wave 2 duration: {time_gap_w2:.2f}h")

            # --------- wave 3 ---------
            w3 = self.data_manager.find_wave_end_by_tiredness(w2, 'up', 'wave3')
            if w3 is None:
                logger.warning("❌  wave-3  – no structural peak ≥ 0.01 % after wave-2")
                return None
            p2_price = float(data.loc[w3])
            logger.info(f"wave-3  peak  {w3}  @ {p2_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 3 (max 2.5 hours)
            time_gap_w3 = (w3 - w2).total_seconds() / 3600
            if time_gap_w3 > 2.5:
                logger.warning(f"❌  wave-3  – too long: {time_gap_w3:.1f}h > 2.5h")
                return None
            logger.debug(f"Wave 3 duration: {time_gap_w3:.2f}h")

            # --------- wave 4 ---------
            w4 = self.data_manager.find_wave_end_by_tiredness(w3, 'down', 'wave4')
            if w4 is None:
                logger.warning("❌  wave-4  – no structural valley after wave-3")
                return None
            v2_price = float(data.loc[w4])
            logger.info(f"wave-4  valley  {w4}  @ {v2_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 4 (max 1.5 hours)
            time_gap_w4 = (w4 - w3).total_seconds() / 3600
            if time_gap_w4 > 1.5:
                logger.warning(f"❌  wave-4  – too long: {time_gap_w4:.1f}h > 1.5h")
                return None
            logger.debug(f"Wave 4 duration: {time_gap_w4:.2f}h")

            # --------- wave 5 ---------
            w5 = self.data_manager.find_wave_end_by_tiredness(w4, 'up', 'wave5')
            if w5 is None:
                logger.warning("❌  wave-5  – no structural peak after wave-4")
                return None
            p3_price = float(data.loc[w5])
            logger.info(f"wave-5  peak  {w5}  @ {p3_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 5 (max 2.0 hours)
            time_gap_w5 = (w5 - w4).total_seconds() / 3600
            if time_gap_w5 > 2.0:
                logger.warning(f"❌  wave-5  – too long: {time_gap_w5:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave 5 duration: {time_gap_w5:.2f}h")

            # --------- validator ---------
            if self._validate_motive_five_waves(start_valley, w1, w2, w3, w4, w5, data):
                logger.info("✅  MOTIVE-5  COMPLETE")
                return {
                    'is_complete': True,
                    'waves': {
                        '1': {'start': start_valley, 'end': w1, 'type': 'valley_to_peak'},
                        '2': {'start': w1, 'end': w2, 'type': 'peak_to_valley'},
                        '3': {'start': w2, 'end': w3, 'type': 'valley_to_peak'},
                        '4': {'start': w3, 'end': w4, 'type': 'peak_to_valley'},
                        '5': {'start': w4, 'end': w5, 'type': 'valley_to_peak'}
                    }
                }
            else:
                logger.warning("❌  validator  – Elliott rules failed")
                return None

        except Exception as e:
            logger.error(f"Crash in motive-5 builder: {e}")
            return None

    def _motive_abc(self, start_valley: pd.Timestamp, raw_peaks: List, raw_valleys: List, 
                            data: pd.Series) -> Optional[Dict]:
        """
        Build motive ABC pattern using DFS processing.
        """
        try:
            logger.debug(f"Trying to build motive ABC from {start_valley}")
            
            # Use DFS to find wave A peak
            waveA_peak = self.data_manager.find_wave_end_by_tiredness(
                start_valley, 'up', 'waveA'
            )
            if not waveA_peak:
                return None
                
            # ✅ NEW: Time gap check for wave A (max 2.0 hours)
            time_gap_A = (waveA_peak - start_valley).total_seconds() / 3600  # hours
            if time_gap_A > 2.0:
                logger.debug(f"❌  wave-A  – too long: {time_gap_A:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave A duration: {time_gap_A:.2f}h")
                
            # Use DFS to find wave B valley
            waveB_valley = self.data_manager.find_wave_end_by_tiredness(
                waveA_peak, 'down', 'waveB'
            )
            if not waveB_valley:
                return None
                
            # ✅ NEW: Time gap check for wave B (max 1.5 hours)
            time_gap_B = (waveB_valley - waveA_peak).total_seconds() / 3600
            if time_gap_B > 1.5:
                logger.debug(f"❌  wave-B  – too long: {time_gap_B:.1f}h > 1.5h")
                return None
            logger.debug(f"Wave B duration: {time_gap_B:.2f}h")
                
            # Use DFS to find wave C peak
            waveC_peak = self.data_manager.find_wave_end_by_tiredness(
                waveB_valley, 'up', 'waveC'
            )
            if not waveC_peak:
                return None
                
            # ✅ NEW: Time gap check for wave C (max 2.0 hours)
            time_gap_C = (waveC_peak - waveB_valley).total_seconds() / 3600
            if time_gap_C > 2.0:
                logger.debug(f"❌  wave-C  – too long: {time_gap_C:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave C duration: {time_gap_C:.2f}h")
            
            # Basic ABC validation
            price_A = data.loc[waveA_peak]
            price_B = data.loc[waveB_valley] 
            price_C = data.loc[waveC_peak]
            
            if price_C > price_A:  # Wave C should be higher than Wave A
                return {
                    'is_complete': True,
                    'waves': {
                        'A': {'start': start_valley, 'end': waveA_peak, 'type': 'valley_to_peak'},
                        'B': {'start': waveA_peak, 'end': waveB_valley, 'type': 'peak_to_valley'},
                        'C': {'start': waveB_valley, 'end': waveC_peak, 'type': 'valley_to_peak'}
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error building motive ABC: {e}")
            return None

    def _corrective_five(self, start_peak: pd.Timestamp, raw_peaks: List, raw_valleys: List, 
                                 data: pd.Series) -> Optional[Dict]:
        """
        Instrumented corrective five-wave builder – returns None **and** prints the exact leg that failed.
        """
        logger.info(f"🛠️  CORRECTIVE-5 START  {start_peak}")
        try:
            # --------- wave 1 ---------
            w1 = self.data_manager.find_wave_end_by_tiredness(start_peak, 'down', 'wave1')
            if w1 is None:
                logger.warning("❌  wave-1  – no structural valley after peak")
                return None
            v1_price = float(data.loc[w1])
            logger.info(f"wave-1  valley  {w1}  @ {v1_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 1 (max 2.0 hours)
            time_gap_w1 = (w1 - start_peak).total_seconds() / 3600  # hours
            if time_gap_w1 > 2.0:
                logger.warning(f"❌  wave-1  – too long: {time_gap_w1:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave 1 duration: {time_gap_w1:.2f}h")

            # --------- wave 2 ---------
            w2 = self.data_manager.find_wave_end_by_tiredness(w1, 'up', 'wave2')
            if w2 is None:
                logger.warning("❌  wave-2  – no structural peak after wave-1")
                return None
            p1_price = float(data.loc[w2])
            logger.info(f"wave-2  peak  {w2}  @ {p1_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 2 (max 1.5 hours)
            time_gap_w2 = (w2 - w1).total_seconds() / 3600
            if time_gap_w2 > 1.5:
                logger.warning(f"❌  wave-2  – too long: {time_gap_w2:.1f}h > 1.5h")
                return None
            logger.debug(f"Wave 2 duration: {time_gap_w2:.2f}h")

            # --------- wave 3 ---------
            w3 = self.data_manager.find_wave_end_by_tiredness(w2, 'down', 'wave3')
            if w3 is None:
                logger.warning("❌  wave-3  – no structural valley ≥ 0.01 % after wave-2")
                return None
            v2_price = float(data.loc[w3])
            logger.info(f"wave-3  valley  {w3}  @ {v2_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 3 (max 2.5 hours)
            time_gap_w3 = (w3 - w2).total_seconds() / 3600
            if time_gap_w3 > 2.5:
                logger.warning(f"❌  wave-3  – too long: {time_gap_w3:.1f}h > 2.5h")
                return None
            logger.debug(f"Wave 3 duration: {time_gap_w3:.2f}h")

            # --------- wave 4 ---------
            w4 = self.data_manager.find_wave_end_by_tiredness(w3, 'up', 'wave4')
            if w4 is None:
                logger.warning("❌  wave-4  – no structural peak after wave-3")
                return None
            p2_price = float(data.loc[w4])
            logger.info(f"wave-4  peak  {w4}  @ {p2_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 4 (max 1.5 hours)
            time_gap_w4 = (w4 - w3).total_seconds() / 3600
            if time_gap_w4 > 1.5:
                logger.warning(f"❌  wave-4  – too long: {time_gap_w4:.1f}h > 1.5h")
                return None
            logger.debug(f"Wave 4 duration: {time_gap_w4:.2f}h")

            # --------- wave 5 ---------
            w5 = self.data_manager.find_wave_end_by_tiredness(w4, 'down', 'wave5')
            if w5 is None:
                logger.warning("❌  wave-5  – no structural valley after wave-4")
                return None
            v3_price = float(data.loc[w5])
            logger.info(f"wave-5  valley  {w5}  @ {v3_price:.2f}")
            
            # ✅ NEW: Time gap check for wave 5 (max 2.0 hours)
            time_gap_w5 = (w5 - w4).total_seconds() / 3600
            if time_gap_w5 > 2.0:
                logger.warning(f"❌  wave-5  – too long: {time_gap_w5:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave 5 duration: {time_gap_w5:.2f}h")

            # --------- validator ---------
            if self._validate_corrective_five_waves(start_peak, w1, w2, w3, w4, w5, data):
                logger.info("✅  CORRECTIVE-5  COMPLETE")
                return {
                    'is_complete': True,
                    'waves': {
                        '1': {'start': start_peak, 'end': w1, 'type': 'peak_to_valley'},
                        '2': {'start': w1, 'end': w2, 'type': 'valley_to_peak'},
                        '3': {'start': w2, 'end': w3, 'type': 'peak_to_valley'},
                        '4': {'start': w3, 'end': w4, 'type': 'valley_to_peak'},
                        '5': {'start': w4, 'end': w5, 'type': 'peak_to_valley'}
                    }
                }
            else:
                logger.warning("❌  validator  – Elliott rules failed")
                return None

        except Exception as e:
            logger.error(f"Crash in corrective-5 builder: {e}")
            return None

    def _corrective_abc(self, start_peak: pd.Timestamp, raw_peaks: List, raw_valleys: List, 
                                data: pd.Series) -> Optional[Dict]:
        """
        Build corrective ABC pattern using DFS processing.
        """
        try:
            logger.debug(f"Trying to build corrective ABC from {start_peak}")
            
            # Use DFS to find wave A valley
            waveA_valley = self.data_manager.find_wave_end_by_tiredness(
                start_peak, 'down', 'waveA'
            )
            if not waveA_valley:
                return None
                
            # ✅ NEW: Time gap check for wave A (max 2.0 hours)
            time_gap_A = (waveA_valley - start_peak).total_seconds() / 3600  # hours
            if time_gap_A > 2.0:
                logger.debug(f"❌  wave-A  – too long: {time_gap_A:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave A duration: {time_gap_A:.2f}h")
                
            # Use DFS to find wave B peak
            waveB_peak = self.data_manager.find_wave_end_by_tiredness(
                waveA_valley, 'up', 'waveB'
            )
            if not waveB_peak:
                return None
                
            # ✅ NEW: Time gap check for wave B (max 1.5 hours)
            time_gap_B = (waveB_peak - waveA_valley).total_seconds() / 3600
            if time_gap_B > 1.5:
                logger.debug(f"❌  wave-B  – too long: {time_gap_B:.1f}h > 1.5h")
                return None
            logger.debug(f"Wave B duration: {time_gap_B:.2f}h")
                
            # Use DFS to find wave C valley
            waveC_valley = self.data_manager.find_wave_end_by_tiredness(
                waveB_peak, 'down', 'waveC'
            )
            if not waveC_valley:
                return None
                
            # ✅ NEW: Time gap check for wave C (max 2.0 hours)
            time_gap_C = (waveC_valley - waveB_peak).total_seconds() / 3600
            if time_gap_C > 2.0:
                logger.debug(f"❌  wave-C  – too long: {time_gap_C:.1f}h > 2.0h")
                return None
            logger.debug(f"Wave C duration: {time_gap_C:.2f}h")
            
            # Basic ABC validation for corrective
            price_A = data.loc[waveA_valley]
            price_B = data.loc[waveB_peak]
            price_C = data.loc[waveC_valley]
            
            if price_C < price_A:  # Wave C should be lower than Wave A
                return {
                    'is_complete': True,
                    'waves': {
                        'A': {'start': start_peak, 'end': waveA_valley, 'type': 'peak_to_valley'},
                        'B': {'start': waveA_valley, 'end': waveB_peak, 'type': 'valley_to_peak'},
                        'C': {'start': waveB_peak, 'end': waveC_valley, 'type': 'peak_to_valley'}
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error building corrective ABC: {e}")
            return None

    def _validate_motive_five_waves(self, v0, p1, v1, p2, v2, p3, data):
        """Validate motive five-wave pattern using Elliott Wave rules."""
        try:
            v0_price, p1_price = float(data.loc[v0]), float(data.loc[p1])
            v1_price, p2_price = float(data.loc[v1]), float(data.loc[p2])
            v2_price, p3_price = float(data.loc[v2]), float(data.loc[p3])
            
            w1_len = p1_price - v0_price
            w3_len = p2_price - v1_price
            min_wave4_floor = v1_price + 0.0005 * v1_price   # 0.05 % buffer above wave-1
            
            # ---------- motive five validator – instrumented ----------
            logger.debug(f"VALIDATOR-CHK  p1≥v0 {p1_price>=v0_price}  v1<p1&v1>v0 {v1_price<p1_price and v1_price>v0_price}  "
                         f"p2≥p1+0.005% {p2_price>=p1_price*1.00005}  w3>w1*0.05% {w3_len>w1_len*0.0005}  "
                         f"v2<p2&v2≥0.75v1 {v2_price<p2_price and v2_price>=v1_price*0.75}  p3>p2 {p3_price>p2_price}  "
                         f"p2≥minWave4 {p2_price>=min_wave4_floor}")

            if not (p1_price >= v0_price):            
                logger.warning("❌  w1 start < w0 start (wave-1 not rising)");            
                return False
            if not (v1_price < p1_price and v1_price > v0_price):
                logger.warning("❌  w2 violates retracement (not between w0 and w1)"); 
                return False
            if not (p2_price >= p1_price * 1.00005):  # FIXED: 0.005% above p1
                logger.warning("❌  w3 top < w1 top + 0.005 %"); 
                return False
            if not (w3_len > w1_len * 0.0005):       
                logger.warning(f"❌  w3 run too short (w3_len={w3_len:.2f}, min={w1_len*0.0005:.2f})");      
                return False
            if not (v2_price < p2_price and v2_price >= v1_price * 0.75):
                logger.warning(f"❌  w4 out of range (v2={v2_price:.2f}, p2={p2_price:.2f}, 75%v1={v1_price*0.75:.2f})");        
                return False
            if not (p3_price > p2_price):            
                logger.warning(f"❌  w5 not new high (p3={p3_price:.2f}, p2={p2_price:.2f})");        
                return False
            if not (p2_price >= min_wave4_floor):    
                logger.warning(f"❌  w4 enters w1 territory (p2={p2_price:.2f}, min_floor={min_wave4_floor:.2f})"); 
                return False

            logger.info("✅  validator  – all rules pass")
            return True
            
        except Exception as e:
            logger.error(f"Error in validator: {e}")
            return False

    def _validate_corrective_five_waves(self, p0, v1, p1, v2, p2, v3, data):
        """Validate corrective five-wave pattern using Elliott Wave rules."""
        try:
            p0_price, v1_price = float(data.loc[p0]), float(data.loc[v1])
            p1_price, v2_price = float(data.loc[p1]), float(data.loc[v2])
            p2_price, v3_price = float(data.loc[p2]), float(data.loc[v3])
            
            w1_len = p0_price - v1_price
            w3_len = p1_price - v2_price
            max_wave4_ceiling = p1_price - 0.0005 * p1_price   # 0.05 % buffer below wave-1
            
            # ---------- corrective five validator – instrumented ----------
            logger.debug(f"VALIDATOR-CHK  v1≤p0 {v1_price<=p0_price}  p1>v1&p1<p0 {p1_price>v1_price and p1_price<p0_price}  "
                         f"v2<v1-0.5% {v2_price<=v1_price*0.995}  w3>w1*0.05% {w3_len>w1_len*0.0005}  "
                         f"p2>v2&p2≤1.15v1 {p2_price>v2_price and p2_price<=v1_price*1.15}  "
                         f"v3<v2 {v3_price<v2_price}  v2≤maxWave4 {v2_price<=max_wave4_ceiling}")

            if not (v1_price <= p0_price):            
                logger.warning("❌  w1 not falling (v1 > p0)");            
                return False
            if not (p1_price > v1_price and p1_price < p0_price):
                logger.warning("❌  w2 violates retracement (not between v1 and p0)"); 
                return False
            if not (v2_price <= v1_price * 0.995):  # FIXED: 0.5% BELOW v1
                logger.warning("❌  w3 not making new low (v2 >= 99.5% of v1)"); 
                return False
            if not (w3_len > w1_len * 0.0005):       
                logger.warning(f"❌  w3 run too short (w3_len={w3_len:.2f}, min={w1_len*0.0005:.2f})");      
                return False
            if not (p2_price > v2_price and p2_price <= v1_price * 1.15):
                logger.warning(f"❌  w4 out of range (p2={p2_price:.2f}, v2={v2_price:.2f}, 115%v1={v1_price*1.15:.2f})");        
                return False
            if not (v3_price < v2_price):            
                logger.warning(f"❌  w5 not new low (v3={v3_price:.2f}, v2={v2_price:.2f})");        
                return False
            if not (v2_price <= max_wave4_ceiling):    
                logger.warning(f"❌  w4 enters w1 territory (v2={v2_price:.2f}, max_ceiling={max_wave4_ceiling:.2f})"); 
                return False

            logger.info("✅  validator  – all rules pass")
            return True
            
        except Exception as e:
            logger.error(f"Error in validator: {e}")
            return False

    def _label_five_wave_pattern(self, pattern: Dict, results: Dict[str, Any], result_key: str):
        """Label a complete five-wave pattern in the results."""
        waves = pattern['waves']
        for wave_num, wave_info in waves.items():
            if wave_info['type'] in ['valley_to_peak', 'peak_to_valley']:
                # For five-wave patterns, we label the start points
                start_point = wave_info['start']
                if start_point in results[result_key].index:
                    results[result_key].loc[start_point] = wave_num

    def _label_abc_pattern(self, pattern: Dict, results: Dict[str, Any], result_key: str):
        """Label a complete ABC pattern in the results."""
        waves = pattern['waves']
        for wave_letter, wave_info in waves.items():
            if wave_letter == 'B' and result_key == 'motive_abc':
                # For motive ABC, mark the B valley
                if wave_info['end'] in results[result_key].index:
                    results[result_key].loc[wave_info['end']] = True
            elif wave_letter == 'A' and result_key == 'corrective_abc':
                # For corrective ABC, mark the A valley  
                if wave_info['end'] in results[result_key].index:
                    results[result_key].loc[wave_info['end']] = True

    def _ensure_numeric_series(self, series: pd.Series) -> pd.Series:
        """Ensure the series is numeric by converting to float."""
        if not isinstance(series, pd.Series):
            return pd.Series(dtype=float)
        
        # Convert to numeric, coercing errors to NaN
        if series.dtype not in [np.float64, np.int64]:
            try:
                series = pd.to_numeric(series, errors='coerce')
                # Fill NaN values with forward fill then backward fill
                series = series.ffill().bfill()
                logger.debug(f"Converted series to numeric type: {series.dtype}")
            except Exception as e:
                logger.error(f"Error converting series to numeric: {e}")
                return pd.Series(dtype=float)
        
        return series

    def _ensure_boolean_series(self, series: pd.Series, index: pd.Index) -> pd.Series:
        """Ensure the series is a proper boolean series with correct index."""
        if not isinstance(series, pd.Series):
            return pd.Series(False, index=index)
        
        # Convert to boolean if not already
        if series.dtype != bool:
            try:
                series = series.astype(bool)
            except:
                series = pd.Series(False, index=index)
        
        # CRITICAL FIX: Ensure the series has the same index as data
        if not series.index.equals(index):
            logger.warning(f"Reindexing boolean series to match data index: {len(series)} -> {len(index)}")
            series = series.reindex(index, fill_value=False)
            
        return series

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
            
            # Use explicit check for empty dictionary
            if not algorithm_inputs or algorithm_inputs is None:
                logger.warning("No algorithm inputs available for subdivision masks")
                empty_data = pd.Series(dtype=float)
                return self._create_empty_masks(empty_data)
            
            # Use .get() with explicit None checks
            data = algorithm_inputs.get('data')
            peak_mask = algorithm_inputs.get('peak_mask')
            valley_mask = algorithm_inputs.get('valley_mask')
            
            # Explicit None checks for each required input
            if data is None or peak_mask is None or valley_mask is None:
                logger.warning("Missing required inputs in algorithm_inputs")
                return self._create_empty_masks(data if data is not None else pd.Series(dtype=float))
            
            # Use DFS sequential processing
            subdivisions = self.analyze_subdivisions_sequential(data, peak_mask, valley_mask)
            
            # FIXED: Proper None handling without ambiguous boolean evaluation
            corrective_mask = subdivisions.get('corrective_abc')
            motive_mask = subdivisions.get('motive_abc')
            
            if corrective_mask is None:
                corrective_mask = pd.Series(False, index=data.index)
            if motive_mask is None:
                motive_mask = pd.Series(False, index=data.index)
            
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
            'last_update': pd.Timestamp.now(),
            'using_structural_turns': self.use_structural_turns,
            'wave_aware_dfs_available': hasattr(self.data_manager, 'get_structural_turn_for_wave'),
            'dfs_enabled': True,
            'last_processed_turn': self.last_processed_turn
        }
        
        # Bullet-proof the statistics counter with crash-safe version
        for subdivision_type, subdivision_data in self.current_subdivisions.items():
            if subdivision_data is not None and not subdivision_data.empty:
                try:
                    if subdivision_data.dtype == object:          # string wave labels
                        count = subdivision_data.notna().sum()
                    else:                                         # numeric/bool mask
                        count = subdivision_data.sum()
                except TypeError:                                 # fallback if .sum() still fails
                    count = subdivision_data.notna().sum()
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
        
        # Use explicit check for empty or None
        if not algorithm_inputs or algorithm_inputs is None:
            logger.warning("No algorithm inputs available for analysis")
            return {}
        
        data = algorithm_inputs.get('data')
        peak_mask = algorithm_inputs.get('peak_mask')
        valley_mask = algorithm_inputs.get('valley_mask')
        
        # Explicit None checks
        if data is None or peak_mask is None or valley_mask is None:
            logger.warning("Missing required inputs in algorithm_inputs")
            return {}
        
        # Use DFS sequential processing
        subdivisions = self.analyze_subdivisions_sequential(data, peak_mask, valley_mask)
        masks = self.get_subdivision_masks()
        stats = self.get_detection_stats()
        
        return {
            'subdivisions': subdivisions,
            'masks': masks,
            'statistics': stats,
            'timestamp': pd.Timestamp.now()
        }

    def create_enhanced_plot(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series,
                           subdivisions: Dict[str, Any], max_points: int = 500) -> None:
        """Create enhanced plot with 4-panel visualization"""
        try:
            # Sample data if too large
            if len(data) > max_points:
                step = len(data) // max_points
                data = data.iloc[::step]
                peak_mask = peak_mask.iloc[::step]
                valley_mask = valley_mask.iloc[::step]
                logger.info(f"Sampled data to {len(data)} points for plotting")

            # Create 4-panel plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Elliott Wave Subdivision Analysis - DFS Engine', fontsize=16, fontweight='bold')

            # Panel 1: Price data with peaks and valleys
            ax1.plot(data.index, data.values, color='black', linewidth=1, label='Price')
            ax1.scatter(data.index[peak_mask], data[peak_mask], color='red', s=30, label='Peaks', alpha=0.7)
            ax1.scatter(data.index[valley_mask], data[valley_mask], color='blue', s=30, label='Valleys', alpha=0.7)
            ax1.set_title('Price Data with Peaks/Valleys')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Panel 2: Motive patterns
            ax2.plot(data.index, data.values, color='gray', linewidth=0.5, alpha=0.7)
            motive_abc_mask = subdivisions.get('motive_abc', pd.Series(False, index=data.index))
            motive_five = subdivisions.get('motive_five', pd.Series(dtype=object))
            
            if motive_abc_mask.any():
                ax2.scatter(data.index[motive_abc_mask], data[motive_abc_mask], 
                          color='green', s=50, label='Motive ABC (B waves)', alpha=0.8)
            
            if not motive_five.empty and motive_five.notna().any():
                valid_waves = motive_five[motive_five.notna()]
                colors = {'1': 'red', '2': 'blue', '3': 'green', '4': 'orange', '5': 'purple'}
                for wave_idx, wave_label in valid_waves.items():
                    if wave_idx in data.index:
                        ax2.scatter(wave_idx, data.loc[wave_idx], 
                                  color=colors.get(wave_label, 'black'), s=60, 
                                  label=f'Wave {wave_label}', alpha=0.8)
            
            ax2.set_title('Motive Patterns (ABC & Five-Wave)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Panel 3: Corrective patterns
            ax3.plot(data.index, data.values, color='gray', linewidth=0.5, alpha=0.7)
            corrective_abc_mask = subdivisions.get('corrective_abc', pd.Series(False, index=data.index))
            corrective_five = subdivisions.get('corrective_five', pd.Series(dtype=object))
            
            if corrective_abc_mask.any():
                ax3.scatter(data.index[corrective_abc_mask], data[corrective_abc_mask], 
                          color='orange', s=50, label='Corrective ABC (A waves)', alpha=0.8)
            
            if not corrective_five.empty and corrective_five.notna().any():
                valid_waves = corrective_five[corrective_five.notna()]
                colors = {'1': 'red', '2': 'blue', '3': 'green', '4': 'orange', '5': 'purple'}
                for wave_idx, wave_label in valid_waves.items():
                    if wave_idx in data.index:
                        ax3.scatter(wave_idx, data.loc[wave_idx], 
                                  color=colors.get(wave_label, 'black'), s=60, 
                                  label=f'Wave {wave_label}', alpha=0.8)
            
            ax3.set_title('Corrective Patterns (ABC & Five-Wave)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Panel 4: Combined overview
            ax4.plot(data.index, data.values, color='black', linewidth=1, label='Price')
            
            # Plot all detected patterns
            if motive_abc_mask.any():
                ax4.scatter(data.index[motive_abc_mask], data[motive_abc_mask], 
                          color='green', s=40, label='Motive ABC', alpha=0.7)
            if corrective_abc_mask.any():
                ax4.scatter(data.index[corrective_abc_mask], data[corrective_abc_mask], 
                          color='orange', s=40, label='Corrective ABC', alpha=0.7)
            if not motive_five.empty and motive_five.notna().any():
                valid_waves = motive_five[motive_five.notna()]
                for wave_idx, wave_label in valid_waves.items():
                    if wave_idx in data.index:
                        ax4.scatter(wave_idx, data.loc[wave_idx], 
                                  color='blue', s=60, label=f'Motive 5-{wave_label}', alpha=0.8)
            if not corrective_five.empty and corrective_five.notna().any():
                valid_waves = corrective_five[corrective_five.notna()]
                for wave_idx, wave_label in valid_waves.items():
                    if wave_idx in data.index:
                        ax4.scatter(wave_idx, data.loc[wave_idx], 
                                  color='red', s=60, label=f'Corrective 5-{wave_label}', alpha=0.8)
            
            ax4.set_title('Combined Overview - All Detected Patterns')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('waves.png', dpi=150)
            print('Chart saved -> waves.png')
            
        except Exception as e:
            logger.error(f"Error creating enhanced plot: {e}")
            # Create simple fallback plot
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data.values, color='black', linewidth=1)
            plt.title('Price Data (Fallback Plot)')
            plt.grid(True, alpha=0.3)
            plt.savefig('waves.png', dpi=150)
            print('Fallback chart saved -> waves.png')

    def create_detailed_plot(self, data: pd.Series, peak_mask: pd.Series, valley_mask: pd.Series,
                           subdivisions: Dict[str, Any], max_points: int = 500) -> None:
        """Create detailed single-panel plot"""
        try:
            # Sample data if too large
            if len(data) > max_points:
                step = len(data) // max_points
                data = data.iloc[::step]
                peak_mask = peak_mask.iloc[::step]
                valley_mask = valley_mask.iloc[::step]

            plt.figure(figsize=(15, 8))
            
            # Plot price data
            plt.plot(data.index, data.values, color='black', linewidth=1, label='Price', alpha=0.8)
            
            # Plot peaks and valleys
            plt.scatter(data.index[peak_mask], data[peak_mask], color='red', s=30, label='Peaks', alpha=0.6)
            plt.scatter(data.index[valley_mask], data[valley_mask], color='blue', s=30, label='Valleys', alpha=0.6)
            
            # Plot detected patterns
            motive_abc_mask = subdivisions.get('motive_abc', pd.Series(False, index=data.index))
            corrective_abc_mask = subdivisions.get('corrective_abc', pd.Series(False, index=data.index))
            motive_five = subdivisions.get('motive_five', pd.Series(dtype=object))
            corrective_five = subdivisions.get('corrective_five', pd.Series(dtype=object))
            
            if motive_abc_mask.any():
                plt.scatter(data.index[motive_abc_mask], data[motive_abc_mask], 
                          color='green', s=80, marker='^', label='Motive ABC', alpha=0.8)
            
            if corrective_abc_mask.any():
                plt.scatter(data.index[corrective_abc_mask], data[corrective_abc_mask], 
                          color='orange', s=80, marker='v', label='Corrective ABC', alpha=0.8)
            
            # Plot five-wave patterns with different markers
            if not motive_five.empty and motive_five.notna().any():
                valid_waves = motive_five[motive_five.notna()]
                for wave_idx, wave_label in valid_waves.items():
                    if wave_idx in data.index:
                        plt.scatter(wave_idx, data.loc[wave_idx], 
                                  color='purple', s=100, marker='*', 
                                  label=f'Motive Wave {wave_label}', alpha=0.9)
            
            if not corrective_five.empty and corrective_five.notna().any():
                valid_waves = corrective_five[corrective_five.notna()]
                for wave_idx, wave_label in valid_waves.items():
                    if wave_idx in data.index:
                        plt.scatter(wave_idx, data.loc[wave_idx], 
                                  color='brown', s=100, marker='s', 
                                  label=f'Corrective Wave {wave_label}', alpha=0.9)
            
            plt.title('Detailed Elliott Wave Analysis - DFS Engine')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('waves.png', dpi=150)
            print('Chart saved -> waves.png')
            
        except Exception as e:
            logger.error(f"Error creating detailed plot: {e}")
            # Create simple fallback plot
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data.values, color='black', linewidth=1)
            plt.title('Price Data (Fallback Plot)')
            plt.grid(True, alpha=0.3)
            plt.savefig('waves.png', dpi=150)
            print('Fallback chart saved -> waves.png')

    def enable_structural_turns(self, enable: bool = True):
        """
        Enable or disable structural turn detection.
        
        Parameters:
        -----------
        enable : bool
            Whether to use structural turns for wave analysis
        """
        self.use_structural_turns = enable
        logger.info(f"Structural turn detection {'enabled' if enable else 'disabled'}")


# ------------------------------------------------------------------
#  REAL-DATA  LOADER  –  visual sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os, logging, sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Try multiple possible locations for the CSV file
    CSV_NAME = "BTC-USD_1minute_data_cleaned.csv"
    
    # List of possible locations to search for the CSV
    possible_paths = [
        # Current directory
        CSV_NAME,
        # Backtesting folder (original assumption)
        os.path.join(os.path.dirname(__file__), '..', 'backtesting', CSV_NAME),
        # Project root directory
        os.path.join(os.path.dirname(__file__), '..', CSV_NAME),
        # Same directory as this script
        os.path.join(os.path.dirname(__file__), CSV_NAME),
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"Found CSV at: {path}")
            break
    
    if csv_path is None:
        print(f"ERROR: Could not find {CSV_NAME} in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease check:")
        print("1. The CSV file exists and is named correctly")
        print("2. The file is in one of the locations listed above")
        print("3. Or update the CSV_NAME variable with the correct filename")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        df.index = df.index.tz_localize(None)  # strip timezone to avoid re-index grief
        
        # Ensure all price columns are numeric
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Forward fill and backward fill any NaN values
        df = df.ffill().bfill()
        
        data = df['close']
        print(f"Successfully loaded data with {len(data)} bars")
        
        # Add the parent directory to Python path to import from core
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from core.data_manager import DataManager
        
        # Create DataManager - peak detection happens automatically during initialization
        dm = DataManager(data=df, mode='backtest', window_size=None, structural_move_threshold=0.001)
        
        # Get algorithm inputs (peak detection is already done)
        inputs = dm.get_algorithm_inputs()
        
        # Verify inputs are properly formatted
        print(f"Data type: {type(inputs.get('data'))}")
        print(f"Data columns: {list(inputs.get('data').columns) if inputs.get('data') is not None else 'None'}")
        print(f"Peak mask type: {type(inputs.get('peak_mask'))}")
        print(f"Valley mask type: {type(inputs.get('valley_mask'))}")
        
        # Check peak detection results
        peaks, valleys = dm.get_peaks_valleys()
        print(f"Peaks detected: {len(peaks)}, Valleys detected: {len(valleys)}")

        # Test structural turns
        print("\n--- TESTING DFS SYSTEM ---")
        wm = WavesManager(dm, use_structural_turns=True)
        
        # Run analysis using the DFS system
        algorithm_inputs = dm.get_algorithm_inputs()
        subdivisions = wm.analyze_subdivisions_sequential(
            algorithm_inputs['data'], 
            algorithm_inputs['peak_mask'], 
            algorithm_inputs['valley_mask']
        )
        
        # Get masks for statistics
        masks = wm.get_subdivision_masks()
        print("motive hits:", masks['motive_mask'].sum(),
              "corrective hits:", masks['corrective_mask'].sum())

        # Get detection stats
        stats = wm.get_detection_stats()
        print(f"\n--- DFS DETECTION STATISTICS ---")
        print(f"Using Structural Turns: {stats['using_structural_turns']}")
        print(f"DFS Available: {stats['wave_aware_dfs_available']}")
        print(f"Total Subdivisions Detected: {stats['total_subdivisions_detected']}")
        for algo, count in stats['subdivision_counts'].items():
            print(f"  {algo}: {count}")

        # Create enhanced plot with 4-panel visualization
        print("\nGenerating DFS visualization with 4-panel plot...")
        wm.create_enhanced_plot(
            data=algorithm_inputs['data']['close'],
            peak_mask=algorithm_inputs['peak_mask'],
            valley_mask=algorithm_inputs['valley_mask'],
            subdivisions=subdivisions,
            max_points=10080
        )
                
    except Exception as e:
        import traceback
        print(f"Error processing CSV file: {e}")
        print("Full traceback:")
        print(traceback.format_exc())
        print("Please check that the CSV file is properly formatted with a datetime index and 'close' column")