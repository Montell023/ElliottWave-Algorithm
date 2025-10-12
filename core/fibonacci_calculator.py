# algostack/core/fibonacci_calculator.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FibonacciLevels:
    """Data class to store Fibonacci calculation results"""
    retracement_levels: pd.Series
    extension_levels: pd.Series
    projection_levels: pd.Series
    start_price: float
    end_price: float
    direction: str  # 'uptrend' or 'downtrend'

class FibonacciCalculator:
    """
    Comprehensive Fibonacci calculator for Elliott Wave analysis.
    Provides retracement, extension, and projection levels with caching
    and validation. Used by Elliott Wave algorithms for wave targets
    and validation.
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize the Fibonacci calculator.
        
        Parameters:
        -----------
        cache_size : int
            Maximum number of calculations to cache
        """
        self.cache_size = cache_size
        
        # Standard Fibonacci ratios used in Elliott Wave analysis
        self.retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
        self.extension_ratios = [1.272, 1.382, 1.5, 1.618, 2.0, 2.618, 3.618]
        self.projection_ratios = [0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        # Calculation cache
        self._cache: Dict[str, FibonacciLevels] = {}
        self._cache_keys = []  # For LRU cache management
        
        logger.info(f"FibonacciCalculator initialized with cache_size={cache_size}")

    def calculate_all_levels(self, start: float, end: float, trend_direction: str = 'auto') -> FibonacciLevels:
        """
        Calculate all Fibonacci levels for a price move.
        
        Parameters:
        -----------
        start : float
            Starting price of the move
        end : float
            Ending price of the move
        trend_direction : str
            'uptrend', 'downtrend', or 'auto' (auto-detect)
            
        Returns:
        --------
        FibonacciLevels
            Object containing all calculated levels
        """
        # Validate inputs
        self._validate_inputs(start, end)
        
        # Auto-detect trend direction if needed
        if trend_direction == 'auto':
            trend_direction = 'uptrend' if end > start else 'downtrend'
        
        # Check cache first
        cache_key = self._generate_cache_key(start, end, trend_direction)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for Fibonacci levels: {cache_key}")
            return self._cache[cache_key]
        
        logger.debug(f"Calculating Fibonacci levels: {start:.4f} -> {end:.4f} ({trend_direction})")
        
        # Calculate all level types
        retracement_levels = self._calculate_retracement_levels(start, end, trend_direction)
        extension_levels = self._calculate_extension_levels(start, end, trend_direction)
        projection_levels = self._calculate_projection_levels(start, end, trend_direction)
        
        # Create result object
        result = FibonacciLevels(
            retracement_levels=retracement_levels,
            extension_levels=extension_levels,
            projection_levels=projection_levels,
            start_price=start,
            end_price=end,
            direction=trend_direction
        )
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        return result

    def calculate_retracement_levels(self, start: float, end: float, trend_direction: str = 'auto') -> pd.Series:
        """
        Calculate Fibonacci retracement levels for a price move.
        
        Parameters:
        -----------
        start : float
            Starting price of the move
        end : float
            Ending price of the move
        trend_direction : str
            'uptrend', 'downtrend', or 'auto'
            
        Returns:
        --------
        pd.Series
            Series with retracement levels indexed by ratio names
        """
        all_levels = self.calculate_all_levels(start, end, trend_direction)
        return all_levels.retracement_levels

    def calculate_extension_levels(self, start: float, end: float, trend_direction: str = 'auto') -> pd.Series:
        """
        Calculate Fibonacci extension levels for a price move.
        
        Parameters:
        -----------
        start : float
            Starting price of the move
        end : float
            Ending price of the move
        trend_direction : str
            'uptrend', 'downtrend', or 'auto'
            
        Returns:
        --------
        pd.Series
            Series with extension levels indexed by ratio names
        """
        all_levels = self.calculate_all_levels(start, end, trend_direction)
        return all_levels.extension_levels

    def calculate_projection_levels(self, start: float, end: float, trend_direction: str = 'auto') -> pd.Series:
        """
        Calculate Fibonacci projection levels for a price move.
        
        Parameters:
        -----------
        start : float
            Starting price of the move
        end : float
            Ending price of the move
        trend_direction : str
            'uptrend', 'downtrend', or 'auto'
            
        Returns:
        --------
        pd.Series
            Series with projection levels indexed by ratio names
        """
        all_levels = self.calculate_all_levels(start, end, trend_direction)
        return all_levels.projection_levels

    def _calculate_retracement_levels(self, start: float, end: float, trend_direction: str) -> pd.Series:
        """Calculate retracement levels based on trend direction."""
        price_range = abs(end - start)
        levels = {}
        
        for ratio in self.retracement_ratios:
            level_name = f"R_{ratio}"
            if trend_direction == 'uptrend':
                # For uptrend retracement: end - ratio * range
                level_price = end - (ratio * price_range)
            else:  # downtrend
                # For downtrend retracement: end + ratio * range
                level_price = end + (ratio * price_range)
            levels[level_name] = level_price
        
        return pd.Series(levels)

    def _calculate_extension_levels(self, start: float, end: float, trend_direction: str) -> pd.Series:
        """Calculate extension levels based on trend direction."""
        price_range = abs(end - start)
        levels = {}
        
        for ratio in self.extension_ratios:
            level_name = f"E_{ratio}"
            if trend_direction == 'uptrend':
                # For uptrend extension: end + ratio * range
                level_price = end + (ratio * price_range)
            else:  # downtrend
                # For downtrend extension: end - ratio * range
                level_price = end - (ratio * price_range)
            levels[level_name] = level_price
        
        return pd.Series(levels)

    def _calculate_projection_levels(self, start: float, end: float, trend_direction: str) -> pd.Series:
        """Calculate projection levels based on trend direction."""
        price_range = abs(end - start)
        levels = {}
        
        for ratio in self.projection_ratios:
            level_name = f"P_{ratio}"
            if trend_direction == 'uptrend':
                # For uptrend projection: start + ratio * range
                level_price = start + (ratio * price_range)
            else:  # downtrend
                # For downtrend projection: start - ratio * range
                level_price = start - (ratio * price_range)
            levels[level_name] = level_price
        
        return pd.Series(levels)

    def calculate_wave_relationships(self, wave1_start: float, wave1_end: float, 
                                   wave2_end: float, wave_type: str = 'impulse') -> Dict[str, float]:
        """
        Calculate Fibonacci relationships between Elliott Waves.
        
        Parameters:
        -----------
        wave1_start : float
            Start price of Wave 1
        wave1_end : float
            End price of Wave 1
        wave2_end : float
            End price of Wave 2
        wave_type : str
            'impulse' or 'corrective'
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of wave relationships and projections
        """
        self._validate_inputs(wave1_start, wave1_end)
        self._validate_inputs(wave1_end, wave2_end)
        
        wave1_range = abs(wave1_end - wave1_start)
        wave2_retracement = abs(wave2_end - wave1_end)
        
        # Calculate Wave 2 retracement of Wave 1
        if wave1_range > 0:
            wave2_retracement_ratio = wave2_retracement / wave1_range
        else:
            wave2_retracement_ratio = 0
        
        # Common Elliott Wave projections
        relationships = {
            'wave2_retracement_ratio': wave2_retracement_ratio,
            'wave1_range': wave1_range,
            'wave2_retracement': wave2_retracement
        }
        
        # Add common projections based on wave type
        if wave_type == 'impulse':
            # Common impulse wave projections
            relationships.update({
                'wave3_1618_target': wave1_end + (1.618 * wave1_range),
                'wave3_2618_target': wave1_end + (2.618 * wave1_range),
                'wave5_0618_target': wave2_end + (0.618 * wave1_range),
                'wave5_1000_target': wave2_end + (1.000 * wave1_range),
            })
        else:  # corrective
            # Common corrective wave projections
            relationships.update({
                'wave_c_0618_target': wave1_end + (0.618 * wave1_range),
                'wave_c_1000_target': wave1_end + (1.000 * wave1_range),
                'wave_c_1618_target': wave1_end + (1.618 * wave1_range),
            })
        
        logger.debug(f"Wave relationships calculated: {wave_type}, "
                    f"Wave 2 retracement: {wave2_retracement_ratio:.3f}")
        
        return relationships

    def find_nearest_fib_level(self, current_price: float, fib_levels: pd.Series, 
                             threshold: float = 0.01) -> Optional[Tuple[str, float, float]]:
        """
        Find the nearest Fibonacci level to the current price.
        
        Parameters:
        -----------
        current_price : float
            Current price to check
        fib_levels : pd.Series
            Fibonacci levels to check against
        threshold : float
            Maximum distance threshold (as percentage)
            
        Returns:
        --------
        Optional[Tuple[str, float, float]]
            (level_name, level_price, distance_percent) or None if no level within threshold
        """
        if fib_levels.empty:
            return None
        
        nearest_level = None
        min_distance = float('inf')
        
        for level_name, level_price in fib_levels.items():
            if pd.isna(level_price):
                continue
                
            distance = abs(level_price - current_price)
            distance_percent = (distance / current_price) * 100
            
            if distance_percent < min_distance and distance_percent <= threshold:
                min_distance = distance_percent
                nearest_level = (level_name, level_price, distance_percent)
        
        if nearest_level:
            logger.debug(f"Nearest Fibonacci level: {nearest_level[0]} at {nearest_level[1]:.4f} "
                        f"(distance: {nearest_level[2]:.2f}%)")
        
        return nearest_level

    def validate_wave_relationships(self, waves: Dict[str, float], 
                                  tolerance: float = 0.05) -> Dict[str, bool]:
        """
        Validate Elliott Wave relationships against Fibonacci ratios.
        
        Parameters:
        -----------
        waves : Dict[str, float]
            Dictionary of wave prices {'Wave_1': price, 'Wave_2': price, ...}
        tolerance : float
            Tolerance for Fibonacci ratio validation
            
        Returns:
        --------
        Dict[str, bool]
            Validation results for common Fibonacci relationships
        """
        validation_results = {}
        
        try:
            # Extract wave prices
            wave_prices = {k: v for k, v in waves.items() if not pd.isna(v)}
            
            if len(wave_prices) < 2:
                logger.warning("Insufficient wave data for validation")
                return validation_results
            
            # Common Fibonacci relationships to validate
            if 'Wave_1' in wave_prices and 'Wave_2' in wave_prices:
                wave1_range = abs(wave_prices['Wave_1'] - wave_prices.get('Starting_Point', wave_prices['Wave_1']))
                wave2_retracement = abs(wave_prices['Wave_2'] - wave_prices['Wave_1'])
                
                if wave1_range > 0:
                    retracement_ratio = wave2_retracement / wave1_range
                    # Common Wave 2 retracement levels
                    validation_results['wave2_retracement_382'] = abs(retracement_ratio - 0.382) <= tolerance
                    validation_results['wave2_retracement_500'] = abs(retracement_ratio - 0.500) <= tolerance
                    validation_results['wave2_retracement_618'] = abs(retracement_ratio - 0.618) <= tolerance
                    validation_results['wave2_retracement_786'] = abs(retracement_ratio - 0.786) <= tolerance
            
            if 'Wave_3' in wave_prices and 'Wave_1' in wave_prices:
                wave1_range = abs(wave_prices['Wave_1'] - wave_prices.get('Starting_Point', wave_prices['Wave_1']))
                wave3_extension = abs(wave_prices['Wave_3'] - wave_prices['Wave_2'])
                
                if wave1_range > 0:
                    extension_ratio = wave3_extension / wave1_range
                    # Common Wave 3 extension levels
                    validation_results['wave3_extension_1618'] = abs(extension_ratio - 1.618) <= tolerance
                    validation_results['wave3_extension_2618'] = abs(extension_ratio - 2.618) <= tolerance
            
            logger.debug(f"Wave validation results: {sum(validation_results.values())}/{len(validation_results)} passed")
            
        except Exception as e:
            logger.error(f"Error in wave validation: {e}")
        
        return validation_results

    def _validate_inputs(self, start: float, end: float):
        """Validate input parameters for Fibonacci calculations."""
        if isinstance(start, (pd.Series, pd.DataFrame)):
            raise ValueError("Start price must be scalar, not Series/DataFrame")
        if isinstance(end, (pd.Series, pd.DataFrame)):
            raise ValueError("End price must be scalar, not Series/DataFrame")
        
        try:
            start_float = float(start)
            end_float = float(end)
        except (ValueError, TypeError):
            raise ValueError("Start and end prices must be convertible to float")
        
        if start_float <= 0 or end_float <= 0:
            raise ValueError("Prices must be positive values")
        
        if abs(start_float - end_float) < 0.0001:  # Avoid division by zero
            raise ValueError("Start and end prices are too close for meaningful Fibonacci calculations")

    def _generate_cache_key(self, start: float, end: float, trend_direction: str) -> str:
        """Generate cache key for Fibonacci calculations."""
        return f"fib_{start:.6f}_{end:.6f}_{trend_direction}"

    def _cache_result(self, key: str, result: FibonacciLevels):
        """Cache calculation result with LRU management."""
        if key in self._cache:
            return  # Already cached
        
        # Add to cache
        self._cache[key] = result
        self._cache_keys.append(key)
        
        # Manage cache size (LRU)
        if len(self._cache_keys) > self.cache_size:
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]
        
        logger.debug(f"Cached Fibonacci calculation: {key} (cache size: {len(self._cache_keys)})")

    def clear_cache(self):
        """Clear calculation cache."""
        self._cache.clear()
        self._cache_keys.clear()
        logger.info("Fibonacci calculator cache cleared")

    def get_cache_info(self) -> Dict[str, any]:
        """Get cache statistics and information."""
        return {
            'cache_size': len(self._cache),
            'cache_max_size': self.cache_size,
            'cache_utilization': len(self._cache) / self.cache_size,
            'cached_keys_sample': self._cache_keys[:5] if self._cache_keys else []
        }

    def get_ratio_info(self) -> Dict[str, List[float]]:
        """Get information about configured Fibonacci ratios."""
        return {
            'retracement_ratios': self.retracement_ratios,
            'extension_ratios': self.extension_ratios,
            'projection_ratios': self.projection_ratios
        }


# Test the FibonacciCalculator
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing FibonacciCalculator...")
    
    # Create calculator
    fib_calc = FibonacciCalculator(cache_size=50)
    
    # Test 1: Basic Fibonacci calculations
    print("\n1. Basic Fibonacci Calculations:")
    
    # Uptrend example
    start_price = 100.0
    end_price = 120.0
    
    print(f"\nUptrend: {start_price} -> {end_price}")
    levels = fib_calc.calculate_all_levels(start_price, end_price, 'uptrend')
    
    print("Retracement Levels:")
    for level_name, price in levels.retracement_levels.items():
        print(f"  {level_name}: {price:.4f}")
    
    print("\nExtension Levels:")
    for level_name, price in levels.extension_levels.items():
        print(f"  {level_name}: {price:.4f}")
    
    # Downtrend example
    start_price = 120.0
    end_price = 100.0
    
    print(f"\nDowntrend: {start_price} -> {end_price}")
    levels = fib_calc.calculate_all_levels(start_price, end_price, 'downtrend')
    
    print("Retracement Levels:")
    for level_name, price in levels.retracement_levels.items():
        print(f"  {level_name}: {price:.4f}")
    
    # Test 2: Wave relationships
    print("\n2. Wave Relationship Calculations:")
    
    wave1_start = 100.0
    wave1_end = 120.0
    wave2_end = 112.0
    
    relationships = fib_calc.calculate_wave_relationships(wave1_start, wave1_end, wave2_end, 'impulse')
    
    print("Wave Relationships:")
    for relationship, value in relationships.items():
        if 'ratio' in relationship:
            print(f"  {relationship}: {value:.3f}")
        else:
            print(f"  {relationship}: {value:.4f}")
    
    # Test 3: Nearest level finding
    print("\n3. Nearest Level Detection:")
    
    current_price = 115.0
    nearest = fib_calc.find_nearest_fib_level(current_price, levels.retracement_levels, threshold=1.0)
    
    if nearest:
        level_name, level_price, distance = nearest
        print(f"Nearest Fibonacci level to {current_price}:")
        print(f"  Level: {level_name} at {level_price:.4f}")
        print(f"  Distance: {distance:.2f}%")
    else:
        print("No Fibonacci level within threshold")
    
    # Test 4: Wave validation
    print("\n4. Wave Validation:")
    
    waves = {
        'Starting_Point': 100.0,
        'Wave_1': 120.0,
        'Wave_2': 112.0,
        'Wave_3': 132.0
    }
    
    validation = fib_calc.validate_wave_relationships(waves)
    
    print("Wave Validation Results:")
    for relationship, is_valid in validation.items():
        status = "PASS" if is_valid else "FAIL"
        print(f"  {relationship}: {status}")
    
    # Test 5: Cache functionality
    print("\n5. Cache Functionality:")
    
    cache_info = fib_calc.get_cache_info()
    print("Cache Information:")
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    ratio_info = fib_calc.get_ratio_info()
    print("\nConfigured Ratios:")
    for ratio_type, ratios in ratio_info.items():
        print(f"  {ratio_type}: {ratios}")
    
    # Test 6: Performance with multiple calculations
    print("\n6. Performance Test:")
    
    import time
    
    start_time = time.time()
    
    # Perform multiple calculations
    for i in range(10):
        start = 100 + i
        end = 120 + i
        _ = fib_calc.calculate_all_levels(start, end, 'uptrend')
    
    end_time = time.time()
    
    print(f"10 Fibonacci calculations completed in {((end_time - start_time) * 1000):.2f} ms")
    
    # Check cache after performance test
    cache_info_after = fib_calc.get_cache_info()
    print(f"Cache utilization after test: {cache_info_after['cache_utilization']:.1%}")
    
    print("\nFibonacciCalculator test completed successfully!")