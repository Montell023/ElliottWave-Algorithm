import pandas as pd
import numpy as np

def calculate_fibonacci_levels(start, end):
    """
    Calculate Fibonacci retracement levels for a downtrend.
    
    Parameters:
      start (float): The starting price (e.g., the peak).
      end (float): The ending price (e.g., the Wave A price).
    
    Returns:
      pd.Series: A Series of retracement levels computed as:
                 start - fib_levels * (start - end)
                 for each Fibonacci ratio.
    
    Note:
      The returned Series can be used in vectorized comparisons (e.g., via broadcasting)
      to dynamically generate boolean masks in your main algorithm.
    """
    # Validate that inputs are scalars
    if isinstance(start, (pd.Series, pd.DataFrame)):
        raise ValueError("'start' must be a scalar value, not a Series or DataFrame. Use close prices only!")
    
    if isinstance(end, (pd.Series, pd.DataFrame)):
        raise ValueError("'end' must be a scalar value, not a Series or DataFrame. Use close prices only!")
    
    # Ensure numeric inputs
    try:
        start = float(start)
        end = float(end)
    except (ValueError, TypeError):
        raise ValueError("'start' and 'end' must be convertible to float values.")
    
    fib_levels = pd.Series([0.236, 0.382, 0.5, 0.618, 0.786])
    return start - fib_levels * (start - end)

def calculate_extension_levels(start, end):
    """
    Calculate Fibonacci extension levels for Wave C.
    
    Parameters:
      start (float): The starting price (e.g., the peak).
      end (float): The ending price (e.g., the Wave A price).
    
    Returns:
      pd.Series: A Series of extension levels computed as:
                 end - fib_levels * (end - start)
                 for each Fibonacci ratio.
    
    Note:
      This Series is designed to be used with broadcasting (e.g., comparing candidate valley prices
      with each extension level using .values[:, None] and .any(axis=1)) to create dynamic masks.
    """
    # Validate that inputs are scalars
    if isinstance(start, (pd.Series, pd.DataFrame)):
        raise ValueError("'start' must be a scalar value, not a Series or DataFrame. Use close prices only!")
    
    if isinstance(end, (pd.Series, pd.DataFrame)):
        raise ValueError("'end' must be a scalar value, not a Series or DataFrame. Use close prices only!")
    
    # Ensure numeric inputs
    try:
        start = float(start)
        end = float(end)
    except (ValueError, TypeError):
        raise ValueError("'start' and 'end' must be convertible to float values.")
    
    fib_levels = pd.Series([1.618, 2.618, 4.236])
    return end - fib_levels * (end - start)