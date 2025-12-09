import sys
import os

# Add the parent directory to the Python path to locate the Waves package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from Waves.motive_abc import motive
from Waves.motive_five import subdivides
from Waves.corrective_abc import corrective
from Waves.corrective_five import subdivides_downtrend

# Define a function to calculate Fibonacci levels
def calculate_fibonacci_levels(start, end):
    fib_levels = pd.Series([0.236, 0.382, 0.5, 0.618, 0.786])  # Fibonacci retracement levels
    fib_retracements = start + fib_levels * (end - start)
    return fib_retracements

# Define a function to calculate Fibonacci extension levels
def calculate_extension_levels(start, end, is_downtrend=False):
    fib_levels = pd.Series([1.618, 2.618, 4.236])  # Fibonacci extension levels
    if is_downtrend:
        # For a downtrend, extend from Wave A (lower) up to the starting point (higher)
        fib_extensions = start - fib_levels * (start - end)
    else:
        # For an uptrend, extend from Wave A (higher) down to the starting point (lower)
        fib_extensions = end - fib_levels * (end - start)
    return fib_extensions

def correction(data, peak_mask, valley_mask):
    waves = {'Wave A': None, 'Wave B': None, 'Wave C': None}

    # Get the indices where peaks and valleys occur
    peaks = data.index[peak_mask]
    valleys = data.index[valley_mask]

    # Check if peaks or valleys is empty
    if len(peaks) == 0 or len(valleys) == 0:
        return waves

    # Identify the first valley as the starting point
    starting_point = data.loc[valleys[0]]

    # Identify Wave A
    for peak in peaks:  # Iterate in natural order
        if peak not in data.index:
            continue
        if data.loc[peak] > data.mean() and data.loc[peak] > starting_point:
            waves['Wave A'] = peak
            break

    if waves['Wave A'] is None:
        return waves

    # Identify Wave B
    for valley in valleys:
        if valley not in data.index:
            continue
        if data.loc[valley] < data.loc[waves['Wave A']] and data.loc[valley] > starting_point:
            fib_retracements = calculate_fibonacci_levels(starting_point, data.loc[waves['Wave A']])
            corrective_waves = corrective(peak_mask, valley_mask)
            subdivides_downtrend_waves = subdivides_downtrend(peak_mask, valley_mask)
            if (data.loc[valley] < fib_retracements).any() or len(corrective_waves) > 0 or len(subdivides_downtrend_waves) > 0:
                waves['Wave B'] = valley
                break

    if waves['Wave B'] is None:
        return waves

    # Identify Wave C
    for peak in peaks:  # Iterate in natural order
        if peak not in data.index:
            continue
        if data.loc[peak] > data.mean() and data.loc[peak] > data.loc[waves['Wave B']]:
            fib_extensions = calculate_extension_levels(starting_point, data.loc[waves['Wave A']])
            motive_waves = motive(peak_mask, valley_mask)
            subdivides_waves = subdivides(peak_mask, valley_mask)
            if (data.loc[peak] > fib_extensions).any() or len(motive_waves) > 0 or len(subdivides_waves) > 0:
                waves['Wave C'] = peak
                break

    return waves