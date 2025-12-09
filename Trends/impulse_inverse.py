import sys
import os

# Add the parent directory to the Python path to locate the Waves package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from Waves.corrective_abc import corrective
from Waves.corrective_five import subdivides_downtrend
from Waves.motive_abc import motive
from Waves.motive_five import subdivides

def calculate_fibonacci_levels(start, end):
    fib_levels = pd.Series({
        '0.236': end - 0.236 * (end - start),
        '0.382': end - 0.382 * (end - start),
        '0.5': end - 0.5 * (end - start),
        '0.618': end - 0.618 * (end - start),
        '0.786': end - 0.786 * (end - start)
    })
    extension_levels = pd.Series({
        '1.618': end + 1.618 * (end - start),
        '2.618': end + 2.618 * (end - start),
        '4.236': end + 4.236 * (end - start)
    })
    return fib_levels, extension_levels

def downtrend(data, mean_price, upper_band, below_mean, peaks, valleys):
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Data must be a DataFrame or Series")
    if not isinstance(peaks, (pd.DataFrame, pd.Series)) or not isinstance(valleys, (pd.DataFrame, pd.Series)):
        raise ValueError("Peaks and valleys must be DataFrames or Series")

    # Convert peaks and valleys to pandas Series if they are not already
    if not isinstance(peaks, pd.Series):
        peaks = pd.Series(peaks)
    if not isinstance(valleys, pd.Series):
        valleys = pd.Series(valleys)

    if peaks.empty or valleys.empty:
        return False  # Indicate failure if peaks or valleys are empty

    classifications = []
    starting_point = None
    wave_1 = None
    wave_2 = None
    wave_3 = None
    wave_4 = None
    wave_5 = None

    for i in range(len(data)):
        # Check if 'i' is within the valid range before accessing elements
        if i < 0 or i >= len(data) or i >= len(peaks) or i >= len(valleys):
            classifications.append('Not Classified')
            continue

        price = data.iloc[i]

        if starting_point is None and peaks.iloc[i] == upper_band.iloc[i] and price > mean_price.iloc[i]:
            classifications.append('Starting Point')
            starting_point = price

        elif starting_point is not None and wave_1 is None and price < starting_point and price < mean_price.iloc[i] and valleys.iloc[i]:
            corrective_waves = corrective(peaks, valleys)
            subdivides_downtrend_waves = subdivides_downtrend(peaks, valleys)
            if corrective_waves or subdivides_downtrend_waves:
                classifications.append('Wave 1')
                wave_1 = price

        elif wave_1 is not None and wave_2 is None and price >= wave_1 and peaks.iloc[i] >= mean_price.iloc[i]:
            fib_levels, _ = calculate_fibonacci_levels(starting_point, wave_1)
            motive_waves = motive(peaks, valleys)
            subdivide_waves = subdivides(peaks, valleys)
            if (any(abs(price - fib_levels[level]) < 0.01 * fib_levels[level] for level in fib_levels)
                and (motive_waves or subdivide_waves)):
                classifications.append('Wave 2')
                wave_2 = price

        elif wave_2 is not None and wave_3 is None and price < wave_1 and valleys.iloc[i]:
            _, extension_levels = calculate_fibonacci_levels(starting_point, wave_1)
            corrective_waves = corrective(peaks, valleys)
            subdivides_downtrend_waves = subdivides_downtrend(peaks, valleys)
            if (any(price < extension_levels[level] for level in extension_levels)
                and (corrective_waves or subdivides_downtrend_waves)):
                classifications.append('Wave 3')
                wave_3 = price

        elif wave_3 is not None and wave_4 is None and price > wave_3 and price > mean_price.iloc[i] and price < wave_1 and peaks.iloc[i]:
            fib_levels, _ = calculate_fibonacci_levels(wave_2, wave_3)
            motive_waves = motive(peaks, valleys)
            subdivide_waves = subdivides(peaks, valleys)
            if (any(abs(price - fib_levels[level]) < 0.01 * fib_levels[level] for level in fib_levels)
                and (motive_waves or subdivide_waves)):
                classifications.append('Wave 4')
                wave_4 = price

        elif wave_4 is not None and wave_5 is None and price < wave_3 and valleys.iloc[i]:
            _, extension_levels = calculate_fibonacci_levels(starting_point, wave_1)
            corrective_waves = corrective(peaks, valleys)
            subdivides_downtrend_waves = subdivides_downtrend(peaks, valleys)
            if (any(price < extension_levels[level] for level in extension_levels)
                and (corrective_waves or subdivides_downtrend_waves)):
                classifications.append('Wave 5')
                wave_5 = price
            else:
                classifications.append('Not Wave 5')

        else:
            classifications.append('Not Classified')

    # Check if any classifications were made
    if 'Starting Point' in classifications or 'Wave 1' in classifications or 'Wave 2' in classifications:
        return True  # Return True if any waves were classified
    else:
        return False  # Return False if no waves were classified
