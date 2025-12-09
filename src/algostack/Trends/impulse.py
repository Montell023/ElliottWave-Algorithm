import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to Python path to import modules from the Waves package.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import Fibonacci functions and subdivision validation modules.
from Waves.fibos import calculate_fibonacci_levels, calculate_extension_levels
from Waves.motive_abc import motive
from Waves.corrective_abc import corrective

def impulse(data, mean, peak_mask, valley_mask, above_mean, below_mean):
    """
    Vectorized Elliott Wave detector for an impulse (non‐inverse) pattern.

    Pseudocode:
      1. Starting Point:
         "Where the data at a valley is equal to below_mean AND is less than the mean,
          designate that valley as the Starting Point."

      2. Wave 1:
         "Where a peak, occurring after the Starting Point, is more than the Starting Point
          AND greater than the mean OR equal to above_mean, designate that peak as Wave 1."

      3. Wave 2:
         "Where a valley, occurring after Wave 1, is equal to below_mean AND less than the mean
          AND less than the Starting Point, designate that valley as Wave 2."

      4. Wave 3:
         "Where a peak, occurring after Wave 2, is greater than the mean AND greater than ANY
          Fibonacci extension level (computed from Wave 1 to the Starting Point) OR equal to above_mean,
          designate that peak as Wave 3."

      5. Wave 4:
         "Where a valley, occurring after Wave 3, is less than the price at Wave 3 AND less than the mean
          AND equal to below_mean, designate that valley as Wave 4."

      6. Wave 5:
         "Where a peak, occurring after Wave 4, is greater than the mean AND greater than the price at Wave 3
          OR equal to above_mean AND greater than ANY Fibonacci extension level (computed from Wave 3 to Wave 4),
          designate that peak as Wave 5."

    Parameters:
      - data, mean, peak_mask, valley_mask: Precomputed inputs; data must be a Series of close prices.
      - above_mean, below_mean: Boolean Series indicating where data is above or below the moving average.

    Notes:
      - Precomputed subdivisions from corrective() and motive() are converted into boolean masks for candidate validation.
      - Dynamic Fibonacci levels are computed on the fly via imported functions (from fibos.py) and applied using broadcasting.
      - This script shares the same structure as correction_inverse.py and correction.py; only the logical conditions differ.
    """
    # --- Convert data to a Series if it is a DataFrame ---
    if isinstance(data, pd.DataFrame):
        data = data['close']

    # --- Explicit Series Assertion ---
    assert isinstance(data, pd.Series), "`data` must be a pandas Series of close prices."

    # --- Index Alignment Checks ---
    assert data.index.equals(mean.index), "Index mismatch: data/mean"
    assert data.index.equals(peak_mask.index), "Index mismatch: data/peak_mask"
    assert data.index.equals(valley_mask.index), "Index mismatch: data/valley_mask"
    assert data.index.equals(above_mean.index), "Index mismatch: data/above_mean"
    assert data.index.equals(below_mean.index), "Index mismatch: data/below_mean"

    # --- Precomputed Static Masks ---
    valleys = data.index[valley_mask]  # Indices where valleys occur.
    peaks = data.index[peak_mask]        # Indices where peaks occur.

    # --- Precomputed Subdivisions for Validation ---
    corrective_waves = corrective(data, peak_mask, valley_mask)
    corrective_indices = [wave.start for wave in corrective_waves] if corrective_waves else []
    motive_waves = motive(data, peak_mask, valley_mask)
    motive_indices = [wave.start for wave in motive_waves] if motive_waves else []
    corrective_mask = pd.Series(data.index.isin(corrective_indices), index=data.index)
    motive_mask = pd.Series(data.index.isin(motive_indices), index=data.index)

    waves = []  # List to store detected wave patterns.

    # --- 1. Starting Point ---
    # Condition: (Valley is below_mean AND price < mean)
    start_mask = (below_mean.loc[valleys]) & (data.loc[valleys] < mean.loc[valleys])
    starting_points = valleys[start_mask]
    if starting_points.empty:
        return pd.DataFrame(waves)
    # Select the valley with the lowest price as the Starting Point.
    starting_point = data.loc[starting_points].idxmin()

    wave = {}
    wave['Starting Point'] = starting_point

    # --- 2. Wave 1 ---
    # Condition: Peak occurs after the Starting Point AND
    #    (price > data at Starting Point AND price > mean) OR (satisfies above_mean).
    wave1_candidates = peaks[
        (peaks > starting_point) &
        (
            ((data.loc[peaks] > data.loc[starting_point]) & (data.loc[peaks] > mean.loc[peaks]))
            | (above_mean.loc[peaks])
        )
    ]
    if wave1_candidates.empty:
        return pd.DataFrame(waves)
    # Select the peak with the highest price as Wave 1.
    wave['Wave 1'] = data.loc[wave1_candidates].idxmax()

    # --- 3. Wave 2 ---
    # Condition: Valley occurs after Wave 1 AND
    #    (price < mean AND price < data at Starting Point) AND (satisfies below_mean).
    wave2_candidates = valleys[
        (valleys > wave['Wave 1']) &
        ((data.loc[valleys] < mean.loc[valleys]) & (data.loc[valleys] < data.loc[starting_point]))
    ]
    if wave2_candidates.empty:
        return pd.DataFrame(waves)
    # Select the valley with the lowest price as Wave 2.
    wave['Wave 2'] = data.loc[wave2_candidates].idxmin()

    # --- 4. Wave 3 ---
    # Condition: Peak occurs after Wave 2 AND
    #    [ (price > mean AND price > ANY Fibonacci extension level computed from (Wave 1 → Starting Point))
    #      OR (satisfies above_mean) ].
    wave3_candidates = peaks[peaks > wave['Wave 2']]
    if wave3_candidates.empty:
        return pd.DataFrame(waves)
    # Compute Fibonacci extension levels from Wave 1 to Starting Point.
    fib_ext_w1_sp = calculate_extension_levels(float(data.loc[wave['Wave 1']]), float(data.loc[starting_point]))
    cond_wave3 = (
        (data.loc[wave3_candidates] > mean.loc[wave3_candidates]) &
        (data.loc[wave3_candidates].values[:, np.newaxis] > fib_ext_w1_sp.values).any(axis=1)
    ) | (above_mean.loc[wave3_candidates])
    valid_wave3 = wave3_candidates[cond_wave3]
    if valid_wave3.empty:
        return pd.DataFrame(waves)
    # Select the peak with the highest price as Wave 3.
    wave['Wave 3'] = data.loc[valid_wave3].idxmax()

    # --- 5. Wave 4 ---
    # Condition: Valley occurs after Wave 3 AND
    #    (price < data at Wave 3 AND price < mean AND satisfies below_mean).
    wave4_candidates = valleys[valleys > wave['Wave 3']]
    if wave4_candidates.empty:
        return pd.DataFrame(waves)
    cond_wave4 = (data.loc[wave4_candidates] < data.loc[wave['Wave 3']]) & (data.loc[wave4_candidates] < mean.loc[wave4_candidates]) & (below_mean.loc[wave4_candidates])
    valid_wave4 = wave4_candidates[cond_wave4]
    if valid_wave4.empty:
        return pd.DataFrame(waves)
    # Select the valley with the lowest price as Wave 4.
    wave['Wave 4'] = data.loc[valid_wave4].idxmin()

    # --- 6. Wave 5 ---
    # Condition: Peak occurs after Wave 4 AND
    #    [ (price > mean AND price > data at Wave 3)
    #      OR (satisfies above_mean AND price > ANY Fibonacci extension level computed from (Wave 3 → Wave 4)) ].
    wave5_candidates = peaks[peaks > wave['Wave 4']]
    if wave5_candidates.empty:
        return pd.DataFrame(waves)
    # Compute Fibonacci extension levels from Wave 3 to Wave 4.
    fib_ext_w3_w4 = calculate_extension_levels(float(data.loc[wave['Wave 3']]), float(data.loc[wave['Wave 4']]))
    cond_wave5 = (
        (data.loc[wave5_candidates] > mean.loc[wave5_candidates]) &
        (data.loc[wave5_candidates] > data.loc[wave['Wave 3']])
    ) | (
        (above_mean.loc[wave5_candidates]) &
        (data.loc[wave5_candidates].values[:, np.newaxis] > fib_ext_w3_w4.values).any(axis=1)
    )
    valid_wave5 = wave5_candidates[cond_wave5]
    if valid_wave5.empty:
        return pd.DataFrame(waves)
    # Select the peak with the highest price as Wave 5.
    wave['Wave 5'] = data.loc[valid_wave5].idxmax()

    waves.append(wave)
    return pd.DataFrame(waves).dropna(how='all')
