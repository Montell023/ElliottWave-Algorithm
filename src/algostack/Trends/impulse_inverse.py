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

def impulse_inverse(data, mean, peak_mask, valley_mask, above_mean, below_mean):
    """
    Vectorized Elliott Wave detector for an impulse inverse pattern.

    Pseudocode:
      1. Starting Point:
         "Where the data at a peak is equal to above_mean AND is greater than the mean,
          designate that peak as the Starting Point."

      2. Wave 1:
         "Where a valley, occurring after the Starting Point, is less than the Starting Point
          AND less than the mean OR validated by corrective criteria, designate that valley as Wave 1."
         --> Select the valley with the lowest price (using idxmin()).

      3. Wave 2:
         "Where a peak, occurring after Wave 1, is equal to above_mean AND is greater than the mean,
          designate that peak as Wave 2."
         --> Select the peak with the highest price (using idxmax()).

      4. Wave 3:
         "Where a valley, occurring after Wave 2, is less than the mean AND its price is less than ANY
          Fibonacci extension level (computed dynamically from Wave 1 to the Starting Point) OR is equal to below_mean,
          designate that valley as Wave 3."
         --> Select the valley with the lowest price (using idxmin()).

      5. Wave 4:
         "Where a peak, occurring after Wave 3, is greater than the price at Wave 3 AND is equal to above_mean
          AND is less than the price at Wave 1, designate that peak as Wave 4."
         --> Select the peak with the highest price (using idxmax()).

      6. Wave 5:
         "Where a valley, occurring after Wave 4, is less than the mean AND less than the price at Wave 3,
          OR is equal to below_mean AND its price is less than ANY Fibonacci extension level (computed dynamically from Wave 3 to Wave 4),
          designate that valley as Wave 5."
         --> Select the valley with the lowest price (using idxmin()).

    Parameters:
      - data, mean, peak_mask, valley_mask: Precomputed inputs; data must be a Series of close prices.
      - above_mean, below_mean: Boolean Series indicating where data is above or below the moving average.

    Notes:
      - Precomputed subdivisions from corrective() and motive() are converted into boolean masks (corrective_mask and motive_mask)
        for candidate validation.
      - Dynamic Fibonacci levels are computed on the fly via our imported functions (from fibos.py)
        and applied with broadcasting (.values[:, None] and .any(axis=1)).
      - This structure mirrors correction_inverse.py and correction.py; only the logical conditions differ.
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
    peaks = data.index[peak_mask]       # Indices where peaks occur.
    valleys = data.index[valley_mask]     # Indices where valleys occur.

    # --- Precomputed Subdivisions for Validation ---
    corrective_waves = corrective(data, peak_mask, valley_mask)
    corrective_indices = [wave.start for wave in corrective_waves] if corrective_waves else []
    motive_waves = motive(data, peak_mask, valley_mask)
    motive_indices = [wave.start for wave in motive_waves] if motive_waves else []
    corrective_mask = pd.Series(data.index.isin(corrective_indices), index=data.index)
    motive_mask = pd.Series(data.index.isin(motive_indices), index=data.index)

    waves = []  # To store detected wave patterns.

    # --- 1. Starting Point ---
    # Condition: (Peak is above_mean AND price > mean)
    start_mask = (above_mean.loc[peaks]) & (data.loc[peaks] > mean.loc[peaks])
    starting_points = peaks[start_mask]

    for sp in starting_points:
        wave = {
            'Starting Point': sp,
            'Wave 1': None,
            'Wave 2': None,
            'Wave 3': None,
            'Wave 4': None,
            'Wave 5': None
        }

        # --- 2. Wave 1 ---
        # Condition: Valley occurs after sp AND
        #    [(price < data at sp AND price < mean) OR validated by corrective criteria].
        wave1_candidates = valleys[
            (valleys > sp) &
            (
                ((data.loc[valleys] < data.loc[sp]) & (data.loc[valleys] < mean.loc[valleys]))
                | corrective_mask.loc[valleys]
            )
        ]
        if wave1_candidates.empty:
            continue
        wave['Wave 1'] = data.loc[wave1_candidates].idxmin()

        # --- 3. Wave 2 ---
        # Condition: Peak occurs after Wave 1 AND
        #    [(above_mean is True AND price > mean) OR validated by motive criteria].
        wave2_candidates = peaks[
            (peaks > wave['Wave 1']) &
            (
                ((above_mean.loc[peaks]) & (data.loc[peaks] > mean.loc[peaks]))
                | motive_mask.loc[peaks]
            )
        ]
        if wave2_candidates.empty:
            continue
        wave['Wave 2'] = data.loc[wave2_candidates].idxmax()

        # --- 4. Wave 3 ---
        # Condition: Valley occurs after Wave 2 AND
        #    [(price < mean AND price < ANY Fibonacci extension level computed from (Wave 1 → Starting Point))
        #     OR (satisfies below_mean)].
        wave3_candidates = valleys[valleys > wave['Wave 2']]
        if wave3_candidates.empty:
            continue
        # Compute Fibonacci extension levels from Wave 1 to Starting Point (corrected direction).
        fib_ext_sp_w1 = calculate_extension_levels(float(data.loc[wave['Wave 1']]), float(data.loc[sp]))
        cond_wave3 = (
            (data.loc[wave3_candidates] < mean.loc[wave3_candidates]) &
            (data.loc[wave3_candidates].values[:, np.newaxis] < fib_ext_sp_w1.values).any(axis=1)
        ) | below_mean.loc[wave3_candidates]
        valid_wave3 = wave3_candidates[cond_wave3]
        if valid_wave3.empty:
            continue
        wave['Wave 3'] = data.loc[valid_wave3].idxmin()

        # --- 5. Wave 4 ---
        # Condition: Peak occurs after Wave 3 AND
        #    [(price > price at Wave 3 AND above_mean is True) AND (price < data at Wave 1)].
        wave4_candidates = peaks[
            (peaks > wave['Wave 3']) &
            (data.loc[peaks] > data.loc[wave['Wave 3']]) &
            (above_mean.loc[peaks]) &
            (data.loc[peaks] < data.loc[wave['Wave 1']])
        ]
        if wave4_candidates.empty:
            continue
        wave['Wave 4'] = data.loc[wave4_candidates].idxmax()

        # --- 6. Wave 5 ---
        # Condition: Valley occurs after Wave 4 AND
        #    [(price < mean AND price < data at Wave 3)
        #     OR (satisfies below_mean AND price < ANY Fibonacci extension level computed from (Wave 3 → Wave 4))].
        wave5_candidates = valleys[valleys > wave['Wave 4']]
        if wave5_candidates.empty:
            continue
        # Compute new Fibonacci extension levels from Wave 3 to Wave 4.
        fib_ext_w3_w4 = calculate_extension_levels(float(data.loc[wave['Wave 3']]), float(data.loc[wave['Wave 4']]))
        cond_wave5 = (
            (data.loc[wave5_candidates] < mean.loc[wave5_candidates]) &
            (data.loc[wave5_candidates] < data.loc[wave['Wave 3']])
        ) | (
            below_mean.loc[wave5_candidates] &
            (data.loc[wave5_candidates].values[:, np.newaxis] < fib_ext_w3_w4.values).any(axis=1)
        )
        valid_wave5 = wave5_candidates[cond_wave5]
        if valid_wave5.empty:
            continue
        wave['Wave 5'] = data.loc[valid_wave5].idxmin()

        waves.append(wave)

    return pd.DataFrame(waves).dropna(how='all')
