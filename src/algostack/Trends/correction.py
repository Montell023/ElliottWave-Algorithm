import pandas as pd
import numpy as np

# Import modules: Fibonacci functions from fibos.py and validation modules.
from algostack.Waves.fibos import calculate_extension_levels #============= correct routing 
from Waves.motive_abc import motive #============= fix
from Waves.corrective_abc import corrective #============= fix 

def correction(data, mean, lower_band, peak_mask, valley_mask, above_mean, below_mean):
    """
    Vectorized Elliott Wave detector for an uptrend correction (inverse of correction_inverse).

    Pseudocode:
      1. Starting Point:
         "Where the data at a valley is equal to below_mean AND is less than the mean
          OR is equal to the lower_band, designate that valley as the Starting Point."

      2. Wave A:
         "Where the data at a peak is equal to above_mean AND is greater than the mean
          OR where motive_abc subdivide is valid, designate that peak as Wave A."
         --> From these candidates, choose the peak with the highest price (using idxmax()).

      3. Wave B:
         "Where the data at a valley is equal to below_mean AND is less than the price at the Starting Point
          OR where corrective_abc subdivide is valid, designate that valley as Wave B."
         --> From these candidates, choose the valley with the lowest price (using idxmin()).

      4. Wave C:
         "Where the data at a peak is equal to above_mean AND is less than the price at Wave A
          OR its price is greater than ANY Fibonacci extension level (calculated dynamically),
          designate that peak as Wave C termination."
         --> From these candidates, choose the peak with the lowest price (using idxmin()).

    Parameters:
      - data, mean, lower_band, peak_mask, valley_mask: Precomputed inputs; data must be a Series of close prices.
      - above_mean, below_mean: Boolean Series indicating where data is above or below the moving average.

    Notes:
      - Precomputed subdivisions from motive_abc and corrective_abc are used for candidate validation.
      - Dynamic Fibonacci levels are computed on the fly (via broadcasting) and applied as dynamic masks.
      - This function mirrors the structure of correction_inverse.py, with the key difference being the inverse logic.
    """
    # --- Convert data to a Series if it is a DataFrame ---
    if isinstance(data, pd.DataFrame):
        data = data['close']

    # --- Explicit Series Assertion ---
    assert isinstance(data, pd.Series), "`data` must be a pandas Series of close prices."

    # --- Index Alignment Checks ---
    assert data.index.equals(mean.index), "Index mismatch: data/mean"
    assert data.index.equals(lower_band.index), "Index mismatch: data/lower_band"
    assert data.index.equals(peak_mask.index), "Index mismatch: data/peak_mask"
    assert data.index.equals(valley_mask.index), "Index mismatch: data/valley_mask"

    # --- Precomputed Static Masks ---
    valleys = data.index[valley_mask]   # Precomputed valley indices
    peaks = data.index[peak_mask]         # Precomputed peak indices

    # --- Precomputed Subdivisions for Validation ---
    corrective_waves = corrective(data, peak_mask, valley_mask)
    corrective_indices = [wave.start for wave in corrective_waves] if corrective_waves else []

    motive_waves = motive(data, peak_mask, valley_mask)
    motive_indices = [wave.start for wave in motive_waves] if motive_waves else []

    # Create boolean masks for fast lookup instead of using isin() on lists.
    corrective_mask = pd.Series(data.index.isin(corrective_indices), index=data.index)
    motive_mask = pd.Series(data.index.isin(motive_indices), index=data.index)

    waves = []  # To store detected wave patterns

    # --- 1. Starting Point Selection ---
    # Condition: (Valley is below_mean AND price < mean) OR (price equals lower_band)
    start_mask = ((below_mean.loc[valleys]) & (data.loc[valleys] < mean.loc[valleys])) | (data.loc[valleys] == lower_band.loc[valleys])
    starting_points = valleys[start_mask]

    for sp in starting_points:
        wave = {'Starting_Point': sp, 'Wave A': None, 'Wave B': None, 'Wave C': None}

        # --- 2. Identify Wave A ---
        # Condition: Candidate peak occurs after sp and either:
        #   - Is above_mean AND price > mean, OR
        #   - Is validated by motive_abc (using motive_mask)
        wave_a_candidates = peaks[
            (peaks > sp) &
            (
                ((above_mean.loc[peaks]) & (data.loc[peaks] > mean.loc[peaks]))
                | (motive_mask.loc[peaks])
            )
        ]
        if wave_a_candidates.empty:
            continue
        # Select candidate with the highest price as Wave A (idxmax)
        wave_a = data.loc[wave_a_candidates].idxmax()
        wave['Wave A'] = wave_a

        # --- 3. Identify Wave B ---
        # Condition: Candidate valley occurs after Wave A and either:
        #   - Is below_mean AND its price < data at Starting Point, OR
        #   - Is validated by corrective_abc (using corrective_mask)
        wave_b_candidates = valleys[
            (valleys > wave_a) &
            (
                ((below_mean.loc[valleys]) & (data.loc[valleys] < data.loc[sp]))
                | (corrective_mask.loc[valleys])
            )
        ]
        if wave_b_candidates.empty:
            continue
        # Select candidate with the lowest price as Wave B (idxmin)
        wave_b = data.loc[wave_b_candidates].idxmin()
        wave['Wave B'] = wave_b

        # --- 4. Identify Wave C ---
        # Condition: Candidate peak occurs after Wave B and either:
        #   - Is above_mean AND its price < data at Wave A, OR
        #   - Its price is greater than ANY dynamically computed Fibonacci extension level.
        wave_c_candidates = peaks[peaks > wave_b]
        if wave_c_candidates.empty:
            continue

        cond1 = (above_mean.loc[wave_c_candidates]) & (data.loc[wave_c_candidates] < data.loc[wave_a])
        # Dynamic Fibonacci extension computed using Starting Point and Wave A.
        fib_ext = calculate_extension_levels(float(data.loc[sp]), float(data.loc[wave_a]))
        cond2 = (data.loc[wave_c_candidates].values[:, np.newaxis] > fib_ext.values).any(axis=1)

        valid_candidates = wave_c_candidates[(cond1 | cond2)]
        if valid_candidates.empty:
            continue
        # Select candidate with the lowest price among valid candidates as Wave C (idxmin)
        wave_c = data.loc[valid_candidates].idxmin()
        wave['Wave C'] = wave_c

        waves.append(wave)

    return pd.DataFrame(waves).dropna(how='all')
