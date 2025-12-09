import pandas as pd
import numpy as np

def subdivides_downtrend(data, peak_mask, valley_mask):
    """
    Identifies a five-wave downtrend structure using vectorized operations and boolean masks.

    Returns:
        pd.Series: Wave labels ('1', '2', '3', '4', '5') indexed by valley positions.
    
    Pseudocode:
      1. Ensure data, peak_mask, and valley_mask have identical indices.
      2. Validate input types and ensure boolean masks.
      3. Extract indices of peaks and valleys.
      4. Handle cases with insufficient data (return empty Series).
      5. Extract prices efficiently using NumPy arrays.
      6. Apply vectorized conditions for wave detection:
         - Wave 1: Valley below previous peak.
         - Wave 2: Peak after Wave 1, above it but below the previous peak.
         - Wave 3: Valley below Wave 1 by specified Fibonacci levels.
         - Wave 4: Peak after Wave 3, below Wave 1.
         - Wave 5: Valley below Wave 3.
      7. Assign wave labels ('1' to '5') only where conditions hold.
      8. Return pd.Series with labels indexed by valley positions.
    """
            # --- Ensure `data` is a Pandas Series of close prices ---
    if isinstance(data, pd.DataFrame):
        data = data['close']  # Extract the 'close' column
    
    assert isinstance(data, pd.Series), "`data` must be a pandas Series of close prices."
    
    # --- Index Alignment Checks ---
    assert data.index.equals(peak_mask.index), "Index mismatch: data/peak_mask"
    assert data.index.equals(valley_mask.index), "Index mismatch: data/valley_mask"

    # --- Ensure peak_mask and valley_mask are boolean Series ---
    assert peak_mask.dtype == 'bool', "peak_mask must be a boolean Series"
    assert valley_mask.dtype == 'bool', "valley_mask must be a boolean Series"

    # --- Identify Peaks and Valleys ---
    peaks = data.index[peak_mask]
    valleys = data.index[valley_mask]

    # --- Handle Insufficient Data ---
    if len(peaks) < 2 or len(valleys) < 5:
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

    # Initialize an empty array for classifications
    classifications = np.full(len(valleys), None, dtype=object)

    # Assign wave labels where conditions are met
    if np.any(cond_wave_1[:-3] & cond_wave_2[:-2] & cond_wave_3[:-1] & cond_wave_4 & cond_wave_5):
        classifications[1] = '1'  # Wave 1
        classifications[2] = '2'  # Wave 2
        classifications[3] = '3'  # Wave 3
        classifications[4] = '4'  # Wave 4
        classifications[5] = '5'  # Wave 5

    return pd.Series(classifications, index=valleys)
