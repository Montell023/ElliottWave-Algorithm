import pandas as pd
import numpy as np

def subdivides(peak_mask, valley_mask, data):
    """
    Identifies a five-wave subdividing structure using vectorized operations.

    Returns:
        pd.Series: Wave labels ('1', '2', '3', '4', '5') indexed by valley positions.

    Pseudocode:
      1. Validate input types and ensure index alignment.
      2. Extract indices of peaks and valleys.
      3. Handle cases with insufficient data.
      4. Apply vectorized conditions for wave structure.
      5. Assign wave labels ('1', '2', '3', '4', '5') only when conditions match.
      6. Return a pd.Series with wave labels indexed by valley positions.
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
    if len(peaks) < 2 or len(valleys) < 3:
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

    # Initialize an empty array for classifications
    classifications = np.full(len(valleys), None, dtype=object)

    # Assign wave labels where conditions hold
    if np.any(cond_wave_1[:-3] & cond_wave_2[:-2] & cond_wave_3[:-1] & cond_wave_4 & cond_wave_5):
        classifications[0] = '1'  # Wave 1
        classifications[1] = '2'  # Wave 2
        classifications[2] = '3'  # Wave 3
        classifications[3] = '4'  # Wave 4
        classifications[4] = '5'  # Wave 5

    return pd.Series(classifications, index=valleys)
