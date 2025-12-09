import logging
import pandas as pd
import numpy as np

from algostack.Waves.fibos import calculate_extension_levels

logger = logging.getLogger(__name__)

def correction_inverse(
    data: pd.Series,
    mean: pd.Series,
    upper_band: pd.Series,
    peak_mask: pd.Series,
    valley_mask: pd.Series,
    above_mean: pd.Series,
    below_mean: pd.Series,
    corrective_mask: pd.Series,
    motive_mask: pd.Series
) -> pd.DataFrame:
    """
    Vectorized Elliott Wave detector for Lumibot, using precomputed subdivides.
    """

    # --- 1) Ensure every input is aligned to data.index via reindex ---
    mean           = mean.reindex(data.index).ffill()
    upper_band     = upper_band.reindex(data.index).ffill()
    peak_mask      = peak_mask.reindex(data.index, fill_value=False)
    valley_mask    = valley_mask.reindex(data.index, fill_value=False)
    above_mean     = above_mean.reindex(data.index, fill_value=False)
    below_mean     = below_mean.reindex(data.index, fill_value=False)
    corrective_mask= corrective_mask.reindex(data.index, fill_value=False)
    motive_mask    = motive_mask.reindex(data.index, fill_value=False)

    # --- 2) Log entry and counts so backtest console shows us immediately ---
    peaks_idx   = data.index[peak_mask]
    valleys_idx = data.index[valley_mask]
    logger.info(
        "correction_inverse started — data pts=%d, peaks=%d, valleys=%d",
        len(data), len(peaks_idx), len(valleys_idx)
    )

    waves = []

    # --- 3) Starting Point selection using boolean masks directly ---
    start_mask = (
        (peak_mask & above_mean & (data > mean))
      | (peak_mask & (data == upper_band))
    )
    starting_points = data.index[start_mask]

    for sp in starting_points:
        wave = {'Starting_Point': sp, 'Wave_A': None, 'Wave_B': None, 'Wave_C': None}

        # --- 4) Wave A candidates: valleys after sp matching below_mean or corrective ---
        mask_a = valley_mask & (data.index > sp) & (
            (below_mean & (data < mean)) | corrective_mask
        )
        wave_a_idx = data.index[mask_a]

        if len(wave_a_idx) == 0:
            logger.debug(f"[{sp}] no Wave A candidates")
            continue

        wave_a = data.loc[wave_a_idx].idxmin()
        wave['Wave_A'] = wave_a

        # --- 5) Wave B candidates: peaks between A and SP matching above_mean or motive ---
        mask_b = peak_mask & (data.index > wave_a) & (data.index < sp) & (
            (above_mean & (data > data.loc[wave_a]) & (data < data.loc[sp]))
          | motive_mask
        )
        wave_b_idx = data.index[mask_b]

        if len(wave_b_idx) == 0:
            logger.debug(f"[{sp}] → [{wave_a}] no Wave B candidates")
            continue

        wave_b = data.loc[wave_b_idx].idxmax()
        wave['Wave_B'] = wave_b

        # --- 6) Wave C candidates: valleys after B matching cond1 or cond2 ---
        c_idx = data.index[valley_mask & (data.index > wave_b)]
        if len(c_idx) == 0:
            logger.debug(f"[{sp}] → [{wave_a}] → [{wave_b}] no Wave C valleys")
            continue

        cond1 = below_mean.loc[c_idx] & (data.loc[c_idx] < data.loc[wave_a])

        sp_val = float(data.loc[sp])
        a_val  = float(data.loc[wave_a])
        fib_ext = calculate_extension_levels(sp_val, a_val).values

        prices = data.loc[c_idx].values
        cond2 = np.array([np.any(p < fib_ext) for p in prices], dtype=bool)

        mask_c = cond1 | pd.Series(cond2, index=c_idx)
        valid_c_idx = c_idx[mask_c]

        if len(valid_c_idx) == 0:
            logger.debug(f"[{sp}] → [{wave_a}] → [{wave_b}] no valid Wave C")
            continue

        wave_c = data.loc[valid_c_idx].idxmin()
        wave['Wave_C'] = wave_c

        waves.append(wave)

    return pd.DataFrame(waves).dropna(how='all')
