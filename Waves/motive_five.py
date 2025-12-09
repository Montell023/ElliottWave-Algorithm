import pandas as pd

def subdivides(peaks, valleys):
    # Ensure peaks and valleys are pandas Series
    if not isinstance(peaks, pd.Series) or not isinstance(valleys, pd.Series):
        raise ValueError("Peaks and valleys must be pandas Series")

    waves = {
        'Wave 1': None,
        'Wave 2': None,
        'Wave 3': None,
        'Wave 4': None,
        'Wave 5': None
    }

    for i in range(1, len(peaks)):
        if waves['Wave 1'] is None and valleys.iloc[i - 1] is not None and peaks.iloc[i] > valleys.iloc[i - 1]:
            waves['Wave 1'] = peaks.iloc[i]
        elif waves['Wave 1'] is not None and waves['Wave 2'] is None and valleys.iloc[i] < waves['Wave 1'] and valleys.iloc[i] > valleys.iloc[i - 1]:
            waves['Wave 2'] = valleys.iloc[i]
        elif waves['Wave 2'] is not None and waves['Wave 3'] is None and (peaks.iloc[i] > waves['Wave 1'] * 1.618 or peaks.iloc[i] > waves['Wave 1'] * 2.618 or peaks.iloc[i] > waves['Wave 1'] * 4.236):
            waves['Wave 3'] = peaks.iloc[i]
        elif waves['Wave 3'] is not None and waves['Wave 4'] is None and valleys.iloc[i] < waves['Wave 3'] and valleys.iloc[i] > waves['Wave 1']:
            waves['Wave 4'] = valleys.iloc[i]
        elif waves['Wave 4'] is not None and waves['Wave 5'] is None and peaks.iloc[i] > waves['Wave 3']:
            waves['Wave 5'] = peaks.iloc[i]
        elif waves['Wave 4'] is not None and waves['Wave 5'] is None:
            expected_wave_5 = waves['Wave 4'] + (waves['Wave 1'] - valleys.iloc[i - 1])
            if peaks.iloc[i] >= expected_wave_5:
                waves['Wave 5'] = peaks.iloc[i]

    return pd.Series(waves)
