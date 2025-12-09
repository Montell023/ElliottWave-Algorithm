import pandas as pd

def corrective(peaks, valleys):
    # Ensure peaks and valleys are pandas Series
    if not isinstance(peaks, pd.Series) or not isinstance(valleys, pd.Series):
        raise ValueError("Peaks and valleys must be pandas Series")

    # Initialize classifications as a pandas Series with 'A' for all indices
    classifications = pd.Series('A', index=valleys.index)

    for i in range(1, len(valleys)):
        if valleys.iloc[i] < peaks.iloc[i - 1]:
            classifications.iloc[i] = 'A'
        elif peaks.iloc[i] > valleys.iloc[i] > valleys.iloc[i - 1]:
            classifications.iloc[i] = 'B'
        else:
            classifications.iloc[i] = 'C'

    return classifications
