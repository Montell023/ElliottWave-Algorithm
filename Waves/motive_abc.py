import pandas as pd

def motive(peaks, valleys):
    # Ensure peaks and valleys are pandas Series
    if not isinstance(peaks, pd.Series) or not isinstance(valleys, pd.Series):
        raise ValueError("Peaks and valleys must be pandas Series")

    classifications = ['A']  # Initialize with 'A'

    for i in range(1, len(peaks)):
        if peaks.iloc[i] > valleys.iloc[i - 1]:
            classifications.append('A')
        elif valleys.iloc[i] < peaks.iloc[i] < peaks.iloc[i - 1]:
            classifications.append('B')
        elif peaks.iloc[i] > peaks.iloc[i - 1] and valleys.iloc[i] > valleys.iloc[i - 1]:
            classifications.append('C')
        else:
            classifications.append('Unknown')

    return pd.Series(classifications, index=peaks.index)
