import pandas as pd

def corrective(data, peak_mask, valley_mask):
    """
    Identifies “corrective” valleys (Wave A candidates) in a little down–up–down mini-pattern:
      • Find valley V  
      • Find the last peak P before V  
      • If V is lower than P, mark V as a corrective valley (True)  
    Returns a boolean Series on data.index.
    """
    # Get the close series
    if isinstance(data, pd.DataFrame):
        data = data['close']
    assert isinstance(data, pd.Series)
    
    # Prepare an all-False mask over every timestamp
    mask = pd.Series(False, index=data.index)
    
    # Extract the actual times of peaks & valleys
    peaks   = data.index[peak_mask]
    valleys = data.index[valley_mask]
    print(f"[corrective] peaks={len(peaks)}, valleys={len(valleys)}")
    
    # We need at least one peak before a valley to call it corrective
    if len(peaks) == 0 or len(valleys) < 1:
        return mask
    
    # For each valley, see if it sits below its preceding peak
    for v in valleys:
        # find the last peak before this valley
        prev_peaks = peaks[peaks < v]
        if len(prev_peaks) == 0:
            continue
        last_peak = prev_peaks[-1]
        
        # if the valley price is lower than that peak price, it's a corrective valley
        if data.loc[v] < data.loc[last_peak]:
            mask.loc[v] = True
    
    return mask
