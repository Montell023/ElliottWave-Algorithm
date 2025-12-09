import pandas as pd

def motive(data, peak_mask, valley_mask):
    """
    “Mountain”-style subdivide: finds little up–down–up shapes (peak A → valley B → peak C).
    Marks each valley B where:
      • there is a peak A just before the valley,
      • and a peak C afterward that is higher than peak A.
    
    Returns a boolean Series on data.index: True at valley-B times.
    """
    # pull out close prices
    if isinstance(data, pd.DataFrame):
        data = data['close']
    assert isinstance(data, pd.Series)
    
    # boolean mask over full index
    result = pd.Series(False, index=data.index)
    
    # get the actual timestamps of peaks & valleys
    peaks   = data.index[peak_mask]
    valleys = data.index[valley_mask]
    
    # for each valley, see if it sits between a rising peak A → peak C
    for v in valleys:
        # 1) find the last peak before this valley
        prev_peaks = peaks[peaks < v]
        if len(prev_peaks) == 0:
            continue
        pA = prev_peaks[-1]
        
        # 2) find any peak after the valley that is higher than pA
        next_peaks = peaks[peaks > v]
        higher_after = [p for p in next_peaks if data[p] > data[pA]]
        if not higher_after:
            continue
        
        # we found a mini mountain: peak-A → valley-v → peak-C
        # mark this valley as a valid “Wave B” subdivide
        result.loc[v] = True
    
    return result
