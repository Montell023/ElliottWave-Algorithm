import pandas as pd

# Reading CSV file
def read_csv_file(filename):
    try:
        data = pd.read_csv(filename, parse_dates=['Datetime'])
        print(f"Successfully read {filename}")
        return data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Calculating Fibonacci levels
def calculate_fibonacci_levels(start, end):
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    fib_retracements = {level: start + (end - start) * level for level in fib_levels}
    return fib_retracements

def calculate_extension_levels(start, end):
    fib_levels = [1.618, 2.618, 4.236]
    fib_extensions = {level: start + (end - start) * level for level in fib_levels}
    return fib_extensions

# Strategy functions
def corrective(peaks, valleys):
    return [(valleys.iloc[i], peaks.iloc[i + 1], valleys.iloc[i + 2])
            for i in range(len(valleys) - 2)
            if valleys.iloc[i] < valleys.iloc[i + 2] and peaks.iloc[i + 1] > valleys.iloc[i]]

def subdivides_downtrend(peaks, valleys):
    return [(peaks.iloc[i], valleys.iloc[i + 1], peaks.iloc[i + 2])
            for i in range(len(peaks) - 2)
            if peaks.iloc[i] > peaks.iloc[i + 2] and valleys.iloc[i + 1] < peaks.iloc[i + 2]]

def motive(peaks, valleys):
    return [(peaks.iloc[i], valleys.iloc[i + 1], peaks.iloc[i + 2])
            for i in range(len(peaks) - 2)
            if peaks.iloc[i] < peaks.iloc[i + 2] and valleys.iloc[i + 1] > valleys.iloc[i + 2]]

def subdivides(peaks, valleys):
    return [(valleys.iloc[i], peaks.iloc[i + 1], valleys.iloc[i + 2])
            for i in range(len(valleys) - 2)
            if valleys.iloc[i] < valleys.iloc[i + 2] and peaks.iloc[i + 1] > valleys.iloc[i]]

def correction(peaks, valleys, mean_price):
    waves = {'Wave A': None, 'Wave B': None, 'Wave C': None}

    if peaks.empty or valleys.empty:
        return waves

    starting_point = valleys.iloc[0]

    wave_a_candidates = peaks[(peaks > mean_price) & (peaks > starting_point)]
    if not wave_a_candidates.empty:
        waves['Wave A'] = wave_a_candidates.iloc[-1]
    else:
        return waves

    valley_candidates = valleys[(valleys < waves['Wave A']) & (valleys > starting_point)]
    if not valley_candidates.empty:
        waves['Wave B'] = valley_candidates.iloc[-1]
    else:
        return waves

    wave_c_candidates = peaks[(peaks > mean_price) & (peaks > waves['Wave B'])]
    if not wave_c_candidates.empty:
        waves['Wave C'] = wave_c_candidates.iloc[-1]

    return waves

def correction_inverse(peaks, valleys, mean_price):
    waves = {'Wave A': None, 'Wave B': None, 'Wave C': None}

    if peaks.empty or valleys.empty:
        return waves

    starting_point = peaks.iloc[0]

    wave_a_candidates = valleys[(valleys < mean_price) & (valleys < starting_point)]
    if not wave_a_candidates.empty:
        waves['Wave A'] = wave_a_candidates.iloc[-1]
    else:
        return waves

    peak_candidates = peaks[(peaks > waves['Wave A']) & (peaks < starting_point)]
    if not peak_candidates.empty:
        waves['Wave B'] = peak_candidates.iloc[-1]
    else:
        return waves

    wave_c_candidates = valleys[(valleys < mean_price) & (valleys < waves['Wave B'])]
    if not wave_c_candidates.empty:
        waves['Wave C'] = wave_c_candidates.iloc[-1]

    return waves

def uptrend(peaks, valleys, mean_price):
    waves = []

    starting_point = valleys[valleys == valleys.rolling(window=20).min()].iloc[0]
    wave_1_candidates = peaks[(peaks > starting_point) & (peaks > mean_price)]
    if not wave_1_candidates.empty:
        wave_1 = wave_1_candidates.iloc[-1]
    else:
        return waves

    wave_2_candidates = valleys[(valleys < wave_1) & (valleys > starting_point)]
    if not wave_2_candidates.empty:
        wave_2 = wave_2_candidates.iloc[-1]
    else:
        return waves

    wave_3_candidates = peaks[(peaks > wave_1 * 1.618) | (peaks > wave_1 * 2.618) | (peaks > wave_1 * 4.236)]
    if not wave_3_candidates.empty:
        wave_3 = wave_3_candidates.iloc[-1]
    else:
        return waves

    wave_4_candidates = valleys[(valleys < wave_3) & (valleys > wave_1)]
    if not wave_4_candidates.empty:
        wave_4 = wave_4_candidates.iloc[-1]
    else:
        return waves

    wave_5_candidates = peaks[(peaks > wave_3) & (peaks > mean_price)]
    if not wave_5_candidates.empty:
        wave_5 = wave_5_candidates.iloc[-1]
    else:
        return waves

    waves.extend(['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5'])
    return waves

def downtrend(peaks, valleys, mean_price):
    waves = []

    starting_point = peaks[peaks == peaks.rolling(window=20).max()].iloc[0]
    wave_1_candidates = valleys[(valleys < starting_point) & (valleys < mean_price)]
    if not wave_1_candidates.empty:
        wave_1 = wave_1_candidates.iloc[-1]
    else:
        return waves

    wave_2_candidates = peaks[(peaks > wave_1) & (peaks < starting_point)]
    if not wave_2_candidates.empty:
        wave_2 = wave_2_candidates.iloc[-1]
    else:
        return waves

    wave_3_candidates = valleys[(valleys < wave_1 * 1.618) | (valleys < wave_1 * 2.618) | (valleys < wave_1 * 4.236)]
    if not wave_3_candidates.empty:
        wave_3 = wave_3_candidates.iloc[-1]
    else:
        return waves

    wave_4_candidates = peaks[(peaks > wave_3) & (peaks < wave_1)]
    if not wave_4_candidates.empty:
        wave_4 = wave_4_candidates.iloc[-1]
    else:
        return waves

    wave_5_candidates = valleys[(valleys < wave_3) & (valleys < mean_price)]
    if not wave_5_candidates.empty:
        wave_5 = wave_5_candidates.iloc[-1]
    else:
        return waves

    waves.extend(['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5'])
    return waves

# Generating signals
def generate_signals(data, strategy_func, strategy_name):
    try:
        peaks = data['High']
        valleys = data['Low']
        mean_price = data['Adj Close'].rolling(window=20).mean()

        print(f"Running {strategy_name} strategy")
        signals = strategy_func(peaks, valleys, mean_price)
        print(f"Generated signals for {strategy_name}: {signals}")
        return signals
    except Exception as e:
        print(f"Error evaluating {strategy_name}: {e}")
        return []

# Main function
def main():
    filenames = ["ETH-USD_5minute_data.csv", "BTC-USD_5minute_data.csv"]
    strategies = {
        "Correction": correction,
        "Correction Inverse": correction_inverse,
        "Impulse": uptrend,
        "Impulse Inverse": downtrend
    }

    for filename in filenames:
        data = read_csv_file(filename)
        if data is not None:
            print(f"Evaluating {filename}...")
            buy_signals = []
            sell_signals = []

            for strategy_name, strategy_func in strategies.items():
                signals = generate_signals(data, strategy_func, strategy_name)
                buy_signals.extend([signal for signal in signals if signal == 'Buy'])
                sell_signals.extend([signal for signal in signals if signal == 'Sell'])

            print(f"Buy Signals for {filename}: {buy_signals}")
            print(f"Sell Signals for {filename}: {sell_signals}")

if __name__ == "__main__":
    main()
