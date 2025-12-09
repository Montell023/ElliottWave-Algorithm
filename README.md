# Elliott Wave Validator & Wave Detection System

##  Project Overview

The Elliott Wave Validator is a sophisticated algorithmic trading system designed to identify and validate Elliott Wave patterns in financial time series data. Unlike simplistic pattern recognition systems, this implementation uses **structural analysis** and **graph traversal algorithms** to filter noise and identify high-probability wave patterns with strict validation rules.

##  System Architecture

### Core Components

1. **`waves_manager.py`** - **Main Elliott Wave Detection Engine**
   - Implements 4 specialized algorithms for wave pattern detection:
     - `motive_five()` - 5-wave impulse patterns (bullish)
     - `corrective_five()` - 5-wave corrective patterns (bearish)
     - `motive_abc()` - ABC motive patterns
     - `corrective_abc()` - ABC corrective patterns
   - Uses **DFS (Depth-First Search)** sequential processing
   - Implements **superposition logic** to resolve multiple pattern possibilities
   - Strict time and price constraints ensure high-quality detections

2. **`core/data_manager.py`** - **Data Processing & Structural Analysis**
   - Manages raw price data and peak/valley detection
   - Integrates with `real_time_peak_detector.py` for initial peak/valley identification
   - Implements **graph traversal algorithms** to find structural turns
   - Provides "quality starting points" by filtering noise from raw peaks/valleys
   - DFS-based wave family termination detection (furthest node/orphan node analogy)

3. **`core/real_time_peak_detector.py`** - **Peak/Valley Detection**
   - Uses SciPy's signal processing for initial peak detection
   - Configurable prominence and distance thresholds
   - Provides raw turning points for further structural analysis

## Key Innovations

### 1. **Structural Turn Detection**
Instead of analyzing every peak and valley (which leads to noise and false signals), the system:
- Uses DFS to identify **structural turning points**
- Filters out minor fluctuations and noise
- Focuses only on significant price movements that matter for Elliott Wave analysis

### 2. **Graph Traversal Algorithms**
- **DFS Depth-First Search**: Traverses price movements to find the "furthest cousin" (most significant termination point)
- **Wave Family Analysis**: Treats related price movements as family trees
- **Orphan Node Handling**: Special cases where waves have minimal noise

### 3. **Strict Validation Rules**
Each wave pattern must satisfy:
- **Time constraints**: Maximum durations for each wave (e.g., Wave 1 ≤ 2.0h, Wave 2 ≤ 1.5h)
- **Price relationships**: Fibonacci ratios, wave overlaps, and progression rules
- **Structural integrity**: Proper retracement levels and wave sequencing

## 📈 Pattern Detection Logic

### Five-Wave Patterns (Impulse/Corrective)
Motive 5-wave: 1-2-3-4-5 (bullish impulse)
Corrective 5-wave: 1-2-3-4-5 (bearish correction)

### ABC Patterns
Motive ABC: A-B-C (bullish correction in uptrend)
Corrective ABC: A-B-C (bearish correction in downtrend)


## Technical Implementation

### Algorithm Workflow
1. **Data Preparation**: Load and clean 1-minute OHLC data
2. **Peak/Valley Detection**: Initial turning point identification
3. **Structural Filtering**: DFS analysis to find quality starting points
4. **Pattern Detection**: Parallel evaluation of all 4 algorithms
5. **Superposition Resolution**: Priority-based pattern selection
6. **Validation**: Strict Elliott Wave rule checking
7. **Output**: Labeled wave points and pattern statistics

### Configuration Parameters
```python
# Time constraints (for 1-minute data)
WAVE_1_MAX_HOURS = 2.0
WAVE_2_MAX_HOURS = 1.5  
WAVE_3_MAX_HOURS = 2.5
WAVE_4_MAX_HOURS = 1.5
WAVE_5_MAX_HOURS = 2.0

# Price constraints
WAVE_3_MIN_PERCENT_ABOVE_W1 = 0.005  # 0.005%
WAVE_4_NO_OVERLAP_W1 = True
FIBONACCI_RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
```

###Data Requirements
Input Format
Frequency: 1-minute intervals (optimal for detailed wave analysis)

Columns Required: open, high, low, close, volume

Time Period: Minimum 2-3 days for meaningful pattern detection

Example Dataset: BTC-USD_1minute_data_cleaned.csv (4233 bars ≈ 2.9 days)

Data Sources
Local CSV files (included in repository for testing)

Yahoo Finance API (planned integration)

Custom data providers (extensible architecture)

###Integration with Backtesting Frameworks
The system is designed to work with:

LumiBot backtester

Backtrader

Zipline

Custom backtesting environments

##References
###Elliott Wave Theory
-Prechter, R. R., & Frost, A. J. (2005). Elliott Wave Principle

-Neely, G. (1990). Mastering Elliott Wave

-Koen van Ginneken, Profitability of Elliott Waves and Fibonacci Retracement Levels in the Foreign Exchange Market

##Disclaimer
###FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY

This software is provided for academic research and educational purposes. Past performance does not guarantee future results. Always conduct thorough backtesting and risk management before using any trading system with real capital.

##Performance Summary
For the 2.9-day BTC/USD 1-minute dataset:

-4233 data points analyzed

-Quality-first approach with DFS filtering

-Strict validation ensuring only high-confidence patterns

-27 wave points detected across 4 algorithms (appropriate density for strict rules)

-Balanced detection between motive and corrective structures

##Conclusion
This software was built to identify elliott waves in real-time. Elliott waves as mentioned in the book  Elliott Wave Principle is usually only found in retrospect. There is 
a process of validators which takes place and a set of pre-determined rules that is immutable. The conditions found in waves_manager.py are immutable rules hence why I use 
boolean operations because its either true or false and no in-between. The peaks and valleys for our data can be noisy so we ensured that our peaks and valleys prominence is adaptive. 
We then have our depth-first-search algorithm implemented which either find the furthest cousin peak or valley which helps us find the most likely termination peak or valley candidate for our waves and this helps us find the wave patterns and makes our validator algorithms oblivious to noisy peaks and valleys as we only want our system to analyse the structural turns and not noisy peaks and valleys. The system helps find waves in real-time and eliminates the retrospect wave identification process. Parameters can be adjusted so that the system can find either high quantity but low quality waves or low quantity but high quality waves. 






