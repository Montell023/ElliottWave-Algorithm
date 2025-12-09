# algostack/backtesting/elliott_wave_backtester.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Lumibot imports
from lumibot.backtesting import PandasDataBacktesting, BacktestingBroker
from lumibot.entities import Asset, Data
from lumibot.strategies import Strategy
from lumibot.traders import Trader

# Import your enhanced Elliott Wave system
try:
    from core.data_manager import DataManager
    from trends_manager.trends_manager import TrendsManager
    from core.real_time_peak_detector import RealTimePeakDetector
    from core.fibonacci_calculator import FibonacciCalculator
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Import error: {e}")
    # Create minimal fallbacks for critical components
    pass

class EnhancedElliottWaveStrategy(Strategy):
    """
    QUANT-GRADE Elliott Wave Strategy with Multi-Factor Signal Enhancement
    Integrates TrendsManager + Volume Confirmation + Trend Alignment + Advanced Risk Management
    """
    
    def initialize(self):
        """Initialize QUANT-GRADE strategy with multi-factor signal enhancement"""
        # Trading parameters
        self.symbol = 'BTC-USD'
        self.base_position_size = 0.02  # 2% base position size
        self.max_position_size = 0.10   # 10% maximum position size
        
        # 🎯 OPTIMIZED Confidence thresholds - BALANCED approach
        self.confidence_thresholds = {
            'high_quality': 0.60,    # >60% - Aggressive trading (was 65%)
            'medium_quality': 0.52,  # 52-60% - Standard trading (was 55%)  
            'low_quality': 0.50,     # 50-52% - Conservative only
            'minimum': 0.50          # <50% - No trading
        }
        
        # 🎯 ENHANCED RISK MANAGEMENT
        self.risk_params = {
            'stop_loss_pct': 0.02,        # 2% stop loss per trade
            'take_profit_ratio': 3.0,     # 1:3 risk-reward ratio (IMPROVED from 2.0)
            'max_daily_loss': 0.05,       # 5% maximum daily loss
            'max_simultaneous_trades': 3, # ALLOW 3 TRADES (NEW)
            'max_portfolio_exposure': 0.15, # 15% total exposure (NEW)
            'min_trade_spacing': 30,      # 30 minutes between signals (NEW)
            'use_fibonacci_targets': True,
            'fibonacci_extension': 1.618
        }
        
        # 🎯 ENHANCED TRACKING
        self.current_position = 0
        self.last_signal = None
        self.trade_log = []
        self.active_trades = {}
        self.daily_pnl = 0
        self.last_trade_date = None
        self.last_trade_time = None  # NEW: Track last trade time
        
        # 🎯 COMPREHENSIVE ANALYTICS
        self.trade_analytics = {
            'timestamp': [], 'action': [], 'quantity': [], 'entry_price': [],
            'exit_price': [], 'exit_type': [], 'pnl': [], 'pnl_percent': [],
            'confidence': [], 'tier': [], 'pattern_type': [], 'hold_time': [],
            'win_loss': [], 'enhanced_score': []  # NEW: Enhanced signal score
        }
        
        # Initialize Elliott Wave system
        try:
            self._initialize_elliott_wave_system()
            self.log_message("✅ Elliott Wave System Initialized Successfully")
        except Exception as e:
            self.log_message(f"⚠️ Elliott Wave system warning: {e}")
            self.data_manager = None
            self.trends_manager = None
        
        self.log_message(f"🚀 QUANT-GRADE Elliott Wave Strategy Initialized")
        self.log_message(f"Symbol: {self.symbol}")
        self.log_message(f"🎯 OPTIMIZED Confidence Thresholds: {self.confidence_thresholds}")
        self.log_message(f"💰 ENHANCED Risk Management: {self.risk_params}")

    def _initialize_elliott_wave_system(self):
        """Initialize the complete Elliott Wave detection system"""
        try:
            self.data_manager = DataManager(
                mode='backtest', window_size=200, ma_length=20,
                bb_multiplier=2.0, peak_prominence=0.5
            )
            
            self.trends_manager = TrendsManager(
                data_manager=self.data_manager, pattern_history_size=100, debug_mode=False
            )
            
        except Exception as e:
            self.log_message(f"❌ Error in Elliott Wave system initialization: {e}")
            raise

    # 🎯 NEW: VOLUME CONFIRMATION METHOD
    def _get_volume_confirmation(self) -> float:
        """Add volume confirmation to signal quality (20% weight)"""
        try:
            bars = self.get_historical_prices(self.symbol, 20)
            if bars is None or bars.df.empty:
                return 0.5  # Neutral if no data
                
            volumes = bars.df['volume'].tail(10)
            if len(volumes) < 5:
                return 0.5
                
            avg_volume = volumes.mean()
            current_volume = volumes.iloc[-1]
            
            # Volume confirmation scoring
            if current_volume > avg_volume * 1.3:
                return 0.9  # Strong confirmation
            elif current_volume > avg_volume * 1.1:
                return 0.7  # Good confirmation
            elif current_volume > avg_volume * 0.8:
                return 0.5  # Neutral
            else:
                return 0.3  # Weak confirmation
                
        except Exception as e:
            self.log_message(f"⚠️ Volume confirmation error: {e}")
            return 0.5

    # 🎯 NEW: TREND ALIGNMENT METHOD  
    def _get_trend_alignment(self) -> float:
        """Add trend alignment confirmation (30% weight)"""
        try:
            bars = self.get_historical_prices(self.symbol, 50)
            if bars is None or bars.df.empty:
                return 0.5
                
            prices = bars.df['close']
            
            # Calculate short and long term trends
            short_ma = prices.tail(10).mean()
            long_ma = prices.tail(30).mean()
            current_price = prices.iloc[-1]
            
            # Trend strength calculation
            if current_price > short_ma > long_ma:
                return 0.8  # Strong uptrend
            elif current_price < short_ma < long_ma:
                return 0.8  # Strong downtrend
            elif (current_price > short_ma and short_ma > long_ma * 0.98):
                return 0.6  # Moderate uptrend
            elif (current_price < short_ma and short_ma < long_ma * 1.02):
                return 0.6  # Moderate downtrend
            else:
                return 0.4  # Weak or ranging trend
                
        except Exception as e:
            self.log_message(f"⚠️ Trend alignment error: {e}")
            return 0.5

    # 🎯 NEW: ENHANCED SIGNAL SCORING
    def _calculate_enhanced_signal_score(self, signals: Dict[str, Any]) -> float:
        """Multi-factor signal scoring for better quality control"""
        base_confidence = signals.get('enhanced_confidence', 0.0)
        
        # Volume confirmation (20% weight)
        volume_score = self._get_volume_confirmation()
        
        # Trend alignment (30% weight)  
        trend_score = self._get_trend_alignment()
        
        # Pattern clarity (50% weight)
        pattern_score = base_confidence
        
        # Combined multi-factor score
        enhanced_score = (pattern_score * 0.5 + trend_score * 0.3 + volume_score * 0.2)
        
        # Boost high-confidence signals slightly
        if base_confidence > 0.6:
            enhanced_score = min(enhanced_score * 1.1, 1.0)
            
        return round(enhanced_score, 3)

    def on_trading_iteration(self):
        """Enhanced trading iteration with multi-factor signals"""
        try:
            if self._exceeded_daily_loss_limit():
                self.log_message("🚫 Daily loss limit exceeded - stopping trading")
                return
            
            asset = Asset(symbol=self.symbol, asset_type=Asset.AssetType.CRYPTO)
            
            if self.data_manager is None or self.trends_manager is None:
                return
            
            success = self._update_market_data(asset)
            if not success:
                return
            
            # Manage active trades
            self._manage_active_trades(asset)
            
            # Get enhanced signals with multi-factor scoring
            signals = self._get_enhanced_trading_signals()
            
            # Execute risk-managed trading
            self._execute_risk_managed_trading(asset, signals)
            
            # Log enhanced state
            self._log_enhanced_trading_state(signals)
            
        except Exception as e:
            self.log_message(f"❌ Error in trading iteration: {e}")

    def _exceeded_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        current_date = self.get_datetime().date()
        
        if self.last_trade_date != current_date:
            self.daily_pnl = 0
            self.last_trade_date = current_date
        
        daily_loss_pct = abs(self.daily_pnl) / self.portfolio_value
        return daily_loss_pct >= self.risk_params['max_daily_loss']

    def _manage_active_trades(self, asset: Asset):
        """Manage stop-loss and take-profit for active trades"""
        current_price = self.get_last_price(asset)
        
        for trade_id, trade_info in list(self.active_trades.items()):
            entry_price = trade_info['entry_price']
            action = trade_info['action']
            
            # Calculate P&L percentage
            if action == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SELL
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop-loss (2%)
            if pnl_pct <= -self.risk_params['stop_loss_pct']:
                self._close_trade(asset, trade_id, 'stop_loss', current_price)
                continue
            
            # Check take-profit (6% - IMPROVED from 4%)
            take_profit_pct = self.risk_params['stop_loss_pct'] * self.risk_params['take_profit_ratio']
            if pnl_pct >= take_profit_pct:
                self._close_trade(asset, trade_id, 'take_profit', current_price)
                continue

    def _close_trade(self, asset: Asset, trade_id: str, exit_type: str, exit_price: float):
        """Close trade with enhanced analytics"""
        trade_info = self.active_trades[trade_id]
        
        entry_price = trade_info['entry_price']
        quantity = trade_info['quantity']
        action = trade_info['action']
        
        # Calculate P&L
        if action == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        pnl_pct = (pnl / (entry_price * quantity)) * 100
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Record enhanced analytics
        self.trade_analytics['timestamp'].append(self.get_datetime())
        self.trade_analytics['action'].append(action)
        self.trade_analytics['quantity'].append(quantity)
        self.trade_analytics['entry_price'].append(entry_price)
        self.trade_analytics['exit_price'].append(exit_price)
        self.trade_analytics['exit_type'].append(exit_type)
        self.trade_analytics['pnl'].append(pnl)
        self.trade_analytics['pnl_percent'].append(pnl_pct)
        self.trade_analytics['confidence'].append(trade_info['confidence'])
        self.trade_analytics['tier'].append(trade_info['tier'])
        self.trade_analytics['pattern_type'].append(trade_info.get('pattern_type', 'unknown'))
        self.trade_analytics['hold_time'].append(self.get_datetime() - trade_info['entry_time'])
        self.trade_analytics['win_loss'].append('WIN' if pnl > 0 else 'LOSS')
        self.trade_analytics['enhanced_score'].append(trade_info.get('enhanced_score', 0.0))
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        self.log_message(f"🔒 Trade Closed: {action} | {exit_type.upper()} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

    def _update_market_data(self, asset: Asset) -> bool:
        """Update market data"""
        try:
            if self.data_manager is None:
                return False
            
            bars = self.get_historical_prices(asset, 200)
            if bars is None or bars.df.empty:
                return False
            
            data_df = bars.df.copy()
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data_df.columns for col in required_columns):
                return False
            
            current_timestamp = data_df.index[-1] if not data_df.empty else None
            success = self.data_manager.update_data(data_df, current_timestamp=current_timestamp)
            
            if success:
                data_summary = self.data_manager.get_data_summary()
                if data_summary['status'] == 'ready':
                    detection_stats = self.data_manager.get_detection_stats()
                    peaks = detection_stats.get('peaks_detected', 0)
                    valleys = detection_stats.get('valleys_detected', 0)
                    self.log_message(f"📊 Data processed: {peaks} peaks, {valleys} valleys")
                    return True
            return False
                
        except Exception as e:
            self.log_message(f"❌ Error updating market data: {e}")
            return False

    def _get_enhanced_trading_signals(self) -> Dict[str, Any]:
        """Get enhanced trading signals with multi-factor scoring"""
        try:
            if self.trends_manager is None:
                return self._create_default_signals()
            
            patterns = self.trends_manager.detect_all_patterns()
            signals = self.trends_manager.get_trading_signals()
            enhanced_signals = self._enhance_signal_analysis(signals, patterns)
            
            # 🎯 ADD ENHANCED SIGNAL SCORING
            enhanced_score = self._calculate_enhanced_signal_score(enhanced_signals)
            enhanced_signals['enhanced_score'] = enhanced_score
            
            return enhanced_signals
            
        except Exception as e:
            self.log_message(f"❌ Error getting trading signals: {e}")
            return self._create_default_signals()

    def _enhance_signal_analysis(self, signals: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced signal analysis"""
        enhanced_signals = signals.copy()
        
        pattern_analysis = self._analyze_pattern_quality(patterns)
        enhanced_signals['pattern_analysis'] = pattern_analysis
        
        base_confidence = enhanced_signals.get('overall_confidence', 0.0)
        pattern_quality_boost = pattern_analysis.get('quality_boost', 0.0)
        
        enhanced_confidence = min(1.0, base_confidence + pattern_quality_boost)
        enhanced_signals['enhanced_confidence'] = enhanced_confidence
        
        # Trading tier determination
        if enhanced_confidence >= self.confidence_thresholds['high_quality']:
            trading_tier = 'high_quality'
            position_multiplier = 2.0
        elif enhanced_confidence >= self.confidence_thresholds['medium_quality']:
            trading_tier = 'medium_quality' 
            position_multiplier = 1.0
        elif enhanced_confidence >= self.confidence_thresholds['low_quality']:
            trading_tier = 'low_quality'
            position_multiplier = 0.5
        else:
            trading_tier = 'below_threshold'
            position_multiplier = 0.0
        
        enhanced_signals['trading_tier'] = trading_tier
        enhanced_signals['position_multiplier'] = position_multiplier
        enhanced_signals['should_trade'] = position_multiplier > 0
        
        return enhanced_signals

    def _analyze_pattern_quality(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pattern quality"""
        quality_metrics = {
            'total_patterns': 0, 'high_confidence_patterns': 0, 'average_confidence': 0.0,
            'pattern_variety': 0, 'quality_boost': 0.0
        }
        
        try:
            confidences = []
            pattern_types = []
            
            for pattern_type, pattern_data in patterns.items():
                if pattern_data is not None and len(pattern_data) > 0:
                    quality_metrics['total_patterns'] += len(pattern_data)
                    pattern_types.append(pattern_type)
                    
                    for _, pattern in pattern_data.iterrows():
                        confidence = pattern.get('confidence', 0.0)
                        confidences.append(confidence)
                        if confidence >= 0.5:
                            quality_metrics['high_confidence_patterns'] += 1
            
            if confidences:
                quality_metrics['average_confidence'] = sum(confidences) / len(confidences)
                quality_metrics['pattern_variety'] = len(set(pattern_types))
                
                confidence_boost = min(0.10, quality_metrics['average_confidence'] * 0.2)
                variety_boost = min(0.05, quality_metrics['pattern_variety'] * 0.025)
                quality_metrics['quality_boost'] = confidence_boost + variety_boost
        
        except Exception as e:
            self.log_message(f"⚠️ Error in pattern quality analysis: {e}")
        
        return quality_metrics

    def _execute_risk_managed_trading(self, asset: Asset, signals: Dict[str, Any]):
        """Execute trading with enhanced risk management"""
        if not signals.get('should_trade', False):
            self.log_message("⏸️ No trading - confidence below threshold")
            return
        
        # 🎯 ENHANCED: Allow multiple trades with limits
        current_time = self.get_datetime()
        
        # Check max simultaneous trades
        if len(self.active_trades) >= self.risk_params['max_simultaneous_trades']:
            self.log_message(f"⏸️ Max trades reached ({len(self.active_trades)}/{self.risk_params['max_simultaneous_trades']})")
            return
        
        # Check portfolio exposure
        current_exposure = sum(
            trade['quantity'] * self.get_last_price(asset) 
            for trade in self.active_trades.values()
        ) / self.portfolio_value
        
        if current_exposure >= self.risk_params['max_portfolio_exposure']:
            self.log_message(f"⏸️ Max exposure reached ({current_exposure:.1%})")
            return
        
        # Check trade spacing
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.risk_params['min_trade_spacing'] * 60:
            self.log_message("⏸️ Too soon since last trade")
            return
        
        trading_tier = signals.get('trading_tier')
        action = signals.get('weighted_action')
        position_multiplier = signals.get('position_multiplier', 1.0)
        enhanced_score = signals.get('enhanced_score', 0.0)
        
        self.log_message(f"🎯 Trading Decision: {action.upper()} ({trading_tier} tier)")
        self.log_message(f"📊 Enhanced Confidence: {signals['enhanced_confidence']:.1%}")
        self.log_message(f"⭐ Enhanced Score: {enhanced_score:.1%}")  # NEW
        self.log_message(f"📈 Position Multiplier: {position_multiplier}x")
        
        # Calculate position size
        base_size = self.base_position_size * position_multiplier
        position_size = min(base_size, self.max_position_size)
        
        # Execute trading
        if action == 'BUY' and self.current_position <= 0:
            self._execute_risk_managed_buy(asset, position_size, signals)
        elif action == 'SELL' and self.current_position >= 0:
            self._execute_risk_managed_sell(asset, position_size, signals)
        else:
            self.log_message(f"🔄 No action needed")

    def _execute_risk_managed_buy(self, asset: Asset, position_size: float, signals: Dict[str, Any]):
        """Execute buy order"""
        try:
            current_price = self.get_last_price(asset)
            quantity = (self.portfolio_value * position_size) / current_price
            
            order = self.create_order(asset, quantity, "buy")
            self.submit_order(order)
            
            self.current_position = quantity
            self.last_signal = 'BUY'
            self.last_trade_time = self.get_datetime()  # NEW
            
            trade_id = f"BUY_{len(self.active_trades) + 1}_{self.get_datetime().strftime('%Y%m%d_%H%M%S')}"
            
            # Record enhanced trade info
            self.active_trades[trade_id] = {
                'entry_price': current_price, 'quantity': quantity, 'action': 'BUY',
                'entry_time': self.get_datetime(), 'confidence': signals['enhanced_confidence'],
                'tier': signals['trading_tier'], 'pattern_type': signals.get('pattern_type', 'unknown'),
                'enhanced_score': signals.get('enhanced_score', 0.0)  # NEW
            }
            
            # Log trade
            self.trade_log.append({
                'timestamp': self.get_datetime(), 'action': 'BUY', 'quantity': quantity,
                'confidence': signals['enhanced_confidence'], 'tier': signals['trading_tier'],
                'price': current_price
            })
            
            self.log_message(f"✅ BUY Order: {quantity:.6f} {asset.symbol}")
            self.log_message(f"💰 Risk Management: 2% Stop-Loss | 6% Take-Profit Active")  # UPDATED
                           
        except Exception as e:
            self.log_message(f"❌ Error executing BUY order: {e}")

    def _execute_risk_managed_sell(self, asset: Asset, position_size: float, signals: Dict[str, Any]):
        """Execute sell order"""
        try:
            current_price = self.get_last_price(asset)
            
            if self.current_position > 0:
                quantity = min(self.current_position, 
                             (self.portfolio_value * position_size) / current_price)
            else:
                quantity = (self.portfolio_value * position_size) / current_price
            
            order = self.create_order(asset, quantity, "sell")
            self.submit_order(order)
            
            self.current_position -= quantity
            self.last_signal = 'SELL'
            self.last_trade_time = self.get_datetime()  # NEW
            
            trade_id = f"SELL_{len(self.active_trades) + 1}_{self.get_datetime().strftime('%Y%m%d_%H%M%S')}"
            
            # Record enhanced trade info
            self.active_trades[trade_id] = {
                'entry_price': current_price, 'quantity': quantity, 'action': 'SELL',
                'entry_time': self.get_datetime(), 'confidence': signals['enhanced_confidence'],
                'tier': signals['trading_tier'], 'pattern_type': signals.get('pattern_type', 'unknown'),
                'enhanced_score': signals.get('enhanced_score', 0.0)  # NEW
            }
            
            # Log trade
            self.trade_log.append({
                'timestamp': self.get_datetime(), 'action': 'SELL', 'quantity': quantity,
                'confidence': signals['enhanced_confidence'], 'tier': signals['trading_tier'],
                'price': current_price
            })
            
            self.log_message(f"✅ SELL Order: {quantity:.6f} {asset.symbol}")
            self.log_message(f"💰 Risk Management: 2% Stop-Loss | 6% Take-Profit Active")  # UPDATED
                           
        except Exception as e:
            self.log_message(f"❌ Error executing SELL order: {e}")

    def _log_enhanced_trading_state(self, signals: Dict[str, Any]):
        """Log enhanced trading state"""
        try:
            if self.data_manager is None:
                return
                
            detection_stats = self.data_manager.get_detection_stats()
            
            # Calculate enhanced analytics
            active_trades = len(self.active_trades)
            win_rate = self._calculate_win_rate()
            avg_trade_pnl = self._calculate_avg_trade_pnl()
            enhanced_score = signals.get('enhanced_score', 0.0)
            
            # Calculate portfolio exposure
            current_exposure = 0
            if active_trades > 0:
                asset = Asset(symbol=self.symbol, asset_type=Asset.AssetType.CRYPTO)
                current_price = self.get_last_price(asset)
                current_exposure = sum(
                    trade['quantity'] * current_price 
                    for trade in self.active_trades.values()
                ) / self.portfolio_value
            
            state_log = (
                f"\n📊 QUANT-GRADE TRADING STATE:\n"
                f"   Portfolio Value: ${self.portfolio_value:,.2f}\n"
                f"   Active Trades: {active_trades}/{self.risk_params['max_simultaneous_trades']}\n"
                f"   Portfolio Exposure: {current_exposure:.1%}\n"
                f"   Daily P&L: ${self.daily_pnl:+.2f}\n"
                f"   Enhanced Score: {enhanced_score:.1%}\n"
                f"   Win Rate: {win_rate:.1%}\n"
                f"   Avg Trade P&L: {avg_trade_pnl:+.2f}%\n"
                f"   Patterns: {detection_stats.get('peaks_detected', 0)} peaks, "
                f"{detection_stats.get('valleys_detected', 0)} valleys\n"
            )
            
            self.log_message(state_log)
            
        except Exception as e:
            self.log_message(f"⚠️ Error logging trading state: {e}")

    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trade_analytics['win_loss']:
            return 0.0
        wins = sum(1 for result in self.trade_analytics['win_loss'] if result == 'WIN')
        return wins / len(self.trade_analytics['win_loss'])

    def _calculate_avg_trade_pnl(self) -> float:
        """Calculate average trade P&L"""
        if not self.trade_analytics['pnl_percent']:
            return 0.0
        return sum(self.trade_analytics['pnl_percent']) / len(self.trade_analytics['pnl_percent'])

    def _create_default_signals(self) -> Dict[str, Any]:
        """Create default signals"""
        return {
            'weighted_action': 'HOLD', 'overall_confidence': 0.0, 'enhanced_confidence': 0.0,
            'trading_tier': 'below_threshold', 'position_multiplier': 0.0, 'should_trade': False,
            'patterns_detected': 0, 'pattern_analysis': {}, 'enhanced_score': 0.0
        }

    def log_message(self, message: str, color: str = None):
        """Enhanced logging"""
        timestamp = self.get_datetime() if hasattr(self, 'get_datetime') else pd.Timestamp.now()
        formatted_message = f"{timestamp} - {message}"
        print(formatted_message)
        
        if not hasattr(self, 'strategy_logs'):
            self.strategy_logs = []
        self.strategy_logs.append(formatted_message)

    def print_performance_report(self):
        """Print QUANT-GRADE performance report"""
        print("\n" + "="*80)
        print("🎯 QUANT-GRADE ELLIOTT WAVE STRATEGY - PERFORMANCE REPORT")
        print("="*80)
        
        # Basic performance
        initial_value = 100000
        final_value = self.portfolio_value
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"📈 Portfolio Performance:")
        print(f"   Initial Value: ${initial_value:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        
        # Trading activity
        total_trades = len(self.trade_analytics['timestamp'])
        buy_trades = len([t for t in self.trade_log if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trade_log if t['action'] == 'SELL'])
        
        print(f"\n📊 Trading Activity:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Buy Trades: {buy_trades}")
        print(f"   Sell Trades: {sell_trades}")
        print(f"   Max Simultaneous Trades: {self.risk_params['max_simultaneous_trades']}")
        
        # Enhanced performance analytics
        if total_trades > 0:
            # Win/Loss Analysis
            wins = sum(1 for result in self.trade_analytics['win_loss'] if result == 'WIN')
            losses = total_trades - wins
            win_rate = (wins / total_trades) * 100
            
            # P&L Analysis
            total_pnl = sum(self.trade_analytics['pnl'])
            avg_pnl = total_pnl / total_trades
            avg_pnl_pct = sum(self.trade_analytics['pnl_percent']) / total_trades
            
            # Risk Analysis
            winning_trades = [pnl for pnl, result in zip(self.trade_analytics['pnl'], self.trade_analytics['win_loss']) if result == 'WIN']
            losing_trades = [pnl for pnl, result in zip(self.trade_analytics['pnl'], self.trade_analytics['win_loss']) if result == 'LOSS']
            
            avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            print(f"\n🎯 QUANT-GRADE ANALYTICS:")
            print(f"   Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
            print(f"   Total P&L: ${total_pnl:+.2f}")
            print(f"   Average P&L per Trade: ${avg_pnl:+.2f} ({avg_pnl_pct:+.2f}%)")
            print(f"   Average Win: ${avg_win:+.2f}")
            print(f"   Average Loss: ${avg_loss:+.2f}")
            print(f"   Risk/Reward Ratio: {risk_reward:.2f}:1")
            
            # Enhanced Score Analysis (NEW)
            if self.trade_analytics['enhanced_score']:
                avg_enhanced_score = sum(self.trade_analytics['enhanced_score']) / len(self.trade_analytics['enhanced_score'])
                high_quality_trades = len([s for s in self.trade_analytics['enhanced_score'] if s >= 0.6])
                print(f"   Average Enhanced Score: {avg_enhanced_score:.1%}")
                print(f"   High Quality Trades (>60%): {high_quality_trades}/{total_trades}")
            
            # Exit Type Analysis
            stop_losses = self.trade_analytics['exit_type'].count('stop_loss')
            take_profits = self.trade_analytics['exit_type'].count('take_profit')
            signal_closes = self.trade_analytics['exit_type'].count('signal_close')
            
            print(f"\n🔒 EXIT ANALYSIS:")
            print(f"   Stop-Loss Exits: {stop_losses}")
            print(f"   Take-Profit Exits: {take_profits} (6% target)")  # UPDATED
            print(f"   Signal Closes: {signal_closes}")
            
            # Enhanced Confidence Analysis
            high_conf_trades = len([c for c in self.trade_analytics['confidence'] if c >= 0.60])
            medium_conf_trades = len([c for c in self.trade_analytics['confidence'] if 0.52 <= c < 0.60])
            low_conf_trades = len([c for c in self.trade_analytics['confidence'] if 0.50 <= c < 0.52])
            
            print(f"\n🎯 ENHANCED CONFIDENCE ANALYSIS:")
            print(f"   High Confidence (>60%): {high_conf_trades} trades")
            print(f"   Medium Confidence (52-60%): {medium_conf_trades} trades")
            print(f"   Low Confidence (50-52%): {low_conf_trades} trades")
            
            # Calculate enhanced win rates
            if high_conf_trades > 0:
                high_conf_wins = sum(1 for i, conf in enumerate(self.trade_analytics['confidence']) 
                                   if conf >= 0.60 and self.trade_analytics['win_loss'][i] == 'WIN')
                high_conf_win_rate = (high_conf_wins / high_conf_trades * 100)
                print(f"   High Confidence Win Rate: {high_conf_win_rate:.1f}%")
        
        # Elliott Wave detection
        try:
            if self.data_manager is not None:
                detection_stats = self.data_manager.get_detection_stats()
                print(f"\n🔍 Elliott Wave Detection:")
                print(f"   Data Points: {detection_stats.get('data_points', 0)}")
                print(f"   Peaks Detected: {detection_stats.get('peaks_detected', 0)}")
                print(f"   Valleys Detected: {detection_stats.get('valleys_detected', 0)}")
        except Exception as e:
            print(f"\n⚠️ Detection stats error: {e}")
        
        print("="*80)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare CSV data for backtesting
    """
    print(f"📁 Loading data from: {file_path}")
    
    try:
        # Load CSV data
        df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
        
        # Ensure timezone awareness
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/New_York')
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ Data loaded successfully:")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def run_enhanced_backtest(csv_file_path: str, initial_budget: float = 100000):
    """
    Run enhanced Elliott Wave backtest with CSV data
    """
    print("🚀 STARTING ENHANCED ELLIOTT WAVE BACKTEST")
    print("="*80)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(csv_file_path)
        
        # Create assets
        base_asset = Asset(symbol="BTC-USD", asset_type=Asset.AssetType.CRYPTO)
        quote_asset = Asset(symbol="USD", asset_type=Asset.AssetType.FOREX)
        
        # Create pandas data for Lumibot
        pandas_data = {
            base_asset: Data(
                base_asset,
                df,
                timestep="minute",
                quote=quote_asset
            )
        }
        
        # Set backtesting period
        backtesting_start = df.index.min()
        backtesting_end = df.index.max()
        
        print(f"\n📅 Backtesting Period:")
        print(f"   Start: {backtesting_start}")
        print(f"   End: {backtesting_end}")
        print(f"   Total Bars: {len(df)}")
        
        # Initialize and run backtest
        trader = Trader(backtest=True)
        
        data_source = PandasDataBacktesting(
            pandas_data=pandas_data,
            datetime_start=backtesting_start,
            datetime_end=backtesting_end
        )
        
        broker = BacktestingBroker(data_source)
        
        strategy = EnhancedElliottWaveStrategy(
            broker=broker,
            budget=initial_budget,
            timestamp=backtesting_start
        )
        
        # Add strategy and run
        trader.add_strategy(strategy)
        
        print(f"\n🎯 Running Enhanced Elliott Wave Backtest...")
        trader.run_all(show_plot=True, show_tearsheet=True)
        
        # Print enhanced performance report
        strategy.print_performance_report()
        
        print("✅ ENHANCED BACKTEST COMPLETED SUCCESSFULLY!")
        
        return strategy
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        raise

# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File path to your CSV data
    CSV_FILE_PATH = 'BTC-USD_1minute_data_cleaned.csv'  
    
    # Run enhanced backtest
    try:
        strategy = run_enhanced_backtest(
            csv_file_path=CSV_FILE_PATH,
            initial_budget=100000
        )
        
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")