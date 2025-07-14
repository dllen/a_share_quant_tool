"""
Turtle Trading Strategy Implementation

This module implements the classic Turtle Trading System developed by Richard Dennis and William Eckhardt.
The strategy includes both the original System 1 (20/10 day) and System 2 (55/20 day) trading rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

class TurtleTradingStrategy:
    """
    Implementation of the Turtle Trading System.
    
    The strategy uses two systems:
    - System 1: 20-day breakout for entry, 10-day breakout for exit
    - System 2: 55-day breakout for entry, 20-day breakout for exit
    
    Position sizing is based on the Average True Range (ATR) and account equity.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.01,
                 max_units_per_market: int = 4,
                 atr_period: int = 20,
                 system1_entry: int = 20,
                 system1_exit: int = 10,
                 system2_entry: int = 55,
                 system2_exit: int = 20):
        """
        Initialize the Turtle Trading Strategy.
        
        Args:
            initial_capital: Starting capital for the strategy
            risk_per_trade: Risk per trade as a fraction of account equity (default: 1%)
            max_units_per_market: Maximum number of units to hold per market (default: 4)
            atr_period: Period for ATR calculation (default: 20)
            system1_entry: Lookback period for System 1 entry (default: 20)
            system1_exit: Lookback period for System 1 exit (default: 10)
            system2_entry: Lookback period for System 2 entry (default: 55)
            system2_exit: Lookback period for System 2 exit (default: 20)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_units_per_market = max_units_per_market
        self.atr_period = atr_period
        self.system1_entry = system1_entry
        self.system1_exit = system1_exit
        self.system2_entry = system2_entry
        self.system2_exit = system2_exit
        
        # State variables
        self.equity = [initial_capital]
        self.positions = {}
        self.trades = []
        self.signals = []
    
    def _flatten_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns in a DataFrame to single level.
        
        Args:
            data: DataFrame with potentially MultiIndex columns
            
        Returns:
            DataFrame with flattened column names
        """
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns by joining levels with underscore
            data.columns = ['_'.join(col).strip('_').lower() for col in data.columns.values]
        return data
    
    def _get_column(self, data: pd.DataFrame, col_name: str) -> pd.Series:
        """
        Get column from DataFrame, handling different column name formats.
        
        Args:
            data: DataFrame containing the data
            col_name: Base column name to get (e.g., 'close', 'high')
            
        Returns:
            Series containing the requested column data
            
        Raises:
            KeyError: If column cannot be found in any expected format
        """
        # List of possible column name variations to try
        variations = [
            col_name.lower(),
            col_name.title(),
            f'adj {col_name}'.lower(),
            f'adj_{col_name}'.lower(),
            f'{col_name}_spy'.lower(),
            f'adj {col_name}_spy'.lower()
        ]
        
        # Try each variation
        for var in variations:
            if var in data.columns:
                return data[var]
        
        # If we get here, no variation was found
        raise KeyError(f"Could not find column matching '{col_name}' in {data.columns.tolist()}")
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for the given price data.
        
        Args:
            data: DataFrame containing 'high', 'low', 'close' columns (case insensitive)
            
        Returns:
            Series containing ATR values
        """
        high = self._get_column(data, 'high')
        low = self._get_column(data, 'low')
        close = self._get_column(data, 'close')
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_breakouts(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate breakout levels for both systems.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added columns for breakout levels
        """
        data = data.copy()
        
        # Get column references
        high = self._get_column(data, 'high')
        low = self._get_column(data, 'low')
        
        # System 1 (20/10 day)
        data['sys1_high'] = high.rolling(window=self.system1_entry).max().shift(1)
        data['sys1_low'] = low.rolling(window=self.system1_entry).min().shift(1)
        data['sys1_exit_high'] = high.rolling(window=self.system1_exit).max().shift(1)
        data['sys1_exit_low'] = low.rolling(window=self.system1_exit).min().shift(1)
        
        # System 2 (55/20 day)
        data['sys2_high'] = high.rolling(window=self.system2_entry).max().shift(1)
        data['sys2_low'] = low.rolling(window=self.system2_entry).min().shift(1)
        data['sys2_exit_high'] = high.rolling(window=self.system2_exit).max().shift(1)
        data['sys2_exit_low'] = low.rolling(window=self.system2_exit).min().shift(1)
        
        return data
    
    def calculate_position_size(self, price: float, atr: float, account_equity: float) -> Tuple[float, int]:
        """
        Calculate position size based on account equity, ATR, and risk parameters.
        
        Args:
            price: Current price of the asset
            atr: Current ATR value
            account_equity: Current account equity
            
        Returns:
            Tuple of (position_size, units)
        """
        if atr <= 0 or price <= 0:
            return 0.0, 0
            
        # Calculate dollar amount to risk
        risk_amount = account_equity * self.risk_per_trade
        
        # Calculate position size (1 unit = 1% of account / (N * point value))
        # For simplicity, we'll assume point value is 1 (can be adjusted for different markets)
        unit_size = (account_equity * 0.01) / (atr * price)
        
        # Calculate maximum units based on risk
        max_units = min(int(risk_amount / (atr * price)), self.max_units_per_market)
        
        # Calculate position value
        position_size = unit_size * price
        
        return position_size, min(int(unit_size), max_units)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the Turtle Trading rules.
        
        Args:
            data: DataFrame containing price data and calculated indicators
            
        Returns:
            DataFrame with signal columns added
        """
        data = data.copy()
        
        # Calculate ATR and breakouts if not already present
        if 'atr' not in data.columns:
            data['atr'] = self.calculate_atr(data)
        
        data = self.calculate_breakouts(data)
        
        # Initialize signal columns
        data['signal'] = 0  # 0: no signal, 1: long, -1: short
        data['units'] = 0
        data['stop_loss'] = 0.0
        data['take_profit'] = 0.0
        
        # Track current position
        current_position = 0
        entry_price = 0.0
        units_held = 0
        
        for i in range(1, len(data)):
            # Skip if we don't have enough data
            if pd.isna(data['atr'].iloc[i]) or pd.isna(data['sys1_high'].iloc[i]):
                continue
                
            current_equity = self.initial_capital if i == 1 else self.equity[-1]
            
            # Get current price data
            current_close = self._get_column(data, 'close').iloc[i]
            current_atr = data['atr'].iloc[i]
            
            # Generate signals based on current position
            if current_position == 0:  # No position
                # System 1 entry signals
                if current_close > data['sys1_high'].iloc[i]:
                    signal = 1  # Long
                    stop_loss = current_close - (2 * current_atr)
                    position_size, units = self.calculate_position_size(
                        current_close, current_atr, current_equity
                    )
                elif current_close < data['sys1_low'].iloc[i]:
                    signal = -1  # Short
                    stop_loss = current_close + (2 * current_atr)
                    position_size, units = self.calculate_position_size(
                        current_close, current_atr, current_equity
                    )
                # System 2 entry signals (only if no System 1 signal)
                elif current_close > data['sys2_high'].iloc[i]:
                    signal = 1  # Long
                    stop_loss = current_close - (2 * current_atr)
                    position_size, units = self.calculate_position_size(
                        current_close, current_atr, current_equity
                    )
                elif current_close < data['sys2_low'].iloc[i]:
                    signal = -1  # Short
                    stop_loss = current_close + (2 * current_atr)
                    position_size, units = self.calculate_position_size(
                        current_close, current_atr, current_equity
                    )
                else:
                    signal = 0
                    stop_loss = 0.0
                    units = 0
                
                if signal != 0:
                    current_position = signal
                    entry_price = data['close'].iloc[i]
                    units_held = units
                    
                    # Record trade
                    self.trades.append({
                        'date': data.index[i],
                        'type': 'long' if signal == 1 else 'short',
                        'entry_price': entry_price,
                        'units': units,
                        'stop_loss': stop_loss,
                        'system': 'System 1' if abs(signal) == 1 and units > 0 else 'System 2'
                    })
            
            # Check exit conditions if in a position
            else:
                # Get current price data
                current_low = self._get_column(data, 'low').iloc[i]
                current_high = self._get_column(data, 'high').iloc[i]
                
                # Check stop loss first
                if (current_position == 1 and current_low <= self.trades[-1]['stop_loss']) or \
                   (current_position == -1 and current_high >= self.trades[-1]['stop_loss']):
                    # Stop loss hit
                    exit_price = self.trades[-1]['stop_loss']
                    signal = 0
                    pnl = (exit_price - entry_price) * current_position * units_held
                    
                    # Update trade with exit
                    self.trades[-1].update({
                        'exit_date': data.index[i],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0.0
                    units_held = 0
                
                # Check system-specific exit conditions
                elif (current_position == 1 and current_close < data['sys1_exit_low'].iloc[i]) or \
                     (current_position == -1 and current_close > data['sys1_exit_high'].iloc[i]):
                    # System 1 exit
                    exit_price = self._get_column(data, 'close').iloc[i]
                    signal = 0
                    pnl = (exit_price - entry_price) * current_position * units_held
                    
                    # Update trade with exit
                    self.trades[-1].update({
                        'exit_date': data.index[i],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'system1_exit'
                    })
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0.0
                    units_held = 0
                
                # Check system 2 exit conditions if in a system 2 trade
                elif (current_position == 1 and current_close < data['sys2_exit_low'].iloc[i]) or \
                     (current_position == -1 and current_close > data['sys2_exit_high'].iloc[i]):
                    # System 2 exit
                    exit_price = self._get_column(data, 'close').iloc[i]
                    signal = 0
                    pnl = (exit_price - entry_price) * current_position * units_held
                    
                    # Update trade with exit
                    self.trades[-1].update({
                        'exit_date': data.index[i],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': 'system2_exit'
                    })
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0.0
                    units_held = 0
                else:
                    signal = current_position
                    
                    # Check for pyramiding (adding to position)
                    if len(self.trades) > 0 and self.trades[-1].get('exit_date') is None:
                        last_trade = self.trades[-1]
                        if last_trade['system'] == 'System 1':  # Only pyramid System 1 trades
                            price_move = (current_close - last_trade['entry_price']) * current_position
                            n = data['atr'].iloc[i]
                            
                            # Add unit every 0.5N move
                            units_to_add = int(price_move / (0.5 * n)) - (last_trade.get('units_added', 0))
                            
                            if units_to_add > 0 and (last_trade.get('units', 0) + units_to_add) <= self.max_units_per_market:
                                # Add to position
                                last_trade['units'] += units_to_add
                                last_trade['units_added'] = last_trade.get('units_added', 0) + units_to_add
                                
                                # Update stop loss (trailing stop at 2N from highest close since entry for longs, lowest for shorts)
                                if current_position == 1:
                                    highest_since_entry = self._get_column(data, 'close').iloc[last_trade.get('entry_idx', i):i+1].max()
                                    last_trade['stop_loss'] = highest_since_entry - (2 * n)
                                else:
                                    lowest_since_entry = self._get_column(data, 'close').iloc[last_trade.get('entry_idx', i):i+1].min()
                                    last_trade['stop_loss'] = lowest_since_entry + (2 * n)
            
            # Update signal and position tracking
            # Update signal and position tracking
            data.loc[data.index[i], 'signal'] = signal
            data.loc[data.index[i], 'units'] = units_held if current_position != 0 else 0
            data.loc[data.index[i], 'position'] = current_position * units_held if current_position != 0 else 0
            
            # Update equity curve (simple version - doesn't account for multiple positions)
            if i > 0:
                if current_position != 0:
                    # Calculate daily P&L for open position
                    prev_close = self._get_column(data, 'close').iloc[i-1]
                    daily_pnl = (current_close - prev_close) * current_position * units_held
                    self.equity.append(self.equity[-1] + daily_pnl)
                else:
                    self.equity.append(self.equity[-1])
            
            # Update stop loss in data for visualization
            if current_position != 0 and len(self.trades) > 0 and self.trades[-1].get('exit_date') is None:
                if 'stop_loss' not in data.columns:
                    data['stop_loss'] = np.nan
                data.loc[data.index[i], 'stop_loss'] = self.trades[-1].get('stop_loss', np.nan)
        
        return data
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run a backtest of the Turtle Trading Strategy.
        
        Args:
            data: DataFrame containing OHLCV data with datetime index
            
        Returns:
            Dictionary containing backtest results including:
            - initial_capital: Starting capital
            - final_equity: Ending equity
            - total_return: Total return over the period
            - annualized_return: Annualized return
            - max_drawdown: Maximum drawdown
            - sharpe_ratio: Risk-adjusted return metric
            - num_trades: Total number of trades
            - win_rate: Percentage of winning trades
            - profit_factor: Gross profit / gross loss
            - trades: List of all trades
            - equity_curve: List of equity values over time
            - data: DataFrame with signals and indicators
        """
        # Reset state
        self.equity = [self.initial_capital]
        self.positions = {}
        self.trades = []
        self.signals = []
        
        # Generate signals
        data_with_signals = self.generate_signals(data)
        
        # Calculate performance metrics
        if len(self.equity) < 2:
            raise ValueError("Insufficient data for backtesting. Need at least 2 data points.")
            
        total_return = (self.equity[-1] / self.initial_capital) - 1
        
        # Calculate annualized return
        if len(data) < 2:
            years = 1.0  # Default to 1 year if insufficient data
        else:
            days = (data.index[-1] - data.index[0]).days
            years = max(days / 365.25, 0.08)  # Minimum 1 month to avoid division by zero
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate maximum drawdown
        equity_series = pd.Series(self.equity)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        # Prepare results
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(self.trades),
            'win_rate': None,
            'profit_factor': None,
            'trades': self.trades,
            'equity_curve': self.equity,
            'data': data_with_signals
        }
        
        # Calculate win rate and profit factor if there are trades
        if self.trades:
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]
            
            results['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0
            
            gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
            gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
            results['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return results


def example_usage():
    """Example of how to use the TurtleTradingStrategy class."""
    import yfinance as yf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Download sample data with progress=False to suppress progress bar
    print("Downloading sample data...")
    try:
        data = yf.download('SPY', start='2020-01-01', end='2023-01-01', progress=False)
        print(f"Successfully downloaded {len(data)} rows of data.")
        
        # Check if we got any data
        if data.empty:
            print("Warning: No data returned from yfinance. Trying with a different date range...")
            data = yf.download('SPY', period='2y', progress=False)
            print(f"Downloaded {len(data)} rows with 'period=2y'")
            
        # Convert to DataFrame if it's a Series
        if isinstance(data, pd.Series):
            data = data.to_frame('close')
        
        # Initialize strategy
        strategy = TurtleTradingStrategy()
        
        # Flatten column names if needed
        data = strategy._flatten_columns(data)
        
        # Ensure we have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        print("\nChecking for required columns...")
        for col in required_columns:
            if col not in data.columns:
                print(f"Warning: Required column '{col}' not found in data")
        
        # Add any missing required columns with NaN values
        for col in required_columns:
            if col not in data.columns:
                print(f"Adding missing column: {col}")
                data[col] = np.nan
        
        # Ensure we have enough data points
        min_required_data_points = max(strategy.system1_entry, strategy.system2_entry, strategy.atr_period) * 2
        if len(data) < min_required_data_points:
            print(f"Warning: Only {len(data)} data points available, but at least {min_required_data_points} are recommended.")
        
        # Print debug information
        print("\nData columns:", data.columns.tolist())
        print("\nFirst few rows of data:")
        print(data.head())
        print("\nLast few rows of data:")
        print(data.tail())
        print("\nData types:")
        print(data.dtypes)
        
        # Check for NaN values
        print("\nNaN values per column:")
        print(data.isna().sum())
        
        # Drop any rows with NaN values in OHLC
        initial_count = len(data)
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        if len(data) < initial_count:
            print(f"Dropped {initial_count - len(data)} rows with NaN values")
        
    except Exception as e:
        print(f"Error downloading or processing data: {str(e)}")
        print("Troubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Try a different ticker symbol (e.g., 'AAPL', 'MSFT')")
        print("3. Try a different date range")
        print("4. Check if yfinance is up to date (pip install --upgrade yfinance)")
        return
    
    # Strategy is already initialized above
    
    # Check if we have enough data after cleaning
    if len(data) < 2:
        print("\nError: Not enough data points for backtesting after cleaning.")
        print(f"Needed at least 2, but only have {len(data)} rows.")
        return
    
    # Run backtest
    print("\nRunning backtest...")
    try:
        results = strategy.backtest(data)
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
        print("\nDebug info:")
        print(f"Data shape: {data.shape}")
        print(f"Data index: {data.index[:5]} ... {data.index[-5:]}")
        print(f"Data columns: {data.columns.tolist()}")
        return
    
    # Print results
    print(f"\nBacktest Results (2020-2022)")
    print("=" * 50)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {results['num_trades']}")
    
    if results['num_trades'] > 0:
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, results['equity_curve'][1:], label='Equity Curve')
    plt.title('Turtle Trading Strategy - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results


if __name__ == "__main__":
    example_usage()
