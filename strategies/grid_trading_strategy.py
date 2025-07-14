"""
Grid Trading Strategy Implementation

This module implements a Grid Trading Strategy that places buy and sell orders at regular price intervals
(price grids) above and below a base price. The strategy profits from price oscillations within a range.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class GridTradingStrategy:
    """
    Implementation of a Grid Trading Strategy.
    
    The strategy works by:
    1. Defining a price range (upper and lower bounds)
    2. Creating a grid of price levels within that range
    3. Placing buy orders below the current price and sell orders above
    4. Profiting from price oscillations within the range
    """
    
    def __init__(self, 
                 upper_price: float,
                 lower_price: float,
                 grid_number: int = 10,
                 initial_capital: float = 100000.0,
                 position_per_grid: float = 0.1,
                 take_profit_pct: float = 0.02,
                 stop_loss_pct: float = 0.1,
                 fee_rate: float = 0.0005):
        """
        Initialize the Grid Trading Strategy.
        
        Args:
            upper_price: Upper bound of the trading range
            lower_price: Lower bound of the trading range
            grid_number: Number of grids to create (default: 10)
            initial_capital: Starting capital for the strategy (default: 100,000)
            position_per_grid: Percentage of capital to use per grid (default: 10%)
            take_profit_pct: Take profit percentage per trade (default: 2%)
            stop_loss_pct: Stop loss percentage per trade (default: 10%)
            fee_rate: Trading fee rate (default: 0.05%)
        """
        if upper_price <= lower_price:
            raise ValueError("Upper price must be greater than lower price")
            
        self.upper_price = upper_price
        self.lower_price = lower_price
        self.grid_number = grid_number
        self.initial_capital = initial_capital
        self.position_per_grid = position_per_grid
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.fee_rate = fee_rate
        
        # Calculate grid levels
        self.grid_levels = self._calculate_grid_levels()
        
        # Initialize state
        self.reset()
    
    def _calculate_grid_levels(self) -> np.ndarray:
        """Calculate price levels for the grid."""
        price_range = self.upper_price - self.lower_price
        grid_size = price_range / (self.grid_number + 1)
        return np.linspace(
            self.lower_price + grid_size,
            self.upper_price - grid_size,
            self.grid_number
        )
    
    def reset(self):
        """Reset the strategy state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity = [self.initial_capital]
        self.current_price = None
        self.grid_positions = {level: 0 for level in self.grid_levels}
    
    def generate_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on grid levels.
        
        Args:
            price_data: DataFrame with OHLCV data and a 'close' column
            
        Returns:
            DataFrame with signals and indicators
        """
        if 'close' not in price_data.columns:
            raise ValueError("Input DataFrame must contain 'close' column")
            
        data = price_data.copy()
        data['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        data['position'] = 0.0
        data['equity'] = self.initial_capital
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            self.current_price = current_price
            
            # Check each grid level for potential trades
            for level in self.grid_levels:
                # Buy signal: price crosses above grid level from below
                if (data['close'].iloc[i-1] < level <= current_price and 
                    self.grid_positions[level] == 0):
                    self._enter_trade(level, current_price, 'buy', data.index[i])
                    data.at[data.index[i], 'signal'] = 1
                
                # Sell signal: price crosses below grid level from above
                elif (data['close'].iloc[i-1] > level >= current_price and 
                      self.grid_positions[level] == 0):
                    self._enter_trade(level, current_price, 'sell', data.index[i])
                    data.at[data.index[i], 'signal'] = -1
            
            # Update position and equity
            data.at[data.index[i], 'position'] = self._get_total_position()
            data.at[data.index[i], 'equity'] = self._calculate_equity(current_price)
            self.equity.append(self._calculate_equity(current_price))
        
        return data
    
    def _enter_trade(self, level: float, price: float, side: str, timestamp) -> None:
        """Enter a new trade at a grid level."""
        position_size = self.initial_capital * self.position_per_grid / price
        
        if side == 'buy':
            self.positions[level] = {
                'entry_price': price,
                'position': position_size,
                'entry_time': timestamp,
                'exit_price': price * (1 + self.take_profit_pct),
                'stop_loss': price * (1 - self.stop_loss_pct),
                'side': 'long'
            }
            self.grid_positions[level] = 1
        else:  # sell
            self.positions[level] = {
                'entry_price': price,
                'position': position_size,
                'entry_time': timestamp,
                'exit_price': price * (1 - self.take_profit_pct),
                'stop_loss': price * (1 + self.stop_loss_pct),
                'side': 'short'
            }
            self.grid_positions[level] = -1
        
        # Record the trade
        self.trades.append({
            'level': level,
            'entry_price': price,
            'position': position_size,
            'entry_time': timestamp,
            'side': side,
            'type': 'entry'
        })
    
    def _exit_trade(self, level: float, price: float, reason: str, timestamp) -> None:
        """Exit a trade at a grid level."""
        if level not in self.positions:
            return
            
        trade = self.positions[level]
        pnl = self._calculate_pnl(trade, price)
        
        # Record the exit
        self.trades.append({
            'level': level,
            'exit_price': price,
            'position': trade['position'],
            'exit_time': timestamp,
            'side': trade['side'],
            'type': 'exit',
            'pnl': pnl,
            'return_pct': (price / trade['entry_price'] - 1) * (-1 if trade['side'] == 'short' else 1),
            'reason': reason
        })
        
        # Update cash and positions
        if trade['side'] == 'long':
            self.cash += trade['position'] * price * (1 - self.fee_rate)
        else:  # short
            self.cash += trade['position'] * (2 * trade['entry_price'] - price) * (1 - self.fee_rate)
        
        # Clean up
        del self.positions[level]
        self.grid_positions[level] = 0
    
    def _calculate_pnl(self, trade: dict, exit_price: float) -> float:
        """Calculate profit/loss for a trade."""
        if trade['side'] == 'long':
            return trade['position'] * (exit_price - trade['entry_price']) * (1 - self.fee_rate)
        else:  # short
            return trade['position'] * (trade['entry_price'] - exit_price) * (1 - self.fee_rate)
    
    def _get_total_position(self) -> float:
        """Calculate total position size."""
        return sum(trade['position'] for trade in self.positions.values())
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate total account equity."""
        position_value = 0.0
        
        for trade in self.positions.values():
            if trade['side'] == 'long':
                position_value += trade['position'] * current_price
            else:  # short
                position_value += trade['position'] * (2 * trade['entry_price'] - current_price)
        
        return self.cash + position_value
    
    def backtest(self, price_data: pd.DataFrame) -> dict:
        """
        Run a backtest of the Grid Trading Strategy.
        
        Args:
            price_data: DataFrame with OHLCV data and a 'close' column
            
        Returns:
            Dictionary containing backtest results
        """
        self.reset()
        data = self.generate_signals(price_data)
        
        # Calculate performance metrics
        total_return = (self.equity[-1] / self.initial_capital - 1) * 100
        
        # Calculate drawdown
        equity_series = pd.Series(self.equity)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Calculate trade statistics
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            # Filter only exit trades (which have P&L)
            exit_trades = trades_df[trades_df['type'] == 'exit']
            
            if not exit_trades.empty:
                win_rate = (exit_trades['pnl'] > 0).mean() * 100
                avg_win = exit_trades[exit_trades['pnl'] > 0]['pnl'].mean()
                avg_loss = exit_trades[exit_trades['pnl'] <= 0]['pnl'].mean()
                profit_factor = abs(exit_trades[exit_trades['pnl'] > 0]['pnl'].sum() / 
                                  exit_trades[exit_trades['pnl'] < 0]['pnl'].sum()) \
                                  if exit_trades[exit_trades['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0.0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity[-1],
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'num_trades': len([t for t in self.trades if t['type'] == 'exit']),
            'win_rate_pct': win_rate,
            'avg_win': avg_win if 'avg_win' in locals() else 0,
            'avg_loss': avg_loss if 'avg_loss' in locals() else 0,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'equity_curve': self.equity,
            'data': data
        }
    
    def plot_grid(self, price_data: pd.DataFrame = None, figsize=(12, 8)) -> None:
        """
        Plot the grid levels and price action.
        
        Args:
            price_data: Optional DataFrame with price data to plot
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Plot price data if provided
        if price_data is not None and not price_data.empty:
            plt.plot(price_data.index, price_data['close'], label='Price', color='blue', alpha=0.7)
        
        # Plot grid levels
        for level in self.grid_levels:
            plt.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
            plt.text(price_data.index[0] if price_data is not None else 0, 
                    level, f"{level:.2f}", 
                    verticalalignment='bottom',
                    backgroundcolor='white')
        
        # Plot upper and lower bounds
        plt.axhline(y=self.upper_price, color='red', linestyle='-', label='Upper Bound')
        plt.axhline(y=self.lower_price, color='green', linestyle='-', label='Lower Bound')
        
        # Mark entry/exit points if available
        if hasattr(self, 'trades') and self.trades:
            trades_df = pd.DataFrame(self.trades)
            entry_trades = trades_df[trades_df['type'] == 'entry']
            exit_trades = trades_df[trades_df['type'] == 'exit']
            
            if not entry_trades.empty and price_data is not None:
                plt.scatter(entry_trades['entry_time'], 
                          entry_trades['entry_price'], 
                          color='green', marker='^', 
                          label='Buy', s=100)
            
            if not exit_trades.empty and price_data is not None:
                plt.scatter(exit_trades['exit_time'], 
                          exit_trades['exit_price'], 
                          color='red', marker='v', 
                          label='Sell', s=100)
        
        plt.title('Grid Trading Strategy')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()


def example_usage():
    """Example of how to use the GridTradingStrategy class."""
    import yfinance as yf
    
    # Download sample data
    print("Downloading sample data...")
    data = yf.download('BTC-USD', start='2023-01-01', end='2023-06-30', progress=False)
    
    if data.empty:
        print("Failed to download data. Using sample data instead.")
        # Create sample data if download fails
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        base_prices = np.linspace(20000, 30000, len(dates))
        noise = np.random.normal(0, 500, len(dates))
        data = pd.DataFrame({
            'open': base_prices + noise,
            'high': base_prices + noise + 100,
            'low': base_prices + noise - 100,
            'close': base_prices + noise + np.random.normal(0, 20, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
    
    # Initialize strategy
    close_prices = data['Close'] if 'Close' in data.columns else data['close']
    upper_price = close_prices.max() * 1.05  # 5% above highest price
    lower_price = close_prices.min() * 0.95  # 5% below lowest price
    
    strategy = GridTradingStrategy(
        upper_price=upper_price,
        lower_price=lower_price,
        grid_number=10,
        initial_capital=100000.0,
        position_per_grid=0.1,  # 10% of capital per grid
        take_profit_pct=0.02,   # 2% take profit
        stop_loss_pct=0.1,      # 10% stop loss
        fee_rate=0.001          # 0.1% trading fee
    )
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(data)
    
    # Print results
    print("\nBacktest Results")
    print("=" * 50)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    if results['num_trades'] > 0:
        print(f"Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"Average Win: ${results['avg_win']:,.2f}")
        print(f"Average Loss: ${results['avg_loss']:,.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Plot results
    strategy.plot_grid(data)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, results['equity_curve'][1:], label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example_usage()
