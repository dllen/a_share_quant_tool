"""
Backtest Grid Trading and Turtle Trading Strategies on A-Share Market Data
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ashare_loader import AShareDataLoader
from strategies.grid_trading_strategy import GridTradingStrategy
from strategies.turtle_trading_strategy import TurtleTradingStrategy


def run_backtest(strategy_name: str, data: pd.DataFrame, params: dict) -> dict:
    """
    Run backtest for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy ('grid' or 'turtle')
        data: DataFrame with OHLCV data
        params: Strategy parameters
        
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*50}")
    print(f"Running {strategy_name.upper()} strategy backtest")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Data points: {len(data)}")
    
    if strategy_name.lower() == 'grid':
        # Initialize Grid Trading Strategy
        strategy = GridTradingStrategy(
            upper_price=data['close'].max() * 1.1,  # 10% above highest price
            lower_price=data['close'].min() * 0.9,  # 10% below lowest price
            grid_number=params.get('grid_number', 10),
            initial_capital=params.get('initial_capital', 100000),
            position_per_grid=params.get('position_per_grid', 0.1),
            take_profit_pct=params.get('take_profit_pct', 0.02),
            stop_loss_pct=params.get('stop_loss_pct', 0.1),
            fee_rate=params.get('fee_rate', 0.0005)
        )
    elif strategy_name.lower() == 'turtle':
        # Initialize Turtle Trading Strategy
        strategy = TurtleTradingStrategy(
            initial_capital=params.get('initial_capital', 100000),
            risk_per_trade=params.get('risk_per_trade', 0.01),
            max_units_per_market=params.get('max_units_per_market', 4),
            atr_period=params.get('atr_period', 20),
            system1_entry=params.get('system1_entry', 20),
            system1_exit=params.get('system1_exit', 10),
            system2_entry=params.get('system2_entry', 55),
            system2_exit=params.get('system2_exit', 20)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Run backtest
    results = strategy.backtest(data)
    
    # Store the strategy instance in results
    results['strategy'] = strategy
    
    # Print summary
    print("\nBacktest Results")
    print("-" * 30)
    print(f"Initial Capital: ￥{results['initial_capital']:,.2f}")
    print(f"Final Equity:    ￥{results['final_equity']:,.2f}")
    
    if strategy_name.lower() == 'grid':
        print(f"Total Return:    {results['total_return_pct']:,.2f}%")
        print(f"Max Drawdown:    {results['max_drawdown_pct']:,.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        if results['num_trades'] > 0:
            print(f"Win Rate:        {results['win_rate_pct']:,.1f}%")
            print(f"Profit Factor:   {results['profit_factor']:,.2f}")
    else:  # turtle
        print(f"Total Return:    {results['total_return'] * 100:,.2f}%")
        print(f"Max Drawdown:    {results['max_drawdown'] * 100:,.2f}%")
        print(f"Number of Trades: {len([t for t in results['trades'] if t['type'] == 'exit'])}")
        if results.get('win_rate') is not None:
            print(f"Win Rate:        {results['win_rate'] * 100:,.1f}%")
    
    return results


def plot_results(strategy_name: str, data: pd.DataFrame, results: dict):
    """Plot backtest results."""
    plt.figure(figsize=(14, 8))
    
    # Plot price and signals
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
    
    # Plot strategy-specific elements
    if strategy_name.lower() == 'grid':
        # Plot grid levels
        for level in results['strategy'].grid_levels:
            ax1.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
        
        # Plot trades
        trades = pd.DataFrame(results['trades'])
        if not trades.empty:
            entry_trades = trades[trades['type'] == 'entry']
            exit_trades = trades[trades['type'] == 'exit']
            
            if not entry_trades.empty and 'entry_time' in entry_trades.columns and 'entry_price' in entry_trades.columns:
                ax1.scatter(entry_trades['entry_time'], 
                          entry_trades['entry_price'], 
                          color='green', marker='^', label='Buy', s=100)
            
            if not exit_trades.empty and 'exit_time' in exit_trades.columns and 'exit_price' in exit_trades.columns:
                ax1.scatter(exit_trades['exit_time'], 
                          exit_trades['exit_price'], 
                          color='red', marker='v', label='Sell', s=100)
    
    elif strategy_name.lower() == 'turtle':
        # Plot entry/exit signals
        signals = results['data']
        if 'signal' in signals.columns:
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]
            
            ax1.scatter(buy_signals.index, 
                       buy_signals['close'], 
                       color='green', marker='^', label='Buy', s=100)
            
            ax1.scatter(sell_signals.index, 
                       sell_signals['close'], 
                       color='red', marker='v', label='Sell', s=100)
    
    ax1.set_title(f'{strategy_name.upper()} Strategy - Price and Trades')
    ax1.set_ylabel('Price (RMB)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot equity curve
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    if strategy_name.lower() == 'grid':
        equity = results['equity_curve']
        # Ensure we have matching indices for price and equity data
        if len(data) == len(equity):
            ax2.plot(data.index, equity, label='Equity Curve', color='green')
        else:
            # If dimensions don't match, use the shorter length
            min_len = min(len(data), len(equity))
            ax2.plot(data.index[:min_len], equity[:min_len], label='Equity Curve', color='green')
    else:  # turtle
        equity = results['equity_curve']
        # Ensure we have matching indices for price and equity data
        if len(data) == len(equity):
            ax2.plot(data.index, equity, label='Equity Curve', color='green')
        else:
            # If dimensions don't match, use the shorter length
            min_len = min(len(data), len(equity))
            ax2.plot(data.index[:min_len], equity[:min_len], label='Equity Curve', color='green')
    
    ax2.set_title('Equity Curve')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity (RMB)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_ashare_strategies(symbol: str = '600000', 
                          start_date: str = '20220101', 
                          end_date: str = '20231231'):
    """
    Test both Grid and Turtle trading strategies on A-share data.
    
    Args:
        symbol: Stock symbol with exchange prefix (e.g., 'sh600000' for 浦发银行)
        start_date: Start date in 'YYYYMMDD' format
        end_date: End date in 'YYYYMMDD' format
    """
    # Initialize data loader
    loader = AShareDataLoader()
    
    # Load A-share data
    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    data = loader.get_stock_data_akshare(symbol, start_date, end_date)
    
    if data.empty:
        print(f"No data found for {symbol}. Trying Tushare...")
        data = loader.get_stock_data_tushare(symbol, start_date, end_date)
        
    if data.empty:
        print("Failed to load data. Please check your internet connection and try again.")
        return
    
    print(f"\nLoaded {len(data)} data points for {symbol}")
    print(data[['open', 'high', 'low', 'close', 'volume']].head())
    
    # Define strategy parameters
    common_params = {
        'initial_capital': 100000,
    }
    
    grid_params = {
        **common_params,
        'grid_number': 10,
        'position_per_grid': 0.1,
        'take_profit_pct': 0.02,
        'stop_loss_pct': 0.1,
        'fee_rate': 0.0005
    }
    
    turtle_params = {
        **common_params,
        'risk_per_trade': 0.01,
        'max_units_per_market': 4,
        'atr_period': 20,
        'system1_entry': 20,
        'system1_exit': 10,
        'system2_entry': 55,
        'system2_exit': 20
    }
    
    # Run backtests
    grid_results = run_backtest('grid', data.copy(), grid_params)
    turtle_results = run_backtest('turtle', data.copy(), turtle_params)
    
    # Plot results
    plot_results('grid', data, grid_results)
    plot_results('turtle', data, turtle_results)
    
    return {
        'grid': grid_results,
        'turtle': turtle_results,
        'data': data
    }


if __name__ == "__main__":
    # Example usage
    test_ashare_strategies(
        symbol='300487',
        start_date='20240101',
        end_date='20250714'
    )
