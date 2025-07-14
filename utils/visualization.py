import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import ticker
import pandas as pd

def plot_stock_data_with_signals(data, signals, title='Stock Price with Trading Signals'):
    """
    绘制股票价格、均线及交易信号
    
    参数:
        data: 包含股票价格和指标数据的DataFrame
        signals: 包含交易信号的DataFrame
        title: 图表标题
    """
    plt.figure(figsize=(14, 8))
    
    # 绘制收盘价
    plt.plot(data.index, data['close'], label='Close Price', color='black', linewidth=1)
    
    # 绘制均线
    for ma in [5, 10, 20, 60, 200]:
        if f'ma{ma}' in data.columns:
            plt.plot(data.index, data[f'ma{ma}'], label=f'MA{ma}', alpha=0.7, linewidth=0.8)
    
    # 标记买卖信号
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], 
               marker='^', color='red', label='Buy Signal', alpha=1, s=100)
    plt.scatter(sell_signals.index, sell_signals['close'], 
               marker='v', color='green', label='Sell Signal', alpha=1, s=100)
    
    # 设置图表格式
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def plot_turnover_indicators(data, title='Turnover Rate Indicators'):
    """
    绘制换手率及其均线
    
    参数:
        data: 包含换手率数据的DataFrame
        title: 图表标题
    """
    plt.figure(figsize=(14, 6))
    
    # 绘制换手率
    plt.bar(data.index, data['turnover_rate'], 
            color='gray', alpha=0.3, width=1, 
            label='Daily Turnover')
    
    # 绘制换手率均线
    for ma in [5, 10]:
        if f'turnover_ma{ma}' in data.columns:
            plt.plot(data.index, data[f'turnover_ma{ma}'], 
                   label=f'Turnover MA{ma}', linewidth=1.5)
    
    # 设置图表格式
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Turnover Rate (%)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴格式为百分比
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def plot_backtest_results(portfolio_value, benchmark=None, title='Backtest Results'):
    """
    绘制回测结果
    
    参数:
        portfolio_value: 投资组合价值序列
        benchmark: 基准指数数据 (可选)
        title: 图表标题
    """
    plt.figure(figsize=(14, 6))
    
    # 绘制投资组合价值曲线
    portfolio_returns = portfolio_value.pct_change().fillna(0)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    plt.plot(cumulative_returns.index, (cumulative_returns - 1) * 100, 
            label='Strategy', linewidth=2, color='blue')
    
    # 绘制基准指数（如果提供）
    if benchmark is not None:
        if isinstance(benchmark, pd.Series):
            benchmark_returns = benchmark.pct_change().fillna(0)
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            plt.plot(benchmark_cumulative.index, (benchmark_cumulative - 1) * 100,
                    label='Benchmark', linewidth=2, color='red', linestyle='--')
    
    # 设置图表格式
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴格式为百分比
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def plot_trade_signals(data, signals, title='Trading Signals'):
    """
    绘制交易信号
    
    参数:
        data: 包含价格数据的DataFrame
        signals: 包含交易信号的DataFrame
        title: 图表标题
    """
    plt.figure(figsize=(14, 8))
    
    # 绘制收盘价
    plt.plot(data.index, data['close'], label='Close Price', color='black', linewidth=1)
    
    # 标记买卖信号
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], 
               marker='^', color='red', label='Buy Signal', alpha=1, s=100)
    plt.scatter(sell_signals.index, sell_signals['close'], 
               marker='v', color='green', label='Sell Signal', alpha=1, s=100)
    
    # 设置图表格式
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def plot_returns_distribution(returns, title='Returns Distribution'):
    """
    绘制收益率分布直方图
    
    参数:
        returns: 收益率序列
        title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制收益率分布直方图
    sns.histplot(returns * 100, kde=True, bins=50, color='blue', alpha=0.6)
    
    # 添加均线和标准差线
    mean_return = returns.mean() * 100
    std_return = returns.std() * 100
    
    plt.axvline(mean_return, color='red', linestyle='--', 
               label=f'Mean: {mean_return:.2f}%')
    plt.axvline(mean_return + std_return, color='green', linestyle=':', 
               label=f'±1 Std Dev: {std_return:.2f}%')
    plt.axvline(mean_return - std_return, color='green', linestyle=':')
    
    # 设置图表格式
    plt.title(title, fontsize=16)
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return plt
