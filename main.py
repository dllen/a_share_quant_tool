import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

# Local application imports
# from strategies.ma_turnover_strategy import MATurnoverStrategy
from strategies.smart_money_strategy import SmartMoneyStrategy
from utils.data_fetcher import AShareDataFetcher
from utils.visualization import (
    plot_backtest_results,
    plot_returns_distribution,
    plot_stock_data_with_signals,
    plot_turnover_indicators,
)

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 初始化数据获取器
    data_fetcher = AShareDataFetcher()
    
    # 设置股票代码和时间范围
    # stock_code = '000001'  # 平安银行
    # stock_code = '600519'  # 五粮液
    # stock_code = '601598'  # 中国外运
    # stock_code = '000999'  # 华润三九
    # stock_code = '688247'  # 宣泰医药
    # stock_code = '600642'   # 申能股份
    # stock_code = '603716'   # 塞力医疗
    # stock_code = '600085' # 同仁堂
    stock_code = '871642'
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365*1)).strftime('%Y%m%d')
    
    print(f"正在获取 {stock_code} 的股票数据...")
    
    # 获取股票数据并计算技术指标
    df = data_fetcher.get_stock_with_indicators(stock_code, start_date, end_date)
    
    if df is None or df.empty:
        print("未能获取股票数据，请检查股票代码或网络连接。")
        return
    
    print(f"成功获取 {len(df)} 个交易日的数据。")
    
    # 初始化策略
    # strategy = MATurnoverStrategy(
    #     ma_windows=[5, 10, 20, 60, 120],
    #     turnover_ma_windows=[5, 10],
    #     turnover_threshold=1.5
    # )
    strategy = SmartMoneyStrategy(
        ma_windows=[5, 10, 20],
        atr_period=14,
        atr_multiplier=1.5,
        position_size=0.25
    )
    
    # 生成交易信号
    print("正在生成交易信号...")
    signals = strategy.generate_signals(df)
    
    # 执行回测
    print("正在执行回测...")
    results = strategy.backtest(signals)
    
    # 打印回测结果
    print("\n=== 回测结果 ===")
    print(f"初始资金: {results['initial_capital']:,.2f} 元")
    print(f"最终资金: {results['final_value']:,.2f} 元")
    print(f"总收益率: {results['total_return']:.2f}%")
    print(f"年化收益率: {results['annual_return']:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2f}%")
    trades = results['trades']
    rows,cols = trades.shape
    print(f"交易次数: {rows if not trades.empty else 0} 次")
    print(trades)
    
    # 创建图表
    print("\n正在生成图表...")
    
    # 1. 股票价格与交易信号
    plt1 = plot_stock_data_with_signals(
        df, 
        signals[signals['signal'] != 0],
        title=f'{stock_code} - 股票价格与交易信号'
    )
    
    # 2. 换手率指标
    plt2 = plot_turnover_indicators(
        df,
        title=f'{stock_code} - 换手率指标'
    )
    
    # 3. 回测结果
    plt3 = plot_backtest_results(
        results['portfolio_value'],
        title=f'{stock_code} - 策略回测结果'
    )
    
    # 4. 收益率分布
    returns = results['portfolio_value'].pct_change().dropna()
    plt4 = plot_returns_distribution(
        returns,
        title=f'{stock_code} - 日收益率分布'
    )
    
    # 显示所有图表
    print("\n图表生成完成，正在显示...")
    plt.show()
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
