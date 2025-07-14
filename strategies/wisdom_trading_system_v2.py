"""
智慧交易系统 V2
基于《炒股的智慧》的交易策略，结合均线、ATR等技术指标
"""
from typing import Dict, List, Optional

import akshare as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class WisdomTradingSystemV2:
    """
    智慧交易系统 V2
    
    核心原则：
    1. 趋势跟踪 - 使用均线系统判断趋势
    2. 风险控制 - 使用ATR进行动态止损
    3. 资金管理 - 控制仓位，分散风险
    """
    
    def __init__(self, 
                 ma_windows: List[int] = [5, 10, 20, 50, 200],
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 position_size: float = 0.1,
                 support_window: int = 20,
                 resistance_window: int = 20,
                 support_threshold: float = 0.02,
                 resistance_threshold: float = 0.02):
        """
        初始化交易系统
        
        参数:
            ma_windows: 均线周期列表，用于判断趋势
            atr_period: ATR计算周期，用于止损
            atr_multiplier: ATR乘数，用于计算止损位
            position_size: 每次交易的仓位比例
            support_window: 支撑线检测窗口大小
            resistance_window: 压力线检测窗口大小
            support_threshold: 支撑位附近买入阈值（百分比）
            resistance_threshold: 压力位附近卖出阈值（百分比）
        """
        self.ma_windows = sorted(ma_windows)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.position_size = position_size
        self.support_window = support_window
        self.resistance_window = resistance_window
        self.support_threshold = support_threshold
        self.resistance_threshold = resistance_threshold
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.trailing_stop = 0.0
        self.data = None
        self.trades = []
        self.support_levels = []
        self.resistance_levels = []
    
    def load_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载股票数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            
        返回:
            pd.DataFrame: 包含股票数据的DataFrame
        """
        print(f"正在获取股票 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
        
        try:
            # 获取后复权数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df.empty:
                raise ValueError(f"未获取到数据，请检查股票代码 {stock_code} 和日期范围")
                
            # 重命名列
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover'
            }
            df = df.rename(columns=column_mapping)
            
            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"获取数据时出错: {str(e)}")
            raise
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """计算平均真实波幅(ATR)"""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=self.atr_period).mean()
        
    def find_support_resistance(self, data: pd.DataFrame) -> tuple:
        """
        识别支撑位和压力位
        
        参数:
            data: 包含价格数据的DataFrame
            
        返回:
            tuple: (support_levels, resistance_levels)
        """
        df = data.copy()
        
        # 识别局部最低点作为支撑位
        df['min'] = df['low'].rolling(window=self.support_window, center=True).min()
        support = df[df['low'] == df['min']]['low']
        support_levels = support.groupby(level=0).first().sort_index()
        
        # 识别局部最高点作为压力位
        df['max'] = df['high'].rolling(window=self.resistance_window, center=True).max()
        resistance = df[df['high'] == df['max']]['high']
        resistance_levels = resistance.groupby(level=0).first().sort_index()
        
        # 存储支撑位和压力位
        self.support_levels = support_levels.tolist()
        self.resistance_levels = resistance_levels.tolist()
        
        return support_levels, resistance_levels
        
    def is_near_support(self, price: float) -> bool:
        """判断价格是否接近支撑位"""
        if not self.support_levels:
            return False
        closest_support = min(self.support_levels, key=lambda x: abs(x - price))
        return abs(closest_support - price) / closest_support <= self.support_threshold
        
    def is_near_resistance(self, price: float) -> bool:
        """判断价格是否接近压力位"""
        if not self.resistance_levels:
            return False
        closest_resistance = min(self.resistance_levels, key=lambda x: abs(x - price))
        return abs(closest_resistance - price) / closest_resistance <= self.resistance_threshold
        
    def is_below_resistance_threshold(self, price: float, threshold_pct: float = 0.3) -> bool:
        """判断价格是否跌破压力位一定比例
        
        参数:
            price: 当前价格
            threshold_pct: 跌破压力位的百分比阈值
            
        返回:
            bool: 如果价格低于最近压力位的(1 - threshold_pct)倍，返回True
        """
        if not self.resistance_levels:
            return False
        closest_resistance = min(self.resistance_levels, key=lambda x: abs(x - price))
        return price <= closest_resistance * (1 - threshold_pct)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        生成交易信号
        
        返回:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        df = self.data.copy()
        
        # 计算均线
        for ma in self.ma_windows:
            df[f'ma{ma}'] = df['close'].rolling(window=ma).mean()
        
        # 计算ATR
        df['atr'] = self.calculate_atr(df)
        
        # 识别支撑位和压力位
        support_levels, resistance_levels = self.find_support_resistance(df)
        
        # 初始化信号列
        df['signal'] = 0  # 1: 买入, -1: 卖出, 0: 持有
        df['near_support'] = False
        df['near_resistance'] = False
        
        # 生成交易信号
        for i in range(max(self.ma_windows), len(df)):
            current = df.iloc[i]
            
            # 获取均线值
            ma_values = [current[f'ma{ma}'] for ma in self.ma_windows]

            # 1. 首先判断均线趋势
            # 判断均线是否形成多头排列（短期均线在长期均线之上）
            ma_trend = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
            # 判断是否处于上升趋势（价格在主要均线之上）
            price_above_ma = current['close'] > current[f'ma{self.ma_windows[-1]}']
            is_uptrend = ma_trend and price_above_ma
            
            # 2. 计算成交量指标
            volume_ma5 = df['volume'].rolling(5).mean().iloc[i]
            volume_increasing = current['volume'] > volume_ma5  # 成交量放大
            
            # 3. 检查价格行为
            prev_close = df.iloc[i-1]['close'] if i > 0 else current['close']
            price_making_new_high = current['close'] > prev_close  # 价格创新高
            
            # 4. 检查是否接近支撑位或压力位（仅在趋势确认后使用）
            near_support = False
            near_resistance = False
            
            # 更新DataFrame中的标记
            df.loc[df.index[i], 'is_uptrend'] = is_uptrend
            df.loc[df.index[i], 'volume_increasing'] = volume_increasing
            df.loc[df.index[i], 'price_making_new_high'] = price_making_new_high
            
            # 只在趋势向上时检查支撑位
            if is_uptrend:
                near_support = self.is_near_support(current['close'])
                near_resistance = self.is_near_resistance(current['close'])
                df.loc[df.index[i], 'near_support'] = near_support
                df.loc[df.index[i], 'near_resistance'] = near_resistance
            
            # 买入条件：首先确认上升趋势，然后检查其他条件
            buy_condition = False
            if is_uptrend:
                buy_condition = (
                    price_making_new_high and  # 价格创新高
                    volume_increasing and      # 成交量放大
                    near_support               # 接近支撑位（作为确认）
                )
            
            # 卖出条件：趋势转弱或接近压力位或跌破压力位30%
            sell_condition = False
            if df['signal'].iloc[i-1] == 1:  # 已有仓位
                below_resistance_threshold = self.is_below_resistance_threshold(current['close'])
                sell_condition = (
                    current['close'] < self.trailing_stop or  # 触发止损
                    not is_uptrend or                        # 趋势转弱
                    (near_resistance and not price_making_new_high) or  # 接近压力位且动能减弱
                    below_resistance_threshold               # 跌破压力位30%
                )
                
                # 记录是否触发30%跌破压力位信号
                if below_resistance_threshold:
                    df.loc[df.index[i], 'below_resistance_threshold'] = True
            
            # 生成信号
            if buy_condition:
                df.loc[df.index[i], 'signal'] = 1
                self.entry_price = current['close']
                self.stop_loss = current['close'] - (self.atr_multiplier * current['atr'])
                self.trailing_stop = self.stop_loss
            elif sell_condition:
                df.loc[df.index[i], 'signal'] = -1
            
            # 更新跟踪止损
            if df['signal'].iloc[i-1] == 1:  # 持有仓位
                new_stop = current['close'] - (self.atr_multiplier * current['atr'])
                self.trailing_stop = max(self.trailing_stop, new_stop)
                
                # 如果接近压力位，考虑部分止盈
                if near_resistance and current['close'] > self.entry_price * 1.05:  # 至少盈利5%
                    df.loc[df.index[i], 'signal'] = -1  # 平仓
        
        self.data = df
        return df
    
    def backtest(self, 
                initial_capital: float = 10000.0,
                commission: float = 0.0005) -> Dict:
        """
        回测策略表现
        
        参数:
            initial_capital: 初始资金
            commission: 交易佣金率
            
        返回:
            包含回测结果的字典
        """
        if self.data is None or 'signal' not in self.data.columns:
            self.generate_signals()
            
        df = self.data.copy()
        
        # 初始化回测变量
        position = 0
        cash = initial_capital
        portfolio_value = [initial_capital]
        trades = []
        
        # 执行回测
        for i in range(1, len(df)):
            current = df.iloc[i]
            
            # 获取信号
            signal = df['signal'].iloc[i]
            
            # 执行交易
            if signal == 1 and position == 0:  # 买入
                position_size = (cash * self.position_size) / current['close']
                position = min(position_size, cash / current['close'])  # 确保不超过可用资金
                trade_value = position * current['close']
                commission_paid = trade_value * commission
                cash -= (trade_value + commission_paid)
                
                trades.append({
                    'date': current.name,
                    'type': 'buy',
                    'price': current['close'],
                    'shares': position,
                    'value': trade_value,
                    'commission': commission_paid,
                    'cash_after': cash
                })
                
            elif signal == -1 and position > 0:  # 卖出
                trade_value = position * current['close']
                commission_paid = trade_value * commission
                cash += (trade_value - commission_paid)
                
                trades.append({
                    'date': current.name,
                    'type': 'sell',
                    'price': current['close'],
                    'shares': position,
                    'value': trade_value,
                    'commission': commission_paid,
                    'cash_after': cash,
                    'profit': (current['close'] - self.entry_price) / self.entry_price * 100
                })
                
                position = 0
            
            # 更新投资组合价值
            portfolio_value.append(cash + (position * current['close'] if position > 0 else 0))
        
        # 计算回测结果
        returns = pd.Series(portfolio_value).pct_change().dropna()
        total_return = (portfolio_value[-1] / initial_capital - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1 if len(returns) > 0 else 0
        
        # 计算最大回撤
        peak = 0
        max_drawdown = 0
        for value in portfolio_value:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算夏普比率
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0
            
        # 计算胜率
        if len(trades) >= 2:
            winning_trades = len([t for t in trades[1::2] if t.get('profit', 0) > 0])  # 只计算卖出交易的盈利情况
            total_trades = len(trades) // 2
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        else:
            win_rate = 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_value[-1],
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades) // 2,
            'trades': trades
        }
    
    def plot_strategy(self, save_path: Optional[str] = None):
        """
        绘制策略表现图
        
        参数:
            save_path: 图片保存路径，如果为None则显示图片
        """
        if self.data is None or 'signal' not in self.data.columns:
            self.generate_signals()
            
        df = self.data.copy()
        
        # 创建图形和轴
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制价格和均线
        ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
        
        # 绘制均线
        for ma in self.ma_windows:
            ax1.plot(df.index, df[f'ma{ma}'], label=f'MA{ma}', linewidth=1)
        
        # 标记买卖信号
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, 
                   buy_signals['close'], 
                   color='red', 
                   label='Buy', 
                   marker='^', 
                   alpha=1)
        
        ax1.scatter(sell_signals.index, 
                   sell_signals['close'], 
                   color='green', 
                   label='Sell', 
                   marker='v', 
                   alpha=1)
        
        # 设置标题和标签
        ax1.set_title('Trading Strategy Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制成交量
        ax2.bar(df.index, df['volume'], color='gray', alpha=0.5, label='Volume')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图片
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


# 示例使用
if __name__ == "__main__":
    # 参数配置
    STOCK_CODE = "605199"
    START_DATE = "20240901"
    END_DATE = "20250709"
    
    print(f"正在分析 {STOCK_CODE} 从 {START_DATE} 到 {END_DATE} 的交易策略...")
    
    try:
        # 创建交易系统实例
        system = WisdomTradingSystemV2(
            ma_windows=[5, 10, 20, 50, 100],
            atr_period=14,
            atr_multiplier=2.0,
            position_size=0.2
        )
        
        # 加载数据
        system.load_data(STOCK_CODE, START_DATE, END_DATE)
        
        # 生成信号
        signals = system.generate_signals()
        
        # 回测
        results = system.backtest()
        
        # 打印回测结果
        print("\n回测结果:")
        print(f"初始资金: {results['initial_capital']:.2f}")
        print(f"最终资金: {results['final_value']:.2f}")
        print(f"总收益率: {results['total_return']:.2f}%")
        print(f"年化收益率: {results['annual_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"胜率: {results['win_rate']:.2f}%")
        print(f"交易次数: {results['num_trades']}")
        print("\n交易明细:")
        print("-" * 100)
        print(f"{'日期':<12} | {'类型':<6} | {'价格':<8} | {'数量':<8} | {'金额':<12} | {'佣金':<8} | {'盈亏%':<8} | {'资金':<12}")
        print("-" * 100)
        
        for i, trade in enumerate(results['trades']):
            if trade['type'] == 'buy':
                print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['type']:<6} | {trade['price']:8.2f} | {trade['shares']:8.2f} | {trade['value']:12.2f} | {trade['commission']:7.2f} | {'-':<8} | {trade['cash_after']:12.2f}")
            else:
                profit = trade.get('profit', 0)
                profit_color = '\033[91m' if profit < 0 else '\033[92m'  # 红色表示亏损，绿色表示盈利
                print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['type']:<6} | {trade['price']:8.2f} | {trade['shares']:8.2f} | {trade['value']:12.2f} | {trade['commission']:7.2f} | {profit_color}{profit:7.2f}%\033[0m | {trade['cash_after']:12.2f}")
        
        print("-" * 100)
        
        # 绘制策略图表
        system.plot_strategy()
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
