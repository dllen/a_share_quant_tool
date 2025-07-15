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
                 ma_windows: List[int] = [5, 10, 20, 50, 200, 250],
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 position_size: float = 0.1,
                 support_window: int = 250,  # 年线(250日)作为支撑线
                 resistance_window: int = 20,  # 月线(20日)作为压力线
                 support_threshold: float = 0.02,
                 resistance_threshold: float = 0.02,
                 volume_ma_window: int = 20,  # 成交量均线窗口
                 volume_spike_multiplier: float = 1.5):  # 成交量放大的倍数
        """
        初始化交易系统
        
        参数:
            ma_windows: 均线周期列表，用于判断趋势
            atr_period: ATR计算周期，用于止损
            atr_multiplier: ATR乘数，用于计算止损位
            position_size: 每次交易的仓位比例
            support_window: 支撑线周期(250日年线)
            resistance_window: 压力线周期(20日月线)
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
        self.volume_ma_window = volume_ma_window
        self.volume_spike_multiplier = volume_spike_multiplier
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.trailing_stop = 0.0
        self.data = None
        self.trades = []
        self.support_levels = []
        self.resistance_levels = []
        self.trend = "sideways"  # 'up', 'down', or 'sideways'
    
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
        
    def find_support_resistance(self, data: pd.DataFrame, num_points: int = 5) -> tuple:
        """
        识别支撑位和压力位
        
        参数:
            data: 包含价格数据的DataFrame
            num_points: 用于计算平均值的高低点数量
            
        返回:
            tuple: (support_levels, resistance_levels)
            
        说明:
            - 支撑位: 最近1年内几个最低点的平均值
            - 压力位: 最近1年内几个最高点的平均值
        """
        df = data.copy()
        
        # 确保数据按日期升序排列
        df = df.sort_index()
        
        # 获取最近1年的数据
        one_year_ago = df.index[-1] - pd.DateOffset(years=1)
        recent_data = df[df.index >= one_year_ago]
        
        if len(recent_data) < num_points:
            # 如果没有足够的数据点，使用全局最小/最大值
            support_level = df['low'].min()
            resistance_level = df['high'].max()
        else:
            # 计算支撑位：最近1年内几个最低点的平均值
            lowest_points = recent_data['low'].nsmallest(num_points).mean()
            
            # 计算压力位：最近1年内几个最高点的平均值
            highest_points = recent_data['high'].nlargest(num_points).mean()
            
            support_level = lowest_points
            resistance_level = highest_points
        
        # 创建与原始数据索引相同的Series
        support_levels = pd.Series(support_level, index=df.index)
        resistance_levels = pd.Series(resistance_level, index=df.index)
        
        # 存储支撑位和压力位
        self.support_levels = [support_level] * len(df)
        self.resistance_levels = [resistance_level] * len(df)
        
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
    
    def determine_trend(self, df: pd.DataFrame, current_idx: int, short_window: int = 20, long_window: int = 50) -> str:
        """
        判断当前市场趋势
        
        参数:
            df: 包含价格数据的DataFrame
            current_idx: 当前索引
            short_window: 短期均线窗口
            long_window: 长期均线窗口
            
        返回:
            str: 'up', 'down', 或 'sideways' 表示当前趋势
        """
        if current_idx < long_window:
            return 'sideways'
            
        # 计算均线
        short_ma = df['close'].iloc[current_idx-short_window+1:current_idx+1].mean()
        long_ma = df['close'].iloc[current_idx-long_window+1:current_idx+1].mean()
        
        # 计算价格在均线附近的波动范围
        price_volatility = df['close'].iloc[current_idx-20:current_idx].std()
        
        # 判断趋势
        if short_ma > long_ma * 1.02:  # 短期均线明显高于长期均线
            return 'up'
        elif short_ma < long_ma * 0.98:  # 短期均线明显低于长期均线
            return 'down'
        else:
            # 如果价格波动较小，则认为是横盘
            if price_volatility < df['close'].iloc[current_idx] * 0.01:
                return 'sideways'
            return 'sideways'
            
    def is_volume_spiking(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        判断成交量是否放大
        
        参数:
            df: 包含成交量数据的DataFrame
            current_idx: 当前索引
            
        返回:
            bool: 如果成交量放大则返回True
        """
        if current_idx < self.volume_ma_window:
            return False
            
        current_volume = df['volume'].iloc[current_idx]
        volume_ma = df['volume'].iloc[current_idx-self.volume_ma_window:current_idx].mean()
        
        return current_volume > volume_ma * self.volume_spike_multiplier
        
    def is_new_1y_high(self, df: pd.DataFrame, current_idx: int, lookback_days: int = 5) -> bool:
        """
        判断是否创出近1年新高
        
        参数:
            df: 包含价格数据的DataFrame
            current_idx: 当前索引
            lookback_days: 确认突破的天数
            
        返回:
            bool: 如果是近1年新高则返回True
        """
        if current_idx < 250:  # 至少需要250个交易日的数据
            return False
            
        current_high = df['high'].iloc[current_idx]
        one_year_ago = max(0, current_idx - 250)
        
        # 检查是否是近1年新高
        is_new_high = current_high > df['high'].iloc[one_year_ago:current_idx].max()
        
        # 确认突破：过去几天持续在1年高点附近
        if is_new_high and current_idx > lookback_days:
            recent_highs = df['high'].iloc[current_idx-lookback_days:current_idx+1]
            return all(recent_highs >= 0.99 * current_high)
            
        return False
        
    def is_sideways_consolidation(self, df: pd.DataFrame, current_idx: int, window: int = 10, threshold_pct: float = 0.02) -> bool:
        """
        判断是否处于横盘整理状态
        
        参数:
            df: 包含价格数据的DataFrame
            current_idx: 当前索引
            window: 观察窗口大小
            threshold_pct: 价格波动阈值（百分比）
            
        返回:
            bool: 如果处于横盘整理则返回True
        """
        if current_idx < window:
            return False
            
        prices = df['close'].iloc[current_idx-window:current_idx+1]
        price_range = prices.max() - prices.min()
        avg_price = prices.mean()
        
        # 如果价格波动范围小于阈值，则认为是横盘
        return (price_range / avg_price) < threshold_pct
        
    def is_near_annual_ma(self, df: pd.DataFrame, current_idx: int, threshold_pct: float = 0.05) -> bool:
        """
        判断价格是否在年线附近
        
        参数:
            df: 包含价格数据的DataFrame
            current_idx: 当前索引
            threshold_pct: 年线附近的阈值（百分比）
            
        返回:
            bool: 如果在年线附近则返回True
        """
        if current_idx < 250:  # 年线需要250个交易日
            return False
            
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
        df['support'] = support_levels
        df['resistance'] = resistance_levels
        
        # 初始化信号列
        df['signal'] = 0  # 1: 买入, -1: 卖出, 0: 持有
        df['trend'] = 'sideways'  # 趋势标记
        
        # 跟踪1年新高后的横盘状态
        new_high_recently = False
        new_high_price = 0.0
        new_high_count = 0
        
        # 生成交易信号
        for i in range(max(self.ma_windows), len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1] if i > 0 else current
            
            # 判断当前趋势
            self.trend = self.determine_trend(df, i)
            df.loc[df.index[i], 'trend'] = self.trend
            
            # 检查是否接近支撑位或压力位
            near_support = self.is_near_support(current['close'])
            near_resistance = self.is_near_resistance(current['close'])
            
            # 检查是否突破支撑位或压力位
            break_below_support = (current['low'] < current['support']) and (prev['low'] >= prev['support'])
            break_above_resistance = (current['high'] > current['resistance']) and (prev['high'] <= prev['resistance'])
            
            # 检查成交量是否放大
            volume_spiking = self.is_volume_spiking(df, i)
            
            # 检查是否创出1年新高
            is_new_high = self.is_new_1y_high(df, i)
            if is_new_high and not new_high_recently:
                new_high_recently = True
                new_high_price = current['high']
                new_high_count = 0
            
            # 检查1年新高后的横盘状态
            new_high_consolidation = False
            if new_high_recently:
                new_high_count += 1
                # 检查是否出现横盘整理（价格在新高附近±2%范围内）
                if (current['high'] <= new_high_price * 1.02 and 
                    current['low'] >= new_high_price * 0.98 and
                    self.is_sideways_consolidation(df, i)):
                    new_high_consolidation = True
                
                # 如果超过20个交易日或价格明显下跌，则重置状态
                if new_high_count > 20 or current['close'] < new_high_price * 0.95:
                    new_high_recently = False
            
            # 检查是否在年线附近
            near_annual_ma = self.is_near_annual_ma(df, i)
            
            # 生成买入信号
            buy_signal = False
            if self.trend == 'up':
                # 上升趋势中，支撑位附近或突破压力位时买入
                buy_signal = near_support or break_above_resistance
                
                # 年线附近且趋势向上，支撑位附近买入
                if near_annual_ma and near_support:
                    buy_signal = True
            
            # 生成卖出信号
            sell_signal = False
            if self.trend == 'down':
                # 下降趋势中，跌破支撑位时卖出
                sell_signal = break_below_support
                
                # 年线附近且趋势向下，压力位附近卖出
                if near_annual_ma and near_resistance:
                    sell_signal = True
            elif self.trend == 'sideways':
                # 横盘时，成交量放大时卖出
                sell_signal = volume_spiking
                
                # 1年新高后横盘且成交量放大，卖出
                if new_high_consolidation and volume_spiking:
                    sell_signal = True
            
            # 设置信号
            if buy_signal:
                df.loc[df.index[i], 'signal'] = 1
                # 设置入场价格和初始止损
                self.entry_price = current['close']
                self.stop_loss = current['close'] * (1 - 0.03)  # 3% 初始止损
                self.trailing_stop = current['close'] - self.atr_multiplier * current['atr']
                
                # 重置1年新高状态
                new_high_recently = False
            elif sell_signal:
                df.loc[df.index[i], 'signal'] = -1
                # 重置交易状态
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.trailing_stop = 0.0
                
                # 重置1年新高状态
                new_high_recently = False
            
            # 管理已有仓位
            if i > 0 and df['signal'].iloc[i-1] == 1:  # 已有仓位
                # 更新跟踪止损
                if self.trailing_stop > 0:
                    self.trailing_stop = max(
                        self.trailing_stop,
                        current['close'] - self.atr_multiplier * current['atr']
                    )
                
                # 检查是否触发止损
                if current['low'] <= self.trailing_stop or current['low'] <= self.stop_loss:
                    df.loc[df.index[i], 'signal'] = -1
                    # 重置交易状态
                    self.entry_price = 0.0
                    self.stop_loss = 0.0
                    self.trailing_stop = 0.0
                    
                    # 重置1年新高状态
                    new_high_recently = False
                
                # 如果接近压力位且盈利超过5%，考虑止盈
                if near_resistance and current['close'] > self.entry_price * 1.05:
                    df.loc[df.index[i], 'signal'] = -1  # 平仓
                    # 重置交易状态
                    self.entry_price = 0.0
                    self.stop_loss = 0.0
                    self.trailing_stop = 0.0
                    
                    # 重置1年新高状态
                    new_high_recently = False
            
            # 如果当前没有信号，保持之前的信号
            if df.loc[df.index[i], 'signal'] == 0 and i > 0:
                df.loc[df.index[i], 'signal'] = df.loc[df.index[i-1], 'signal']
        
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

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
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
