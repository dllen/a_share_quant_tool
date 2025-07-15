from typing import Dict, List

import numpy as np
import pandas as pd


class SmartMoneyStrategy:
    """
    基于《炒股的智慧》的交易策略
    核心原则：
    1. 顺势而为 - 只做上升趋势的股票
    2. 止损第一 - 严格止损控制风险
    3. 让利润奔跑 - 截断亏损，让利润奔跑
    4. 资金管理 - 控制仓位，分散风险
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
        初始化策略参数
        
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
        self.support_levels = []
        self.resistance_levels = []
        self.trend = "sideways"  # 'up', 'down', or 'sideways'
    
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
        if current_idx < 252:  # 确保有足够的历史数据
            return False
            
        # 获取近1年的最高价
        one_year_high = df['high'].iloc[current_idx-252:current_idx].max()
        
        # 检查当前价格是否创新高
        if df['close'].iloc[current_idx] < one_year_high:
            return False
            
        # 确认突破：过去lookback_days内有收盘价低于一年最高价
        for i in range(1, lookback_days + 1):
            if current_idx - i < 0:
                break
            if df['close'].iloc[current_idx - i] < one_year_high:
                return True
                
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
            
        # 计算窗口内的价格范围
        window_high = df['high'].iloc[current_idx-window:current_idx].max()
        window_low = df['low'].iloc[current_idx-window:current_idx].min()
        
        # 计算价格波动范围
        price_range = (window_high - window_low) / window_low
        
        return price_range <= threshold_pct
        
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
        if 'ma250' not in df.columns or current_idx < 250:
            return False
            
        current_price = df['close'].iloc[current_idx]
        annual_ma = df['ma250'].iloc[current_idx]
        
        return abs(current_price - annual_ma) / annual_ma <= threshold_pct
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含价格和指标数据的DataFrame
            
        返回:
            包含交易信号的DataFrame
        """
        df = data.copy()
        
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
        df['new_1y_high'] = False  # 是否创1年新高
        df['sideways'] = False  # 是否横盘
        df['near_annual_ma'] = False  # 是否在年线附近
        
        # 生成交易信号
        for i in range(max(self.ma_windows), len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1] if i > 0 else current
            
            # 判断当前趋势
            self.trend = self.determine_trend(df, i)
            df.loc[df.index[i], 'trend'] = self.trend
            
            # 记录特征状态
            df.loc[df.index[i], 'new_1y_high'] = self.is_new_1y_high(df, i)
            df.loc[df.index[i], 'sideways'] = self.is_sideways_consolidation(df, i)
            df.loc[df.index[i], 'near_annual_ma'] = self.is_near_annual_ma(df, i)
            
            # 检查是否接近支撑位或压力位
            near_support = self.is_near_support(current['close'])
            # near_resistance = self.is_near_resistance(current['close'])
            
            # 检查是否突破支撑位或压力位
            break_below_support = (current['low'] < current['support']) and (prev['low'] >= prev['support'])
            break_above_resistance = (current['high'] > current['resistance']) and (prev['high'] <= prev['resistance'])
            
            # 检查成交量是否放大
            volume_spiking = self.is_volume_spiking(df, i)
            
            # 生成买入信号
            buy_signal = False
            if self.trend == 'up':
                # 上升趋势中，支撑位附近或突破压力位时买入
                buy_signal = near_support or break_above_resistance
            
            # 检查是否满足1年新高突破后横盘放量卖出条件
            new_1y_high = self.is_new_1y_high(df, i)
            sideways = self.is_sideways_consolidation(df, i)
            near_annual_ma = self.is_near_annual_ma(df, i)
            
            # 生成卖出信号
            sell_signal = False
            if self.trend == 'down':
                # 下降趋势中，跌破支撑位时卖出
                sell_signal = break_below_support
            elif self.trend == 'sideways':
                # 横盘时，成交量放大时卖出
                sell_signal = volume_spiking
                
                # 1年新高后横盘放量卖出
                if new_1y_high and sideways and volume_spiking:
                    sell_signal = True
                    
            # 年线附近的买卖决策
            if near_annual_ma:
                if self.trend == 'up' and (near_support or break_above_resistance):
                    # 上升趋势中，年线附近支撑位买入或突破压力位买入
                    buy_signal = True
                elif self.trend == 'down' and break_below_support:
                    # 下降趋势中，年线下方支撑位跌破卖出
                    sell_signal = True
            
            # 设置信号
            if buy_signal:
                df.loc[df.index[i], 'signal'] = 1
                # 设置入场价格和初始止损
                self.entry_price = current['close']
                self.stop_loss = current['close'] * (1 - 0.03)  # 3% 初始止损
                self.trailing_stop = current['close'] - self.atr_multiplier * current['atr']
            elif sell_signal:
                df.loc[df.index[i], 'signal'] = -1
                # 重置交易状态
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.trailing_stop = 0.0
            
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
            
            # 如果当前没有信号，保持之前的信号
            if df.loc[df.index[i], 'signal'] == 0 and i > 0:
                df.loc[df.index[i], 'signal'] = df.loc[df.index[i-1], 'signal']
        
        return df
    
    def backtest(self, 
                data: pd.DataFrame, 
                initial_capital: float = 100000.0,
                commission: float = 0.0003) -> Dict:
        """
        回测策略表现
        
        参数:
            data: 包含价格、指标和信号的DataFrame
            initial_capital: 初始资金
            commission: 交易佣金率
            
        返回:
            包含回测结果的字典
        """
        if 'signal' not in data.columns:
            data = self.generate_signals(data)
        
        # 初始化回测变量
        position = 0
        cash = initial_capital
        portfolio_value = [initial_capital]
        trades = []
        
        # 执行回测
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            # 获取信号
            signal = data['signal'].iloc[i]
            
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
                    'cash_after': cash
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
            risk_free_rate = 0.03  # 假设无风险利率为3%
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_value[-1],
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'portfolio_value': pd.Series(portfolio_value, index=data.index[:len(portfolio_value)])
        }
