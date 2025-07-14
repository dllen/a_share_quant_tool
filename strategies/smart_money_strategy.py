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
                 ma_windows: List[int] = [5, 10, 20],
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 position_size: float = 0.1,
                 support_window: int = 20,
                 resistance_window: int = 20,
                 support_threshold: float = 0.02,
                 resistance_threshold: float = 0.02):
        """
        初始化策略参数
        
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
        self.support_levels = []
        self.resistance_levels = []
    
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
            
            # 卖出条件：趋势转弱或接近压力位
            sell_condition = False
            if df['signal'].iloc[i-1] == 1:  # 已有仓位
                sell_condition = (
                    current['close'] < self.trailing_stop or  # 触发止损
                    not is_uptrend or                       # 趋势转弱
                    (near_resistance and not price_making_new_high)  # 接近压力位且动能减弱
                )
            
            # 生成信号
            if buy_condition:
                df.loc[df.index[i], 'signal'] = 1
                self.entry_price = current['close']
                self.stop_loss = current['close'] - (self.atr_multiplier * current['atr'])
                self.trailing_stop = self.stop_loss
            elif sell_condition:
                df.loc[df.index[i], 'signal'] = -1
            
            # 更新跟踪止损
        for i in range(1, len(df)):
            if df['signal'].iloc[i-1] == 1:  # 持有仓位
                current = df.iloc[i]
                
                # 更新跟踪止损：最高价的回撤2倍ATR
                high_since_entry = df.loc[df.index[i-1]:df.index[i], 'high'].max()
                self.trailing_stop = max(self.trailing_stop, high_since_entry - (self.atr_multiplier * current['atr']))
                
                # 如果价格低于跟踪止损，平仓
                if current['close'] < self.trailing_stop:
                    df.loc[df.index[i], 'signal'] = -1  # 平仓
        
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
