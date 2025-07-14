from typing import Dict, List

import numpy as np
import pandas as pd


class MATurnoverStrategy:
    """
    均线+换手率策略
    """
    def __init__(self, 
                 ma_windows: List[int] = [5, 10, 20, 60, 120],
                 turnover_ma_windows: List[int] = [5, 10],
                 turnover_threshold: float = 3.0):
        """
        初始化策略参数
        
        参数:
            ma_windows: 均线周期列表
            turnover_ma_windows: 换手率均线周期列表
            turnover_threshold: 换手率阈值（相对于均值的倍数）
        """
        self.ma_windows = sorted(ma_windows)
        self.turnover_ma_windows = sorted(turnover_ma_windows)
        self.turnover_threshold = turnover_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含价格和指标数据的DataFrame
            
        返回:
            包含交易信号的DataFrame
        """
        df = data.copy()
        
        # 确保所有需要的列都存在
        required_columns = ['close', 'turnover_rate']
        for ma in self.ma_windows:
            required_columns.append(f'ma{ma}')
        for ma in self.turnover_ma_windows:
            required_columns.append(f'turnover_ma{ma}')
            
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 初始化信号列
        df['signal'] = 0  # 1: 买入, -1: 卖出, 0: 持有
        
        # 生成交易信号
        for i in range(max(self.ma_windows), len(df)):
            # 获取当前数据点
            current = df.iloc[i]
            # prev = df.iloc[i-1] # 前一天的点
            
            # 获取当前和前一天的各均线值
            ma5, ma10, ma20, ma60, ma120 = [current[f'ma{ma}'] for ma in [5, 10, 20, 60, 120]]
            prev = df.iloc[i-1] if i > 0 else df.iloc[i]
            prev_ma5, prev_ma10, prev_ma20, prev_ma60, prev_ma120 = [prev[f'ma{ma}'] for ma in [5, 10, 20, 60, 120]]
            
            # 检查均线是否形成多头排列（5>10>20>60>120）
            ma_trend = (ma5 >= ma10 >= ma20 >= ma60 >= ma120)
            
            # 检查是否有均线金叉
            # 5日线上穿10日线
            cross_5_10 = (prev_ma5 <= prev_ma10) and (ma5 >= ma10)
            # 10日线上穿20日线
            cross_10_20 = (prev_ma10 <= prev_ma20) and (ma10 >= ma20)
            # 20日线上穿60日线
            cross_20_60 = (prev_ma20 <= prev_ma60) and (ma20 >= ma60)
            # 60日线上穿120日线
            cross_60_120 = (prev_ma60 <= prev_ma120) and (ma60 >= ma120)
            
            # 如果出现均线金叉，认为是买入信号
            golden_cross = cross_5_10 or cross_10_20 or cross_20_60 or cross_60_120
            
            # 计算换手率条件
            # turnover_condition = current['turnover_rate'] > current[f'turnover_ma{self.turnover_ma_windows[0]}'] * self.turnover_threshold
            
            # 生成买入信号 - 使用均线金叉作为主要标准
            if (golden_cross and                            # 出现均线金叉
                ma_trend and                                # 均线呈多头排列
                current['close'] > current['ma120']):       # 价格在120日均线上方
                df.loc[df.index[i], 'signal'] = 1
                
            # 生成卖出信号 - 当价格跌破120日均线时卖出
            elif (current['close'] < current['ma120'] and  # 当前价格低于120日均线
                  not ma_trend):
                df.loc[df.index[i], 'signal'] = -1
        
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
            prev = data.iloc[i-1]
            
            # 获取信号
            signal = data['signal'].iloc[i-1]
            
            # 执行交易
            if signal == 1 and position == 0:  # 买入
                position = cash / current['close'] * (1 - commission)
                cash = 0
                trades.append({
                    'date': current.name,
                    'type': 'buy',
                    'price': current['close'],
                    'shares': position,
                    'value': position * current['close']
                })
            elif signal == -1 and position > 0:  # 卖出
                cash = position * current['close'] * (1 - commission)
                trades.append({
                    'date': current.name,
                    'type': 'sell',
                    'price': current['close'],
                    'shares': position,
                    'value': cash
                })
                position = 0
            
            # 更新投资组合价值
            if position > 0:
                portfolio_value.append(position * current['close'])
            else:
                portfolio_value.append(cash)
        
        # 计算回测指标
        returns = pd.Series(portfolio_value).pct_change().dropna()
        total_return = (portfolio_value[-1] / initial_capital - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1 if len(returns) > 0 else 0
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() > 0 else 0
        max_drawdown = (pd.Series(portfolio_value) / pd.Series(portfolio_value).cummax() - 1).min() * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_value[-1],
            'total_return': total_return,
            'annual_return': annual_return * 100,  # 转换为百分比
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'portfolio_value': pd.Series(portfolio_value, index=data.index)
        }
