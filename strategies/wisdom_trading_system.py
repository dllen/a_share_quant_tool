from typing import Any, Dict

import akshare as ak
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class WisdomTradingSystem:
    """智慧交易系统，实现基于移动平均线的交易策略"""
    
    def __init__(self, stock_code: str, start: str, end: str):
        """
        初始化交易系统
        
        参数:
            stock_code (str): 股票代码，例如 '600519'
            start (str): 开始日期，格式 'YYYYMMDD'
            end (str): 结束日期，格式 'YYYYMMDD'
        """
        self.stock_code = stock_code
        self.start = start
        self.end = end
        self.position = 0  # 0: 空仓, 1: 持仓
        self.entry_price = 0
        self.high_after_buy = 0
        self.trades = []
        self.print_data = False
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """
        加载股票数据并计算技术指标
        
        返回:
            pd.DataFrame: 包含股票数据和技术指标的DataFrame
        """
        # 获取股票代码前缀（sh或sz）
        # 保留exchange变量用于未来可能的扩展
        
        print(f"\n{'='*50}")
        print(f"开始加载数据 - 股票: {self.stock_code}, 时间范围: {self.start} 到 {self.end}")
        print(f"{'='*50}")
        
        try:
            # 获取后复权数据
            print(f"\n调用akshare.stock_zh_a_hist() 获取数据...")
            print(f"股票代码: {self.stock_code}, 开始日期: {self.start}, 结束日期: {self.end}")
            
            # 确保股票代码是6位数字
            stock_code = str(self.stock_code).zfill(6)
            
            # 获取数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code, 
                period="daily", 
                start_date=self.start, 
                end_date=self.end, 
                adjust="qfq"
            )
            
            # 检查返回的数据
            if df is None:
                raise ValueError("akshare API返回了None，可能是网络问题或API限制")
                
            print(f"akshare API调用完成，返回数据形状: {df.shape if not df.empty else '空DataFrame'}")
            
            # 检查数据是否为空
            if df.empty:
                raise ValueError(f"未获取到股票 {stock_code} 在 {self.start} 到 {self.end} 期间的数据。请检查:"
                              f"\n1. 股票代码 {stock_code} 是否正确"
                              f"\n2. 日期范围 {self.start} 到 {self.end} 是否有效"
                              f"\n3. 该股票在此期间是否有交易数据")
            
            # 检查返回的DataFrame是否为空
            if df is None:
                raise ValueError("akshare API返回了None，可能是网络问题或API限制")
                
            if df.empty:
                raise ValueError(
                    f"\n{'!'*50}\n"
                    f"错误: 未获取到数据\n"
                    f"{'!'*50}\n"
                    f"可能的原因:\n"
                    f"1. 股票代码 {self.stock_code} 可能不存在或已退市\n"
                    f"2. 日期范围 {self.start} 到 {self.end} 可能没有交易数据\n"
                    f"3. 网络连接可能有问题\n"
                    f"4. 数据源可能暂时不可用\n"
                    f"{'!'*50}"
                )
                
            print(f"成功获取到 {len(df)} 条数据记录")
            print(f"数据列: {df.columns.tolist()}")
            if not df.empty:
                print(f"日期范围: {df['日期'].iloc[0]} 到 {df['日期'].iloc[-1]}")
            
            print(f"成功获取到 {len(df)} 条数据")
            print("原始列名:", df.columns.tolist())
            
            # 检查必要的列是否存在
            required_columns = {'日期', '开盘', '最高', '最低', '收盘', '成交量'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise KeyError(f"缺少必要的列: {missing_columns}. 实际列: {df.columns.tolist()}")
            
            # 重命名列 - 根据实际API响应调整列名映射
            column_mapping = {
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Amount',
                '振幅': 'Amplitude',
                '涨跌幅': 'ChangePercent',
                '涨跌额': 'Change',
                '换手率': 'Turnover'
            }
            df = df.rename(columns=column_mapping)
            
            print("重命名后列名:", df.columns.tolist())
            
            # 转换日期格式并设为索引
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True)  # 确保时间升序
            
            if self.print_data:
                # 打印数据统计信息
                print("\n" + "="*50)
                print(f"{' 数据统计信息 ':=^50}")
                print("="*50)
                print(f"股票代码: {self.stock_code}")
                print(f"时间范围: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
                print(f"交易日数: {len(df)} 天")
                print(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
                
                # 打印数据预览
                print("\n" + "-"*50)
                print(f"{' 数据预览 ':-^50}")
                print("-"*50)
                print(df.head().to_string())
                
                # 打印关键统计信息
                print("\n" + "-"*50)
                print(f"{' 关键统计信息 ':-^50}")
                print("-"*50)
                stats = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
                print(stats.to_string())
                
                # 打印涨跌幅信息
                if 'ChangePercent' in df.columns:
                    print("\n" + "-"*50)
                    print(f"{' 涨跌幅统计 ':-^50}")
                    print("-"*50)
                    print(f"最大单日涨幅: {df['ChangePercent'].max():.2f}%")
                    print(f"最大单日跌幅: {df['ChangePercent'].min():.2f}%")
                    print(f"平均日涨跌幅: {df['ChangePercent'].mean():.2f}%")
                    print(f"上涨天数: {len(df[df['ChangePercent'] > 0])} 天")
                    print(f"下跌天数: {len(df[df['ChangePercent'] < 0])} 天")
                    print(f"平盘天数: {len(df[df['ChangePercent'] == 0])} 天")
                
                # 打印成交额信息
                if 'Amount' in df.columns:
                    print("\n" + "-"*50)
                    print(f"{' 成交额统计 ':-^50}")
                    print("-"*50)
                    print(f"总成交额: {df['Amount'].sum()/1e8:.2f} 亿元")
                    print(f"日均成交额: {df['Amount'].mean()/1e8:.2f} 亿元")
                    print(f"最大日成交额: {df['Amount'].max()/1e8:.2f} 亿元 ({df['Amount'].idxmax().strftime('%Y-%m-%d')})")
                
            # 打印缺失值信息
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("\n" + "!"*50)
                print(f"{' 警告: 发现缺失值 ':=^50}")
                print("!"*50)
                for col, count in missing.items():
                    if count > 0:
                        print(f"{col}: {count} 个缺失值 ({count/len(df)*100:.2f}%)")
            else:
                print("\n✓ 数据完整，没有缺失值")
                
            print("\n" + "="*50 + "\n")
            
            # 计算技术指标
            print("\n计算技术指标...")
            df = self._calculate_technical_indicators(df)
            
            # 最终数据检查
            print("\n数据加载完成，最终数据检查:")
            print(f"数据形状: {df.shape}")
            print(f"数据列: {df.columns.tolist()}")
            print(f"日期范围: {df.index[0]} 到 {df.index[-1]}")
            print(f"数据样本:\n{df.head(2).to_string()}")
            print(f"\n{'='*50}\n数据加载完成\n{'='*50}")
            
            return df
            
        except Exception as e:
            error_msg = f"获取数据时出错: {str(e)}"
            print(f"\n{'!'*50}")
            print(f"{' 错误信息 ':=^50}")
            print(f"{error_msg}")
            print(f"{'!'*50}\n")
            
            # 尝试获取更详细的信息
            try:
                print("\n尝试获取数据示例...")
                test_codes = ['000001', '600000', '000333']  # 测试几个常用股票代码
                
                for test_code in test_codes:
                    try:
                        print(f"\n尝试获取股票 {test_code} 的示例数据...")
                        sample = ak.stock_zh_a_hist(
                            symbol=test_code, 
                            period="daily", 
                            start_date='20240101', 
                            end_date='20240110', 
                            adjust="qfq"
                        )
                        if sample is not None and not sample.empty:
                            print(f"\n{'-'*50}")
                            print(f"股票 {test_code} 示例数据:")
                            print(f"数据形状: {sample.shape}")
                            print(f"列名: {sample.columns.tolist()}")
                            print(f"日期范围: {sample.iloc[0]['日期']} 到 {sample.iloc[-1]['日期']}")
                            print(f"数据预览:\n{sample.head(2).to_string()}")
                            break
                        else:
                            print(f"股票 {test_code} 未返回数据")
                    except Exception as test_e:
                        print(f"测试股票 {test_code} 时出错: {str(test_e)}")
                else:
                    print("\n所有测试股票均未成功获取数据，可能是网络或API问题")
                    
            except Exception as e2:
                print(f"\n获取示例数据时发生错误: {str(e2)}")
            
            # 返回一个空的DataFrame而不是抛出异常
            print("\n返回空的DataFrame...")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            df (pd.DataFrame): 包含基础股票数据的DataFrame
            
        返回:
            pd.DataFrame: 添加了技术指标的DataFrame
        """
        # 计算移动平均线
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # 计算移动平均线交叉信号
        df['SMA20_50_Cross'] = np.where(df['SMA20'] > df['SMA50'], 1, -1)
        df['Signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 生成交易信号
        df.loc[df['SMA20_50_Cross'] > df['SMA20_50_Cross'].shift(1), 'Signal'] = 1   # 金叉买入
        df.loc[df['SMA20_50_Cross'] < df['SMA20_50_Cross'].shift(1), 'Signal'] = -1  # 死叉卖出
        
        # 计算每日收益率
        df['Daily_Return'] = df['Close'].pct_change()
        
        # 计算策略收益率
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        
        # 计算基准收益率
        df['Benchmark_Return'] = (1 + df['Daily_Return']).cumprod()
        
        return df.dropna()

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        计算策略表现指标
        
        返回:
            Dict[str, Any]: 包含各种性能指标的字典
        """
        if self.data.empty:
            return {}
            
        df = self.data.copy()
        
        # 总收益率
        total_return = df['Cumulative_Return'].iloc[-1] - 1
        
        # 年化收益率
        years = len(df) / 252  # 252个交易日
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # 最大回撤
        df['Max_Drawdown'] = df['Cumulative_Return'].div(df['Cumulative_Return'].cummax()).sub(1)
        max_drawdown = df['Max_Drawdown'].min()
        
        # 夏普比率（假设无风险收益率为3%）
        risk_free_rate = 0.03
        excess_returns = df['Strategy_Return'] - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # 胜率
        winning_trades = df[df['Strategy_Return'] > 0]['Strategy_Return'].count()
        total_trades = abs(df[df['Signal'] != 0]['Signal']).sum() / 2  # 每次完整交易包含一买一卖
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均盈利/亏损
        winning_returns = df[df['Strategy_Return'] > 0]['Strategy_Return']
        losing_returns = df[df['Strategy_Return'] < 0]['Strategy_Return']
        
        avg_win = winning_returns.mean() * 100 if not winning_returns.empty else 0
        avg_loss = abs(losing_returns.mean()) * 100 if not losing_returns.empty else 0
        
        # 盈亏比
        profit_factor = abs(winning_returns.sum() / losing_returns.sum()) if not losing_returns.empty else float('inf')
        
        return {
            'total_return': total_return * 100,  # 百分比
            'annual_return': annual_return * 100,
            'max_drawdown': abs(max_drawdown) * 100,  # 转为正数
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate * 100,
            'total_trades': int(total_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def plot_strategy(self, save_path: str = None):
        """
        绘制策略表现图表
        
        参数:
            save_path (str, optional): 保存图表的路径，如果为None则显示图表
        """
        if self.data.empty:
            print("没有可用的数据来绘制图表")
            return
            
        df = self.data.copy()
        
        # 设置图表样式
        plt.style.use('ggplot')  # 使用ggplot样式，这是一个内置的matplotlib样式
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 解决中文显示问题
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 创建图表
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        
        # 第一个子图：价格和移动平均线
        ax1 = fig.add_subplot(gs[0])
        
        # 绘制价格和移动平均线
        ax1.plot(df.index, df['Close'], label='收盘价', color='black', alpha=0.9, linewidth=1.5)
        ax1.plot(df.index, df['SMA20'], label='20日均线', color='blue', alpha=0.7, linewidth=1)
        ax1.plot(df.index, df['SMA50'], label='50日均线', color='orange', alpha=0.7, linewidth=1)
        
        # 标记买卖信号
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        ax1.scatter(
            buy_signals.index, 
            buy_signals['Close'],
            marker='^', 
            color='red', 
            label='买入信号', 
            alpha=1, 
            s=100,
            zorder=5
        )
        
        ax1.scatter(
            sell_signals.index, 
            sell_signals['Close'],
            marker='v', 
            color='green', 
            label='卖出信号', 
            alpha=1, 
            s=100,
            zorder=5
        )
        
        # 添加标题和标签
        ax1.set_title(f'股票 {self.stock_code} 交易策略表现 (移动平均线交叉策略)', fontsize=14, pad=20)
        ax1.set_ylabel('价格 (元)')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 第二个子图：成交量
        ax2 = fig.add_subplot(gs[1])
        
        # 绘制成交量
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.5, width=0.6)
        ax2.set_ylabel('成交量')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # 第三个子图：累计收益率
        ax3 = fig.add_subplot(gs[2])
        
        # 绘制累计收益率
        ax3.plot(df.index, df['Cumulative_Return'], label='策略收益', color='blue', linewidth=1.5)
        ax3.plot(df.index, df['Benchmark_Return'], label='买入持有', color='gray', linestyle='--', linewidth=1.5)
        
        # 添加0%基准线
        ax3.axhline(y=1, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 添加标题和标签
        ax3.set_title('累计收益率对比', fontsize=12, pad=10)
        ax3.set_ylabel('收益率')
        ax3.legend(loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # 格式化x轴日期
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_signals(self) -> pd.DataFrame:
        """
        生成交易信号
        
        返回:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if self.data.empty:
            return pd.DataFrame()
            
        df = self.data.copy()
        
        # 初始化信号列
        df['Buy'] = 0
        df['Sell'] = 0
        
        # 生成买入信号（金叉）
        df.loc[df['Signal'] == 1, 'Buy'] = 1
        
        # 生成卖出信号（死叉）
        df.loc[df['Signal'] == -1, 'Sell'] = 1
        
        return df[['Buy', 'Sell']]
    
    def backtest(self):
        """策略回测"""
        signals = self.generate_signals()
        df = self.data.join(signals)
        df['Return'] = df['Close'].pct_change()
        df['Strategy'] = 0
        
        position = 0
        for i in range(len(df)):
            if df['Buy'].iloc[i] and position == 0:
                position = 1
            elif df['Sell'].iloc[i] and position == 1:
                position = 0
            df['Strategy'].iloc[i] = df['Return'].iloc[i] * position
        
        # 计算累计收益
        df['Cumulative_Return'] = (1 + df['Strategy']).cumprod()
        
        # 计算基准收益（买入持有）
        df['Benchmark'] = (1 + df['Return']).cumprod()
        
        return df
    
    def plot_results(self):
        """可视化回测结果"""
        result = self.backtest()
        plt.figure(figsize=(14, 10))
        
        # 价格曲线
        plt.subplot(211)
        plt.plot(result['Close'], label='价格')
        plt.plot(result['SMA200'], label='200日均线', alpha=0.5)
        
        # 标注买卖点
        buy_signals = result[result['Buy'] == 1]
        sell_signals = result[result['Sell'] == 1]
        
        plt.scatter(buy_signals.index, 
                   buy_signals['Close'], 
                   marker='^', color='g', s=100, label='买入信号')
        plt.scatter(sell_signals.index, 
                   sell_signals['Close'], 
                   marker='v', color='r', s=100, label='卖出信号')
        
        # 标注交易类型
        for trade in self.trades:
            if '买入' in trade[0]:
                plt.annotate(trade[0], (trade[1], trade[2]), 
                            xytext=(trade[1], trade[2] * 0.95),
                            arrowprops=dict(facecolor='green', shrink=0.05))
            else:
                plt.annotate(trade[0], (trade[1], trade[2]), 
                            xytext=(trade[1], trade[2] * 1.05),
                            arrowprops=dict(facecolor='red', shrink=0.05))
        
        plt.title(f'{self.stock_code} 交易信号', fontsize=14)
        plt.ylabel('价格 (元)', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # 收益曲线
        plt.subplot(212)
        plt.plot(result['Cumulative_Return'], label='策略收益', color='b', linewidth=2)
        plt.plot(result['Benchmark'], label='买入持有收益', alpha=0.7, color='gray')
        
        # 标注最终收益
        final_strategy = result['Cumulative_Return'].iloc[-1]
        final_benchmark = result['Benchmark'].iloc[-1]
        
        plt.annotate(f'策略收益: {final_strategy:.2f}倍', 
                    (result.index[-1], final_strategy),
                    xytext=(result.index[-1] - pd.Timedelta(days=180), final_strategy * 0.8))
        
        plt.annotate(f'买入持有: {final_benchmark:.2f}倍', 
                    (result.index[-1], final_benchmark),
                    xytext=(result.index[-1] - pd.Timedelta(days=180), final_benchmark * 1.1))
        
        plt.title('累计收益对比', fontsize=14)
        plt.ylabel('收益倍数', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
        
        # 打印交易记录
        print("\n交易记录:")
        print(f"{'方向':<10} | {'日期':<12} | {'价格':<8} | {'备注'}")
        print("-" * 50)
        for trade in self.trades:
            print(f"{trade[0]:<10} | {trade[1].strftime('%Y-%m-%d'):<12} | {trade[2]:<8.2f} | ", end='')
            
            if '买入' in trade[0]:
                print("突破买入")
            elif '止损' in trade[0]:
                print("动态止损")
            elif '破位' in trade[0]:
                print("趋势破位")
            elif '止盈' in trade[0]:
                print("移动止盈")
        
        # 计算策略表现指标
        total_return = final_strategy - 1
        benchmark_return = final_benchmark - 1
        excess_return = total_return - benchmark_return
        
        win_rate = len([t for t in self.trades if "卖出" in t[0] and t[2] > self.entry_price]) / \
                  (len(self.trades) / 2) if len(self.trades) > 0 else 0
        
        print("\n策略表现统计:")
        print(f"总收益率: {total_return * 100:.2f}%")
        print(f"基准收益率: {benchmark_return * 100:.2f}%")
        print(f"超额收益率: {excess_return * 100:.2f}%")
        print(f"交易次数: {len(self.trades) // 2} 次")
        print(f"胜率: {win_rate * 100:.2f}%")

# 示例使用
if __name__ == "__main__":
    # 参数配置
    STOCK_CODE = "605199"
    START_DATE = "20240101"
    END_DATE = "20250709"  # 更新到当前日期
    
    print(f"正在分析 {STOCK_CODE} 从 {START_DATE} 到 {END_DATE} 的交易策略...")
    
    try:
        # 创建交易系统实例
        system = WisdomTradingSystem(STOCK_CODE, START_DATE, END_DATE)
        
        # 生成交易信号
        signals = system.generate_signals()
        
        # 计算并显示策略表现指标
        metrics = system.calculate_metrics()
        print("\n=== 策略表现指标 ===")
        print(f"总收益率: {metrics.get('total_return', 0):.2f}%")
        print(f"年化收益率: {metrics.get('annual_return', 0):.2f}%")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"胜率: {metrics.get('win_rate', 0):.2f}%")
        print(f"总交易次数: {metrics.get('total_trades', 0)}")
        print(f"平均盈利: {metrics.get('avg_win', 0):.2f}%")
        print(f"平均亏损: {metrics.get('avg_loss', 0):.2f}%")
        print(f"盈亏比: {metrics.get('profit_factor', 0):.2f}")
        
        # 绘制策略图表
        # output_path = f"{STOCK_CODE}_strategy_analysis.png"
        system.plot_strategy(None)
        
        # print(f"\n分析完成！策略图表已保存至: {output_path}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()