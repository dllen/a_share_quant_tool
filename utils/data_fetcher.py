import os
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


class AShareDataFetcher:
    """
    A股数据获取与处理类
    """
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_stock_data(self, stock_code, start_date=None, end_date=None):
        """
        获取单只股票的历史数据
        
        参数:
            stock_code (str): 股票代码，例如 '000001'
            start_date (str): 开始日期，格式 'YYYYMMDD'
            end_date (str): 结束日期，格式 'YYYYMMDD'
            
        返回:
            pd.DataFrame: 包含股票历史数据的DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
            
        try:
            # 使用akshare获取A股历史数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date,
                adjust="hfq"  # 前复权
            )
            
            # 重命名列以保持一致性
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
                '换手率': 'turnover_rate'
            })
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            return df
            
        except Exception as e:
            print(f"获取股票 {stock_code} 数据失败: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        
        参数:
            df (pd.DataFrame): 包含股票数据的DataFrame
            
        返回:
            pd.DataFrame: 添加了技术指标的DataFrame
        """
        if df is None or df.empty:
            return None
            
        # 计算移动平均线
        for ma in [5, 10, 20, 30, 60, 120, 200]:
            df[f'ma{ma}'] = df['close'].rolling(window=ma).mean()
        
        # 计算换手率均线
        df['turnover_ma5'] = df['turnover_rate'].rolling(window=5).mean()
        df['turnover_ma10'] = df['turnover_rate'].rolling(window=10).mean()
        
        return df
    
    def get_stock_with_indicators(self, stock_code, start_date=None, end_date=None):
        """
        获取股票数据并计算技术指标
        
        参数:
            stock_code (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            
        返回:
            pd.DataFrame: 包含技术指标的股票数据
        """
        df = self.get_stock_data(stock_code, start_date, end_date)
        if df is not None:
            df = self.calculate_technical_indicators(df)
        return df
