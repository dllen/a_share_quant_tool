"""
A-Share Market Data Loader

This module provides functionality to load A-share market data from various sources.
"""
import os
import pandas as pd
import akshare as ak
import tushare as ts


class AShareDataLoader:
    """A class to load A-share market data from different sources."""
    
    def __init__(self, data_dir: str = 'data/ashare'):
        """
        Initialize the AShareDataLoader.
        
        Args:
            data_dir: Directory to store cached data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize Tushare (if API key is available in environment)
        self.ts_pro = None
        if 'TUSHARE_TOKEN' in os.environ:
            ts.set_token(os.environ['TUSHARE_TOKEN'])
            self.ts_pro = ts.pro_api()
    
    def get_stock_data_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for an A-share stock using AKShare.
        
        Args:
            symbol: Stock symbol with exchange prefix (e.g., 'sh600000' for 浦发银行)
            start_date: Start date in 'YYYYMMDD' format
            end_date: End date in 'YYYYMMDD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol to AKShare format if needed
            if symbol.startswith(('sh', 'sz')):
                symbol = symbol.upper()
            
            # Get data from AKShare
            df = ak.stock_zh_a_hist(
                symbol=symbol[2:],  # Remove exchange prefix
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='hfq'  # 后复权
            )
            
            # Rename columns to standard format
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {symbol} using AKShare: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_data_tushare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for an A-share stock using Tushare (requires API key).
        
        Args:
            symbol: Stock symbol with exchange prefix (e.g., '600000.SH' for 浦发银行)
            start_date: Start date in 'YYYYMMDD' format
            end_date: End date in 'YYYYMMDD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.ts_pro:
            print("Tushare Pro API not initialized. Please set TUSHARE_TOKEN environment variable.")
            return pd.DataFrame()
            
        try:
            # Convert symbol to Tushare format if needed
            if not (symbol.endswith('.SH') or symbol.endswith('.SZ')):
                if symbol.startswith(('6', '9')):
                    symbol = f"{symbol}.SH"
                else:
                    symbol = f"{symbol}.SZ"
            
            # Get data from Tushare
            df = ts.pro_bar(
                ts_code=symbol,
                asset='E',
                adj='hfq',  # 后复权
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                print(f"No data returned from Tushare for {symbol}")
                return pd.DataFrame()
            
            # Sort by trade date ascending
            df = df.sort_values('trade_date')
            
            # Rename columns to standard format
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
                'pct_chg': 'pct_chg',
                'change': 'change',
                'turnover_rate': 'turnover'
            })
            
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {symbol} using Tushare: {str(e)}")
            return pd.DataFrame()
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for a market index.
        
        Args:
            index_code: Index code (e.g., 'sh000001' for 上证指数)
            start_date: Start date in 'YYYYMMDD' format
            end_date: End date in 'YYYYMMDD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Remove exchange prefix if present
            if index_code.startswith(('sh', 'sz')):
                index_code = index_code[2:]
            
            # Get index data from AKShare
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            
            # Filter by date range
            df = df[(df.index >= pd.to_datetime(start_date)) & 
                   (df.index <= pd.to_datetime(end_date))]
            
            # Rename columns to standard format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            print(f"Error loading index data for {index_code}: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_list(self, exchange: str = 'SSE') -> pd.DataFrame:
        """
        Get list of A-share stocks.
        
        Args:
            exchange: Exchange code ('SSE' for Shanghai, 'SZSE' for Shenzhen)
            
        Returns:
            DataFrame with stock list
        """
        try:
            if exchange.upper() == 'SSE':
                df = ak.stock_info_sh_name_code()
                df['exchange'] = 'SH'
            else:  # SZSE
                df = ak.stock_info_sz_name_code()
                df['exchange'] = 'SZ'
            
            return df
            
        except Exception as e:
            print(f"Error loading stock list for {exchange}: {str(e)}")
            return pd.DataFrame()


def example_usage():
    """Example usage of AShareDataLoader."""
    import matplotlib.pyplot as plt
    
    # Initialize data loader
    loader = AShareDataLoader()
    
    # Example 1: Get stock data using AKShare
    print("Loading stock data using AKShare...")
    df_ak = loader.get_stock_data_akshare(
        symbol='sh600000',  # 浦发银行
        start_date='20220101',
        end_date='20221231'
    )
    
    if not df_ak.empty:
        print(f"Loaded {len(df_ak)} rows of data for 600000.SH")
        print(df_ak.head())
        
        # Plot closing price
        plt.figure(figsize=(12, 6))
        plt.plot(df_ak.index, df_ak['close'])
        plt.title('600000.SH Close Price (2022)')
        plt.xlabel('Date')
        plt.ylabel('Price (RMB)')
        plt.grid(True)
        plt.show()
    
    # Example 2: Get index data
    print("\nLoading index data...")
    df_idx = loader.get_index_data(
        index_code='sh000001',  # 上证指数
        start_date='20220101',
        end_date='20221231'
    )
    
    if not df_idx.empty:
        print(f"Loaded {len(df_idx)} rows of data for 000001.SH")
        print(df_idx.head())
    
    # Example 3: Get stock list
    print("\nLoading stock list...")
    stocks = loader.get_stock_list('SSE')
    if not stocks.empty:
        print(f"Loaded {len(stocks)} stocks from SSE")
        print(stocks.head())


if __name__ == "__main__":
    example_usage()
