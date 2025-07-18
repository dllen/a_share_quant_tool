"""
A-Share Market Data Loader

This module provides functionality to load A-share market data from various sources.
"""
import os

import akshare as ak
import pandas as pd


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
        
    
    def get_stock_data_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for an A-share stock using AKShare.
        
        Args:
            symbol: Stock symbol with or without exchange prefix (e.g., '600000' or 'sh600000' for 浦发银行)
            start_date: Start date in 'YYYYMMDD' format
            end_date: End date in 'YYYYMMDD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Clean and validate the symbol
            original_symbol = str(symbol).strip()
            symbol = original_symbol.lower()
            
            print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
            
            # Get data from AKShare
            def try_akshare_method(method_name, **kwargs):
                """Helper function to try different AKShare methods with error handling"""
                try:
                    method = getattr(ak, method_name, None)
                    if not method:
                        print(f"Method {method_name} not found in AKShare")
                        return None
                    
                    print(f"Trying {method_name}...")
                    df = method(**kwargs)
                    
                    if df is None or df.empty:
                        print(f"{method_name} returned no data")
                        return None
                        
                    print(f"{method_name} returned {len(df)} rows with columns: {df.columns.tolist()}")
                    return df
                    
                except Exception as e:
                    print(f"{method_name} failed: {str(e)}")
                    return None
            
            # Try the primary method first
            df = try_akshare_method(
                'stock_zh_a_hist',
                symbol=symbol,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='hfq'
            )
            
            # If primary method failed, try alternative methods
            if df is None or df.empty:
                print("Primary method failed, trying alternative methods...")
                
                # Try to get spot data to verify stock exists
                spot_data = try_akshare_method('stock_zh_a_spot_em')
                if spot_data is not None:
                    stock_info = spot_data[spot_data['代码'] == symbol]
                    if not stock_info.empty:
                        print(f"Stock {symbol} exists in spot data: {stock_info['名称'].iloc[0]}")
                        
                        # Try with explicit market prefix (sh for 6/9, sz for 0/3)
                        market_prefix = 'sh' if symbol[0] in ['6', '9'] else 'sz'
                        print(f"Trying with explicit market prefix {market_prefix}...")
                        
                        # First try stock_zh_a_hist_tx which sometimes works better
                        df = try_akshare_method(
                            'stock_zh_a_hist_tx',
                            symbol=f"{market_prefix}{symbol}",  # Add market prefix
                            start_date=start_date,
                            end_date=end_date,
                            adjust='hfq'
                        )
                        
                        # If that fails, try stock_zh_a_hist_em
                        if df is None or df.empty:
                            df = try_akshare_method(
                                'stock_zh_a_hist_em',
                                symbol=f"{market_prefix}{symbol}",  # Add market prefix
                                period='daily',
                                start_date=start_date,
                                end_date=end_date,
                                adjust='hfq'
                            )
                        
                        # If still no data, try the original method
                        if df is None or df.empty:
                            df = try_akshare_method(
                                'stock_zh_a_hist',
                                symbol=f"{market_prefix}{symbol}",  # Add market prefix
                                period='daily',
                                start_date=start_date,
                                end_date=end_date,
                                adjust='hfq'
                            )
                        
                        # If still no data, try without adjustment
                        if df is None or df.empty:
                            df = try_akshare_method(
                                'stock_zh_a_hist_em',
                                symbol=f"{market_prefix}{symbol}",  # Add market prefix
                                period='daily',
                                start_date=start_date,
                                end_date=end_date,
                                adjust=''  # Try without adjustment
                            )
                            
                            if df is None or df.empty:
                                df = try_akshare_method(
                                    'stock_zh_a_hist',
                                    symbol=f"{market_prefix}{symbol}",  # Add market prefix
                                    period='daily',
                                    start_date=start_date,
                                    end_date=end_date,
                                    adjust=''  # Try without adjustment
                                )
                    else:
                        print(f"Warning: Stock {symbol} not found in spot data")
                
                # Try alternative method with different parameters if still no data
                if df is None or df.empty:
                    print("Trying with original symbol and no adjustment...")
                    df = try_akshare_method(
                        'stock_zh_a_hist',
                        symbol=symbol,
                        period='daily',
                        start_date=start_date,
                        end_date=end_date,
                        adjust=''  # Try without adjustment
                    )
            
            # If we still don't have data, return empty DataFrame
            if df is None or df.empty:
                print(f"Warning: No data available for {original_symbol} using any AKShare method")
                return pd.DataFrame()
            
            # Standardize column names with priority to English column names
            column_mapping = {
                # Chinese column names
                '日期': 'date', '时间': 'date',
                '开盘': 'open', '开盘价': 'open',
                '最高': 'high', '最高价': 'high',
                '最低': 'low', '最低价': 'low',
                '收盘': 'close', '收盘价': 'close',
                '成交量': 'volume', '成交数量': 'volume', 'vol': 'volume',
                '成交额': 'amount', '成交金额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg', '涨跌(%)': 'pct_chg', 'pct_chg': 'pct_chg',
                '涨跌额': 'change', '涨跌': 'change', 'change': 'change',
                '换手率': 'turnover', 'turnover': 'turnover',
                # English column names (as fallback)
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'
            }
            
            # Rename columns to standard format, only including columns that exist
            rename_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=rename_columns)
            
            # Ensure all required columns exist, fill missing ones with NaN
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'volume':
                        # For volume, try to use amount/close as an approximation if available
                        if 'amount' in df.columns and 'close' in df.columns:
                            df['volume'] = (df['amount'] / df['close']).round().astype(int)
                        else:
                            df['volume'] = 0  # Default to 0 if volume data is not available
                    else:
                        df[col] = pd.NA  # Use NA for other missing columns
            
            # Check if we have the date column
            if 'date' not in df.columns:
                print("Warning: 'date' column not found in AKShare response. Using index as date.")
                df = df.reset_index()
                
                # Try to find a date column if not named 'date'
                date_cols = [col for col in df.columns if 'date' in col.lower() or '时间' in col]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: 'date'})
                else:
                    print("No date column found, using index as date")
                    df['date'] = df.index
            
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            print(f"Successfully loaded {len(df)} rows of data for {symbol} from AKShare")
            return df
                
        except Exception as e:
            print(f"Error loading data for {symbol} using AKShare: {str(e)}")
            import traceback
            print(traceback.format_exc())
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
        symbol='600000',  # 浦发银行
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
        index_code='000001',  # 上证指数
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
