"""
Stock data utility functions for fetching market data.
"""
import pandas as pd
import akshare as ak
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_csi300_constituents() -> pd.DataFrame:
    """
    获取沪深300指数成分股
    
    Returns:
        pd.DataFrame: 包含沪深300成分股信息的DataFrame
        
    Example:
        >>> csi300 = get_csi300_constituents()
        >>> print(csi300.head())
    """
    return _get_index_constituents("000300", "CSI 300")

def get_csi500_constituents() -> pd.DataFrame:
    """
    获取中证500指数成分股
    
    Returns:
        pd.DataFrame: 包含中证500成分股信息的DataFrame
        
    Example:
        >>> csi500 = get_csi500_constituents()
        >>> print(csi500.head())
    """
    return _get_index_constituents("000905", "CSI 500")

def _get_index_constituents(index_code: str, index_name: str) -> pd.DataFrame:
    """
    获取指数成分股的内部实现函数
    
    Args:
        index_code: 指数代码
        index_name: 指数名称（用于日志）
    """
    try:
        logger.info(f"Fetching {index_name} constituents...")
        df = ak.index_stock_cons_csindex(index_code)
        
        if df.empty:
            logger.warning(f"No data returned for {index_name} constituents")
            return pd.DataFrame()
            
        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '指数代码': 'index_code',
            '指数名称': 'index_name',
            '指数英文名称': 'index_name_en',
            '成分券代码': 'stock_code',
            '成分券名称': 'stock_name',
            '成分券英文名称': 'stock_name_en',
            '交易所': 'exchange',
            '交易所英文名称': 'exchange_en'
        }
        
        # 只重命名存在的列
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # 添加交易所前缀
        if 'stock_code' in df.columns:
            df['full_code'] = df['stock_code'].apply(
                lambda x: f"sh{x}" if x.startswith(('5', '6', '9')) else f"sz{x}"
            )
        
        logger.info(f"Successfully fetched {len(df)} {index_name} constituents")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {index_name} constituents: {str(e)}")
        return pd.DataFrame()

def save_constituents_to_csv(index_type: str = "csi300", filename: str = None):
    """
    保存指数成分股到CSV文件
    
    Args:
        index_type (str): 指数类型，支持'csi300'或'csi500'
        filename (str): 保存的文件名，如果为None则自动生成
    """
    try:
        if index_type.lower() == "csi300":
            df = get_csi300_constituents()
            default_filename = "csi300_constituents.csv"
        elif index_type.lower() == "csi500":
            df = get_csi500_constituents()
            default_filename = "csi500_constituents.csv"
        else:
            logger.error(f"Unsupported index type: {index_type}. Use 'csi300' or 'csi500'")
            return
            
        if filename is None:
            filename = default_filename
            
        if not df.empty:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"{index_type.upper()} constituents saved to {filename}")
        else:
            logger.warning("No data to save")
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    print("=== 沪深300成分股 ===")
    csi300 = get_csi300_constituents()
    if not csi300.empty:
        print("\n沪深300成分股前5名：")
        print(csi300[['stock_code', 'stock_name', 'exchange']].head())
        save_constituents_to_csv("csi300")
    else:
        print("未能获取沪深300成分股数据")
    
    print("\n=== 中证500成分股 ===")
    csi500 = get_csi500_constituents()
    if not csi500.empty:
        print("\n中证500成分股前5名：")
        print(csi500[['stock_code', 'stock_name', 'exchange']].head())
        save_constituents_to_csv("csi500")
    else:
        print("未能获取中证500成分股数据")
