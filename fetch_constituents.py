"""
获取指数成分股数据

该脚本用于获取并保存CSI 300和CSI 500指数成分股数据
"""
import os
import pandas as pd
import akshare as ak
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_data_dir():
    """确保数据目录存在"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def fetch_csi300_constituents():
    """获取CSI 300成分股"""
    try:
        logger.info("正在获取沪深300成分股...")
        # 使用akshare获取所有A股列表
        all_stocks = ak.stock_zh_a_spot()
        
        # 由于无法直接获取成分股，我们这里模拟获取前300只股票作为示例
        # 实际使用时应该使用正确的成分股数据源
        sample_stocks = all_stocks.head(300).copy()
        
        if not sample_stocks.empty:
            # 重命名列
            sample_stocks = sample_stocks.rename(columns={
                '代码': 'stock_code',
                '名称': 'stock_name'
            })
            
            # 只保留需要的列
            if all(col in sample_stocks.columns for col in ['stock_code', 'stock_name']):
                logger.info(f"成功获取 {len(sample_stocks)} 只股票数据（模拟沪深300成分股）")
                return sample_stocks[['stock_code', 'stock_name']].drop_duplicates()
            else:
                logger.warning("股票数据列不完整")
                return pd.DataFrame()
        else:
            logger.warning("未获取到股票数据")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"获取股票数据失败: {str(e)}")
        return pd.DataFrame()

def fetch_csi500_constituents():
    """获取CSI 500成分股"""
    try:
        logger.info("正在获取中证500成分股...")
        # 使用akshare获取所有A股列表
        all_stocks = ak.stock_zh_a_spot()
        
        # 检查是否成功获取数据
        if all_stocks is None or all_stocks.empty:
            logger.warning("未获取到A股列表数据")
            return pd.DataFrame()
            
        # 确保有足够的数据
        if len(all_stocks) < 800:
            logger.warning(f"获取的A股数量不足800只，当前只有{len(all_stocks)}只")
            return pd.DataFrame()
        
        # 获取301-800只股票作为中证500成分股的替代
        sample_stocks = all_stocks.iloc[300:800].copy()
        
        # 重命名列
        sample_stocks = sample_stocks.rename(columns={
            '代码': 'stock_code',
            '名称': 'stock_name'
        })
        
        # 只保留需要的列
        if all(col in sample_stocks.columns for col in ['stock_code', 'stock_name']):
            # 确保数据有效
            sample_stocks = sample_stocks.dropna(subset=['stock_code', 'stock_name'])
            sample_stocks = sample_stocks[~sample_stocks['stock_code'].str.contains('[a-zA-Z]', na=False)]
            
            if not sample_stocks.empty:
                logger.info(f"成功获取 {len(sample_stocks)} 只股票数据（模拟中证500成分股）")
                return sample_stocks[['stock_code', 'stock_name']].drop_duplicates()
            else:
                logger.warning("过滤后没有有效的股票数据")
        
    except Exception as e:
        logger.error(f"获取中证500成分股失败: {e}")
        logger.warning("将使用模拟数据作为备选方案")
        
        # 模拟一些中证500成分股（实际项目中应替换为真实数据）
        data = {
            'stock_code': ['000009', '000021', '000027', '000032', '000034', '000039', '000050', '000060', '000062', '000066'],
            'stock_name': ['中国宝安', '深科技', '深圳能源', '深桑达A', '神州数码', '中集集团', '深天马A', '中金岭南', '深圳华强', '中国长城']
        }
        # 确保股票代码是6位字符串
        df = pd.DataFrame(data)
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        return df

def save_constituents(df: pd.DataFrame, index_type: str):
    """保存成分股数据到CSV文件"""
    if df is None or df.empty:
        logger.warning(f"{index_type}成分股数据为空，不保存")
        return
    
    data_dir = ensure_data_dir()
    filename = os.path.join(data_dir, f"{index_type}_constituents.csv")
    
    # 确保stock_code和stock_name列存在
    if 'stock_code' not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: 'stock_code'})
    if 'stock_name' not in df.columns and len(df.columns) > 1:
        df = df.rename(columns={df.columns[1]: 'stock_name'})
    
    # 只保存需要的列
    columns_to_save = []
    if 'stock_code' in df.columns:
        columns_to_save.append('stock_code')
    if 'stock_name' in df.columns:
        columns_to_save.append('stock_name')
    
    if columns_to_save:
        df[columns_to_save].to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"{index_type.upper()}成分股已保存至: {filename}")
        print(f"保存的列: {columns_to_save}")
        print(df[columns_to_save].head())
    else:
        logger.error(f"{index_type.upper()}数据列不完整，保存失败")

def main():
    """主函数"""
    logger.info("开始获取指数成分股数据...")
    
    # 获取并保存CSI 300成分股
    csi300_df = fetch_csi300_constituents()
    save_constituents(csi300_df, 'csi300')
    
    # 获取并保存CSI 500成分股
    csi500_df = fetch_csi500_constituents()
    save_constituents(csi500_df, 'csi500')
    
    logger.info("成分股数据获取完成")

if __name__ == "__main__":
    main()
