"""
组合策略分析器：结合WisdomTradingSystem和SmartMoneyStrategy

该脚本用于：
1. 读取CSI 300和CSI 500成分股列表
2. 对每只股票应用两种策略
3. 生成买卖信号
4. 输出信号汇总
"""
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
from tqdm import tqdm

from strategies.smart_money_strategy import SmartMoneyStrategy
from strategies.wisdom_trading_system_v2 import WisdomTradingSystemV2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CombinedStrategyAnalyzer:
    """组合策略分析器"""
    
    def __init__(self, 
                 data_dir: str = 'data',
                 support_window: int = 20,
                 resistance_window: int = 20,
                 support_threshold: float = 0.02,
                 resistance_threshold: float = 0.02):
        """
        初始化分析器
        
        参数:
            data_dir: 数据目录，用于存储成分股CSV文件
            support_window: 支撑线检测窗口大小
            resistance_window: 压力线检测窗口大小
            support_threshold: 支撑位附近买入阈值（百分比）
            resistance_threshold: 压力位附近卖出阈值（百分比）
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化策略
        self.wisdom_system = WisdomTradingSystemV2(
            ma_windows=[5, 10, 20, 50, 100],
            support_window=support_window,
            resistance_window=resistance_window,
            support_threshold=support_threshold,
            resistance_threshold=resistance_threshold,
            atr_period=14,
            atr_multiplier=2.0,
            position_size=0.2
        )

        self.smart_money = SmartMoneyStrategy(
            ma_windows=[5, 10, 20],
            support_window=support_window,
            resistance_window=resistance_window,
            support_threshold=support_threshold,
            resistance_threshold=resistance_threshold,
            atr_period=14,
            atr_multiplier=2.0,
            position_size=0.1
        )
        
        # 存储结果
        self.signals = []
        
    def _get_stock_name(self, stock_code: str) -> str:
        """
        使用akshare获取股票名称
        
        参数:
            stock_code: 股票代码
            
        返回:
            str: 股票名称，如果获取失败则返回'股票{code}'
        """
        def try_get_name_from_individual_info():
            try:
                stock_info = ak.stock_individual_info_em(symbol=stock_code)
                
                if stock_info is None:
                    logger.warning(f"未获取到股票 {stock_code} 的信息: 返回值为None")
                    return None
                    
                logger.info(f"获取股票 {stock_code} 的信息: {stock_info}")

                # 处理DataFrame返回
                if isinstance(stock_info, pd.DataFrame) and not stock_info.empty:
                    try:
                        # 查找'item'列为'股票简称'的行，并获取对应的'value'列值
                        if 'item' in stock_info.columns and 'value' in stock_info.columns:
                            name_row = stock_info[stock_info['item'] == '股票简称']
                            if not name_row.empty:
                                name = name_row['value'].iloc[0] if len(name_row) > 0 else None
                                if pd.notna(name) and str(name).strip():
                                    return str(name).strip()
                    except Exception as df_e:
                        logger.warning(f"处理DataFrame数据时出错 (股票 {stock_code}): {str(df_e)}"
                                    f"\nDataFrame内容:\n{stock_info}")
                
                # 处理其他可能的数据类型
                elif isinstance(stock_info, (dict, pd.Series)):
                    # 尝试从字典或Series中获取股票名称
                    for key in ['股票简称', 'name', '证券简称']:
                        if key in stock_info:
                            name = stock_info[key]
                            if pd.notna(name) and str(name).strip():
                                return str(name).strip()
                
                # 如果以上方法都失败，尝试直接转换为字符串
                try:
                    if hasattr(stock_info, '__str__') and str(stock_info).strip():
                        return str(stock_info).strip()
                except (AttributeError, ValueError, TypeError) as e:
                    logger.debug(f"Error converting stock info to string: {e}")
                    pass
                    
            except Exception as e:
                logger.warning(f"从stock_individual_info_em获取股票 {stock_code} 名称时出错: {str(e)}")
                
            return None
            
        def try_get_name_from_spot():
            try:
                stock_spot = ak.stock_zh_a_spot()
                if not stock_spot.empty and '代码' in stock_spot.columns:
                    stock_data = stock_spot[stock_spot['代码'] == stock_code]
                    if not stock_data.empty and '名称' in stock_data.columns:
                        name = stock_data['名称'].iloc[0]
                        if pd.notna(name) and str(name).strip():
                            return str(name).strip()
            except Exception as e:
                logger.warning(f"从stock_zh_a_spot获取股票 {stock_code} 名称时出错: {str(e)}")
            return None
        
        # 首先尝试从个股信息接口获取
        name = try_get_name_from_individual_info()
        if name:
            return name
            
        # 如果失败，尝试从实时行情接口获取
        name = try_get_name_from_spot()
        if name:
            return name
            
        logger.warning(f"无法从API获取股票 {stock_code} 的名称")
        return f'股票{stock_code}'  # 失败时返回默认名称
        
    def load_constituents(self, index_type: str = 'csi300') -> pd.DataFrame:
        """
        加载指数成分股
        
        参数:
            index_type: 指数类型，'csi300' 或 'csi500'
            
        返回:
            pd.DataFrame: 包含stock_code和stock_name的DataFrame
        """
        try:
            # 直接从项目根目录读取CSV文件
            filename = f"{index_type}_constituents.csv"
            if not os.path.exists(filename):
                logger.error(f"成分股文件不存在: {filename}")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
                
            # 读取CSV文件，处理可能的编码问题
            try:
                df = pd.read_csv(filename, usecols=['stock_code', 'stock_name'])
            except UnicodeDecodeError:
                # 尝试不同的编码
                df = pd.read_csv(filename, usecols=['stock_code', 'stock_name'], encoding='gbk')
            
            # 检查是否成功读取数据
            if df.empty:
                logger.warning(f"成分股文件 {filename} 为空")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
            
            # 检查必要的列是否存在
            if 'stock_code' not in df.columns:
                logger.error(f"成分股文件 {filename} 缺少必要的列: 'stock_code'")
                return pd.DataFrame(columns=['stock_code', 'stock_name'])
            
            # 确保stock_code是字符串类型并去除空格
            df['stock_code'] = df['stock_code'].astype(str).str.strip()
            
            # 如果stock_name列不存在，则创建一个空列
            if 'stock_name' not in df.columns:
                df['stock_name'] = '未知'
            
            logger.info(f"成功加载 {len(df)} 只{index_type.upper()}成分股")
            return df[['stock_code', 'stock_name']].drop_duplicates()
            
        except Exception as e:
            logger.error(f"加载成分股数据时发生错误: {str(e)}")
            logger.exception("详细错误信息:")
            return pd.DataFrame(columns=['stock_code', 'stock_name'])
    
    def analyze_stock(self, stock_code: str, start_date: str = None, end_date: str = None) -> Optional[Dict]:
        """
        分析单只股票
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期，格式'YYYYMMDD'，如果为None则使用两周前
            end_date: 结束日期，格式'YYYYMMDD'，如果为None则使用昨天
            
        返回:
            Dict: 包含分析结果的字典，如果分析失败则返回None
        """
        # 设置日期范围 - 默认使用最近2周数据（不包括今天）
        end_date_dt = datetime.now() - timedelta(days=1)  # 默认使用昨天
        start_date_dt = end_date_dt - timedelta(weeks=40)  # 默认使用两周前
        
        # 如果提供了日期参数，则使用提供的日期
        if end_date:
            try:
                end_date_dt = datetime.strptime(end_date, '%Y%m%d')
            except ValueError:
                logger.warning(f"结束日期格式错误: {end_date}，使用默认值")
                
        if start_date:
            try:
                start_date_dt = datetime.strptime(start_date, '%Y%m%d')
            except ValueError:
                logger.warning(f"开始日期格式错误: {start_date}，使用默认值")
        
        # 确保开始日期不早于结束日期
        if start_date_dt >= end_date_dt:
            start_date_dt = end_date_dt - timedelta(weeks=40)
            
        # 格式化为字符串
        start_str = start_date_dt.strftime('%Y%m%d')
        end_str = end_date_dt.strftime('%Y%m%d')
        
        # 确保stock_code是字符串并去除空格
        stock_code = str(stock_code).strip()
        
        try:
            # 提取纯数字部分
            code_digits = ''.join(filter(str.isdigit, stock_code))
            
            # 确保股票代码格式正确（6位数字）
            if len(code_digits) < 1:
                logger.error(f"股票代码 {stock_code} 格式不正确，无法提取有效数字")
                return None
                
            # 如果不足6位，前面补0
            code_digits = code_digits.zfill(6)
            
            # 如果超过6位，只取后6位
            if len(code_digits) > 6:
                code_digits = code_digits[-6:]
                
            logger.info(f"分析股票: 代码={code_digits}, 日期范围={start_str} 至 {end_str}")
            
            # 尝试获取数据
            max_retries = 3  # 增加重试次数
            retry_delay = 10  # 增加重试延迟
            
            for attempt in range(max_retries):
                try:
                    # 初始化WisdomTradingSystem，使用纯数字股票代码
                    logger.info(f"尝试获取数据 (尝试 {attempt + 1}/{max_retries})...")
                    df = self.wisdom_system.load_data(code_digits, start_str, end_str)
                    
                    if df is None or df.empty:
                        logger.warning(f"{code_digits}: 获取的数据为空")
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 1)  # 指数退避
                            logger.info(f"等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                            continue
                        logger.error(f"{code_digits}: 达到最大重试次数，无法获取数据")
                        return None
                    
                    # 检查数据是否包含足够的记录
                    min_records_required = 20  # 至少需要20条记录才能进行有效分析
                    if len(df) < min_records_required:
                        logger.warning(f"{code_digits}: 数据记录不足，只有 {len(df)} 条记录，至少需要 {min_records_required} 条")
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 1)
                            logger.info(f"等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                            continue
                        return None
                        
                    logger.info(f"成功获取 {code_digits} 的数据，共 {len(df)} 条记录")
                    break  # 成功获取数据，退出重试循环
                    
                except Exception as e:
                    logger.error(f"获取 {code_digits} 数据时出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt >= max_retries - 1:
                        logger.error(f"{code_digits}: 达到最大重试次数，放弃获取数据")
                        return None
                    time.sleep(5)  # 添加延迟后重试
            
            # 如果数据获取失败，返回None
            if not hasattr(self, 'wisdom_system') or self.wisdom_system.data is None or self.wisdom_system.data.empty:
                return None
                
            # 生成WisdomTradingSystem信号
            wisdom_signals = self.wisdom_system.generate_signals()
            
            # 验证wisdom_signals
            if wisdom_signals is None or wisdom_signals.empty:
                raise ValueError("WisdomTradingSystem 未生成有效信号")
                
            # 准备SmartMoneyStrategy数据
            df = self.wisdom_system.data.copy()
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 生成SmartMoneyStrategy信号
            smart_money_signals = self.smart_money.generate_signals(df)
            
            # 验证smart_money_signals
            if smart_money_signals is None or smart_money_signals.empty:
                raise ValueError("SmartMoneyStrategy 未生成有效信号")
            
            # 检查信号DataFrame的列
            logger.info(f"Wisdom Signals columns: {wisdom_signals.columns.tolist()}")
            logger.info(f"Smart Money Signals columns: {smart_money_signals.columns.tolist()}")
            
            # 获取最新信号
            latest_wisdom = wisdom_signals.iloc[-1]
            latest_smart = smart_money_signals.iloc[-1]
            
            # 记录信号详情
            logger.info(f"Latest Wisdom Signals (last row):\n{latest_wisdom.to_dict()}")
            logger.info(f"Latest Smart Money Signals (last row):\n{latest_smart.to_dict()}")
            
            # 确定信号列名 - 根据实际列名调整
            buy_col = 'signal' if 'signal' in wisdom_signals.columns else 'Buy'
            sell_col = 'signal' if 'signal' in wisdom_signals.columns else 'Sell'
            
            # 计算volume_ratio
            volume_ratio = None
            if not df.empty and len(df) >= 20:
                vol_mean = df['volume'].rolling(20).mean().iloc[-1]
                if vol_mean != 0:
                    volume_ratio = df['volume'].iloc[-1] / vol_mean
            
            # 合并信号
            signal = {
                'stock_code': stock_code,
                'date': df.index[-1].strftime('%Y-%m-%d') if not df.empty else None,
                'price': df['close'].iloc[-1] if not df.empty else None,
                'wisdom_buy': float(latest_wisdom.get(buy_col, 0)),
                'wisdom_sell': float(latest_wisdom.get(sell_col, 0)),
                'smart_money_signal': float(latest_smart.get('signal', 0)),
                'ma5': df['close'].rolling(5).mean().iloc[-1] if not df.empty and len(df) >= 5 else None,
                'ma10': df['close'].rolling(10).mean().iloc[-1] if not df.empty and len(df) >= 10 else None,
                'ma20': df['close'].rolling(20).mean().iloc[-1] if not df.empty and len(df) >= 20 else None,
                'volume_ratio': volume_ratio
            }
            
            # 计算综合信号
            signal['combined_signal'] = self._calculate_combined_signal(signal)
            
            return signal
            
        except Exception as e:
            import traceback
            error_msg = f"分析 {stock_code} 时出错: {str(e)}\n"
            error_msg += "\n=== 异常堆栈跟踪 ===\n"
            error_msg += traceback.format_exc()
            logger.error(error_msg)
            return None
    
    def _calculate_combined_signal(self, signal: Dict) -> str:
        """
        计算综合信号
        
        参数:
            signal: 包含各策略信号的字典
            
        返回:
            str: 综合信号 ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')
        """
        try:
            # 记录信号值以便调试
            logger.info(f"Calculating combined signal with values: {signal}")
            
            # 获取信号值，确保是数值类型
            wisdom_buy = float(signal.get('wisdom_buy', 0))
            wisdom_sell = float(signal.get('wisdom_sell', 0))
            smart_money = float(signal.get('smart_money_signal', 0))
            
            logger.info(f"wisdom_buy: {wisdom_buy}, wisdom_sell: {wisdom_sell}, smart_money: {smart_money}")
            
            # 两个策略都看多
            if wisdom_buy > 0 and smart_money > 0:
                return 'strong_buy'
                
            # 一个策略看多，一个策略中性
            elif wisdom_buy > 0 or smart_money > 0:
                return 'buy'
                
            # 两个策略都看空
            elif wisdom_sell > 0 and smart_money < 0:
                return 'strong_sell'
                
            # 一个策略看空，一个策略中性
            elif wisdom_sell > 0 or smart_money < 0:
                return 'sell'
                
            # 其他情况保持中性
            return 'hold'
            
        except Exception as e:
            logger.error(f"计算综合信号时出错: {str(e)}")
            logger.exception("详细错误信息:")
            return 'hold'  # 出错时默认返回中性信号
    
    def analyze_index(self, index_type: str = 'csi300', 
                     start_date: str = None, 
                     end_date: str = None,
                     max_stocks: int = None,
                     stock_codes: List[str] = None) -> pd.DataFrame:
        """
        分析指数成分股或指定股票列表
        
        参数:
            index_type: 指数类型，'csi300' 或 'csi500'，当stock_codes为None时使用
            start_date: 开始日期，格式'YYYYMMDD'，如果为None则使用两周前
            end_date: 结束日期，格式'YYYYMMDD'，如果为None则使用昨天
            max_stocks: 最大分析股票数量，当stock_codes为None时有效
            stock_codes: 要分析的股票代码列表，如果提供则忽略index_type和max_stocks
            
        返回:
            pd.DataFrame: 包含所有股票分析结果的DataFrame
        """
        # 设置日期范围 - 默认使用最近2周数据（不包括今天）
        end_date_dt = datetime.now() - timedelta(days=1)  # 默认使用昨天
        start_date_dt = end_date_dt - timedelta(weeks=30)  # 默认使用两周前
            
        # 如果提供了日期参数，则使用提供的日期
        if end_date:
            try:
                end_date_dt = datetime.strptime(end_date, '%Y%m%d')
            except ValueError:
                logger.warning(f"结束日期格式错误: {end_date}，使用默认值")
                    
        if start_date:
            try:
                start_date_dt = datetime.strptime(start_date, '%Y%m%d')
            except ValueError:
                logger.warning(f"开始日期格式错误: {start_date}，使用默认值")
            
        # 确保开始日期不早于结束日期
        if start_date_dt >= end_date_dt:
            start_date_dt = end_date_dt - timedelta(weeks=30)
                
        # 格式化为字符串
        start_str = start_date_dt.strftime('%Y%m%d')
        end_str = end_date_dt.strftime('%Y%m%d')
                
        logger.info(f"开始分析{index_type.upper()}成分股，时间范围: {start_str} 至 {end_str}")
        
        # 处理股票代码列表
        if stock_codes is not None and len(stock_codes) > 0:
            # 使用提供的股票代码列表
            constituents = pd.DataFrame({
                'stock_code': stock_codes,
                'stock_name': [self._get_stock_name(code) for code in stock_codes]  # 使用akshare获取股票名称
            })
            logger.info(f"使用提供的 {len(constituents)} 只股票代码进行分析")
        else:
            # 加载指数成分股
            try:
                constituents = self.load_constituents(index_type)
                
                # 检查是否成功加载成分股
                if constituents is None or len(constituents) == 0:
                    logger.warning(f"未找到{index_type.upper()}成分股数据")
                    return pd.DataFrame(columns=['stock_code', 'stock_name', 'combined_signal'])
                    
                logger.info(f"共加载 {len(constituents)} 只{index_type.upper()}成分股")
                
                # 如果指定了最大股票数量，则随机选择
                if max_stocks and len(constituents) > max_stocks:
                    try:
                        random_indices = random.sample(range(len(constituents)), max_stocks)
                        constituents = constituents.iloc[random_indices].copy()
                        logger.info(f"随机选择 {len(constituents)} 只股票进行分析")
                    except Exception as e:
                        logger.error(f"随机选择股票时出错: {str(e)}")
                        return pd.DataFrame(columns=['stock_code', 'stock_name', 'combined_signal'])
                
            except Exception as e:
                logger.error(f"加载{index_type.upper()}成分股失败: {str(e)}")
                logger.exception("详细错误信息:")
                return pd.DataFrame(columns=['stock_code', 'stock_name', 'combined_signal'])
        
        # 确保有股票需要分析
        if len(constituents) == 0:
            logger.warning("没有可分析的股票")
            return pd.DataFrame(columns=['stock_code', 'stock_name', 'combined_signal'])
            
        # 分析每只股票
        results = []
        total_stocks = min(len(constituents), max_stocks) if max_stocks else len(constituents)
        success_count = 0
        fail_count = 0
            
        logger.info(f"\n{'='*80}")
        logger.info(f"开始分析 {total_stocks} 只{index_type.upper()}成分股...")
        logger.info(f"日期范围: {start_str} 至 {end_str}")
        logger.info("="*80)
            
        with tqdm(total=total_stocks, desc=f"分析{index_type.upper()}成分股") as pbar:
            for idx, row in constituents.iterrows():
                if max_stocks and len(results) >= max_stocks:
                    break
                        
                stock_code = str(row['stock_code']).strip()
                stock_name = str(row.get('stock_name', '未知')).strip()
                    
                # 更新进度条描述
                pbar.set_description(f"分析 {stock_name} ({stock_code})")
                time.sleep(10)
                try:
                    # 分析股票
                    result = self.analyze_stock(stock_code, start_str, end_str)
                        
                    if result is not None:
                        result['stock_name'] = stock_name
                        results.append(result)
                        success_count += 1
                        logger.info(f"分析成功: {stock_name} ({stock_code}) - {result.get('combined_signal', 'N/A')}")
                    else:
                        fail_count += 1
                        logger.warning(f"分析跳过: {stock_name} ({stock_code}) - 返回空结果")
                            
                except Exception as e:
                    fail_count += 1
                    logger.error(f"分析失败: {stock_name} ({stock_code}) - {str(e)}")
                    logger.debug("详细错误信息: %s", str(e), exc_info=True)
                        
                finally:
                    pbar.update(1)
                    # 添加延迟，避免请求过于频繁
                    time.sleep(1)
            
        # 输出分析摘要
        logger.info("\n" + "="*80)
        logger.info(f"分析完成 - 成功: {success_count}, 失败: {fail_count}, 总计: {total_stocks}")
        logger.info("="*80)
            
        # 如果没有分析结果，返回空DataFrame
        if not results:
            logger.warning(f"没有成功分析任何{index_type.upper()}成分股")
            return pd.DataFrame()
                
        try:
            # 转换为DataFrame
            df = pd.DataFrame(results)
                
            # 按综合信号排序
            signal_order = {'strong_buy': 0, 'buy': 1, 'hold': 2, 'sell': 3, 'strong_sell': 4}
            df['signal_order'] = df['combined_signal'].map(signal_order)
            df = df.sort_values('signal_order').drop('signal_order', axis=1)
                
            # 保存结果
            os.makedirs(self.data_dir, exist_ok=True)
            output_file = os.path.join(self.data_dir, f"{index_type}_analysis_{datetime.now().strftime('%Y%m%d')}.csv")
            # 检查文件是否存在，如果存在则追加，否则创建新文件
            header = not os.path.exists(output_file)
            df.to_csv(output_file, mode='a', index=False, encoding='utf-8-sig', header=header)
            logger.info(f"分析结果已{'追加到' if not header else '保存至'}: {output_file}")
            
            return df
        except Exception as e:
            logger.error(f"处理分析结果时出错: {str(e)}")
            logger.exception("详细错误信息:")
            return pd.DataFrame()

    def _calculate_combined_signal(self, signal: Dict) -> str:
        """
        计算综合信号
        
        参数:
            signal: 包含各策略信号的字典
                
        返回:
            str: 综合信号 ('buy', 'sell', 'hold')
        """
        logger.info(signal)
        # 两个策略都看多
        if (signal['wisdom_buy'] > 0 and signal['smart_money_signal'] > 0):
            return 'strong_buy'
                
        # 一个策略看多，一个策略中性
        elif (signal['wisdom_buy'] > 0 or signal['smart_money_signal'] > 0):
            return 'buy'
                
        # 两个策略都看空
        elif (signal['wisdom_sell'] < 0 and signal['smart_money_signal'] < 0):
            return 'strong_sell'
                
        # 一个策略看空，一个策略中性
        elif (signal['wisdom_sell'] > 0 or signal['smart_money_signal'] < 0):
            return 'sell'
                
        # 其他情况保持中性
        return 'hold'

def main():
    """主函数"""
    start_date_dt = (datetime.now() - timedelta(weeks=30)).strftime('%Y%m%d')
    end_date_dt = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    # start_date_dt = '20240110'
    # end_date_dt = '20250310'
    num_stocks = 100
    stock_list = ['000513', '600023', '600098', '600380', '600642', '600982', '600521', '600993', '600129', '600062', '600436', '002107', '000623', '002603', '002412', '002317', '000538', '000989', '000999', '002020', '002287', '002424', '002644', '002737', '002773', '002864', '002880', '300181', '300534', '301207', '600085', 
    '600161', '600196', '600211', '600285', '600329', '600332', '600422', '600479', '600535', '600557', '600566', '600572', '600750', '600867', 
    '600976', '603087', '603439', '603858', '603896', '603998', '688247', '688658', '000963', '002393']
    try:
        # 创建分析器
        analyzer = CombinedStrategyAnalyzer(data_dir='data')
        
        # 分析CSI 300成分股
        try:
            logger.info("开始分析CSI 300成分股...")
            csi300_results = analyzer.analyze_index('csi300', max_stocks=num_stocks, start_date=start_date_dt, end_date=end_date_dt, stock_codes=stock_list)  # 限制股票数量用于测试
            if csi300_results is not None and not csi300_results.empty:
                logger.info("\nCSI 300分析结果:")
                logger.info(csi300_results[['stock_code', 'stock_name', 'combined_signal']].to_string())
            else:
                logger.warning("未获取到有效的CSI 300分析结果")
        except Exception as e:
            logger.error(f"分析CSI 300成分股时出错: {str(e)}")
            logger.exception("详细错误信息:")
        
        if stock_list is not None and len(stock_list) > 0:
            return
            
        # 分析CSI 500成分股
        try:
            logger.info("\n开始分析CSI 500成分股...")
            csi500_results = analyzer.analyze_index('csi500', max_stocks=num_stocks, start_date=start_date_dt, end_date=end_date_dt, stock_codes=stock_list)  # 限制股票数量用于测试
            if csi500_results is not None and not csi500_results.empty:
                logger.info("\nCSI 500分析结果:")
                logger.info(csi500_results[['stock_code', 'stock_name', 'combined_signal']].to_string())
            else:
                logger.warning("未获取到有效的CSI 500分析结果")
        except Exception as e:
            logger.error(f"分析CSI 500成分股时出错: {str(e)}")
            logger.exception("详细错误信息:")
            
    except Exception as e:
        logger.critical(f"程序发生严重错误: {str(e)}")
        logger.exception("错误详情:")

if __name__ == "__main__":
    main()
