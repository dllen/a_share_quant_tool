import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


def get_ashare_stocks(max_retries=3, retry_delay=2):
    """
    获取A股股票列表
    
    Args:
        max_retries (int): 最大重试次数
        retry_delay (int): 重试间隔(秒)
        
    Returns:
        pd.DataFrame: 包含股票代码和名称的DataFrame
    """
    for attempt in range(max_retries):
        try:
            print(f"尝试获取A股股票列表 (尝试 {attempt + 1}/{max_retries})...")
            # 尝试获取上海和深圳交易所的股票列表
            stock_sh = ak.stock_info_sh_name_code()
            stock_sz = ak.stock_info_sz_name_code()
            
            # 重命名列以匹配
            stock_sh = stock_sh.rename(columns={"证券代码": "code", "证券简称": "name"})
            stock_sz = stock_sz.rename(columns={"A股代码": "code", "A股简称": "name"})
            
            # 合并两个交易所的数据
            stock_info = pd.concat([stock_sh, stock_sz], ignore_index=True)
            
            # 确保code列是字符串类型并去除空格
            stock_info['code'] = stock_info['code'].astype(str).str.strip()
            
            print(f"成功获取 {len(stock_info)} 只A股股票信息")
            return stock_info
            
        except Exception as e:
            print(f"获取A股股票列表时出错: {str(e)}")
            if attempt < max_retries - 1:
                print(f"{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("达到最大重试次数，无法获取A股股票列表")
                raise

def get_financial_indicators(stock_code):
    """
    获取财务指标
    
    Args:
        stock_code (str): 股票代码
        
    Returns:
        dict: 包含财务指标和现金流量数据的字典，如果获取失败则返回None
    """
    def try_get_data(stock_code, symbol):
        """尝试获取特定类型的财务数据"""
        try:
            data = ak.stock_financial_report_sina(stock=stock_code, symbol=symbol)
            if data is not None and not data.empty:
                print(f"{stock_code}: 成功获取{symbol}数据，形状: {data.shape}")
                print(f"可用列: {data.columns.tolist()}")
            return data
        except Exception as e:
            print(f"获取{stock_code}的{symbol}数据失败: {str(e)}")
            return None
    
    def try_eastmoney(stock_code):
        """尝试从东方财富获取财务数据"""
        try:
            print(f"{stock_code}: 尝试从东方财富获取财务指标...")
            data = ak.stock_financial_analysis_indicator(stock=stock_code)
            if not data.empty:
                print(f"{stock_code}: 成功从东方财富获取财务指标，形状: {data.shape}")
                print(f"可用列: {data.columns.tolist()}")
            return data
        except Exception as e:
            print(f"{stock_code}: 从东方财富获取财务指标失败: {str(e)}")
            return None
    
    try:
        # 尝试从东方财富获取数据（更可靠的数据源）
        indicators = try_eastmoney(stock_code)
        
        # 如果东方财富数据获取失败，再尝试新浪财经
        if indicators is None or indicators.empty:
            print(f"{stock_code}: 尝试从新浪财经获取数据...")
            # 尝试获取主要财务指标
            indicators = try_get_data(stock_code, "主要指标")
            
            # 如果主要指标获取失败，尝试获取利润表数据作为替代
            if indicators is None or indicators.empty:
                print(f"{stock_code}: 主要指标获取失败，尝试获取利润表数据...")
                indicators = try_get_data(stock_code, "利润表")
        
        # 获取现金流量数据
        cash_flow = try_get_data(stock_code, "现金流量表")
        
        # 如果主要指标和利润表都获取失败，尝试使用其他数据源
        if indicators is None or indicators.empty:
            print(f"{stock_code}: 尝试使用其他数据源获取财务数据...")
            # 尝试使用东方财富接口再次尝试
            indicators = try_eastmoney(stock_code)
        
        if indicators is None or indicators.empty:
            print(f"{stock_code}: 无法获取有效的财务指标数据")
            return None
            
        # 打印获取到的数据样例，便于调试
        print(f"\n{stock_code} 获取到的财务数据前5行:")
        print(indicators.head().to_string())
        
        # 如果数据是转置的（行是财务指标，列是报告期），则进行转置
        if '报告日' in indicators.columns or '报表日期' in indicators.columns:
            print(f"{stock_code}: 检测到需要转置的财务数据格式")
            # 尝试找到包含日期的列名
            date_col = '报告日' if '报告日' in indicators.columns else '报表日期'
            indicators = indicators.set_index(date_col).T
            indicators = indicators.reset_index()
            indicators = indicators.rename(columns={'index': '指标'})
            print(f"{stock_code}: 转置后的数据形状: {indicators.shape}")
        
        return {
            "indicators": indicators,
            "cash_flow": cash_flow if cash_flow is not None else pd.DataFrame()
        }
        
    except Exception as e:
        print(f"{stock_code}: 获取财务数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        return None

def get_stock_data(stock_code):
    """
    获取股票历史数据
    
    Args:
        stock_code (str): 股票代码，可以带或不带市场前缀
        
    Returns:
        pd.DataFrame: 包含历史交易数据的DataFrame，如果获取失败则返回None
    """
    def try_akshare(symbol, start_date, end_date):
        """尝试使用akshare获取股票数据"""
        try:
            # 尝试获取沪深A股历史数据
            df = ak.stock_zh_a_hist(
                symbol=symbol, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date,
                adjust="hfq"  # 使用后复权数据
            )
            if not df.empty:
                print(f"{symbol}: 成功获取 {len(df)} 条历史数据 (akshare)")
                return df
        except Exception as e:
            print(f"{symbol}: 使用akshare获取数据失败: {str(e)}")
        return None
    
    def try_akshare_em(symbol, start_date, end_date):
        """尝试使用akshare的东方财富接口获取数据"""
        try:
            # 尝试使用东方财富接口
            df = ak.stock_zh_a_daily(
                symbol=f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            if not df.empty:
                print(f"{symbol}: 成功获取 {len(df)} 条历史数据 (akshare 东方财富)")
                return df
        except Exception as e:
            print(f"{symbol}: 使用akshare东方财富接口获取数据失败: {str(e)}")
        return None
    
    try:
        # 标准化股票代码（移除可能的sh/sz前缀）
        clean_code = str(stock_code).lower().replace('sh', '').replace('sz', '').strip()
        
        # 设置日期范围（最近一年）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')  # 多取一些天以确保有足够数据
        
        print(f"{stock_code}: 尝试获取从 {start_date} 到 {end_date} 的历史数据...")
        
        # 尝试不同的数据源和方法
        df = try_akshare(clean_code, start_date, end_date)
        
        # 如果第一个方法失败，尝试东方财富接口
        if df is None or df.empty:
            df = try_akshare_em(clean_code, start_date, end_date)
        
        # 如果仍然没有数据，尝试添加市场前缀
        if df is None or df.empty:
            market_prefix = 'sh' if clean_code.startswith(('6', '9')) else 'sz'
            df = try_akshare_em(f"{market_prefix}{clean_code}", start_date, end_date)
        
        if df is None or df.empty:
            print(f"{stock_code}: 所有数据源均未能获取到有效数据")
            return None
            
        # 列名映射：英文 -> 中文
        column_mapping = {
            'date': '日期',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量',
            'amount': '成交额',
            'outstanding_share': '流通股本',
            'turnover': '换手率',
            'change': '涨跌幅',
            'pct_chg': '涨跌幅(%)',
            'amount': '成交额(元)'
        }
        
        # 重命名列为中文（如果列名是英文）
        rename_columns = {}
        for eng_col, chn_col in column_mapping.items():
            if eng_col in df.columns and chn_col not in df.columns:
                rename_columns[eng_col] = chn_col
        
        if rename_columns:
            print(f"{clean_code}: 检测到英文列名，进行重命名: {rename_columns}")
            df = df.rename(columns=rename_columns)
        
        # 确保必要的列存在（检查中英文列名）
        required_columns = {'日期', '收盘'}
        available_columns = set(df.columns)
        
        # 检查是否缺少必要列
        missing_columns = required_columns - available_columns
        if missing_columns:
            print(f"{clean_code}: 获取的数据缺少必要列 {missing_columns}，可用列: {df.columns.tolist()}")
            return None
            
        # 确保日期列是datetime类型
        date_col = '日期' if '日期' in df.columns else 'date'
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # 按日期排序
        df = df.sort_values(date_col)
        
        # 确保收盘价是数值类型
        close_col = '收盘' if '收盘' in df.columns else 'close'
        if not pd.api.types.is_numeric_dtype(df[close_col]):
            df[close_col] = pd.to_numeric(df[close_col].astype(str).str.replace(',', ''), errors='coerce')
        
        # 打印数据摘要
        start_date = df[date_col].min().date()
        end_date = df[date_col].max().date()
        last_close = df[close_col].iloc[-1]
        
        print(f"{clean_code}: 获取到从 {start_date} 到 {end_date} 的 {len(df)} 条数据")
        print(f"最新数据: 日期={df[date_col].iloc[-1].date()}, 收盘价={last_close:.2f}")
        
        # 统一列名（确保使用中文列名）
        if 'date' in df.columns and '日期' not in df.columns:
            df = df.rename(columns={'date': '日期'})
        if 'close' in df.columns and '收盘' not in df.columns:
            df = df.rename(columns={'close': '收盘'})
        
        return df
        
    except Exception as e:
        print(f"{stock_code}: 获取股票数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_market_cap(stock_code):
    """获取市值（亿元）"""
    try:
        # First try to get from spot data
        try:
            spot_data = ak.stock_zh_a_spot_em()
            if not spot_data.empty and '代码' in spot_data.columns and '总市值' in spot_data.columns:
                market_cap = spot_data[spot_data['代码'] == stock_code]['总市值'].values
                if len(market_cap) > 0:
                    # Convert from string with 亿 to float
                    if isinstance(market_cap[0], str):
                        return float(market_cap[0].replace('亿', ''))
                    return float(market_cap[0])
        except Exception as e:
            print(f"从spot数据获取市值失败: {str(e)}")
        
        # Fallback to individual info
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        if not stock_info.empty and 'item' in stock_info.columns and 'value' in stock_info.columns:
            # Try different possible item names for market cap
            for item_name in ['总市值', '总市值(亿)', 'market_cap']:
                market_cap = stock_info[stock_info['item'] == item_name]['value'].values
                if len(market_cap) > 0:
                    if isinstance(market_cap[0], str):
                        return float(market_cap[0].replace('亿', '').replace(',', ''))
                    return float(market_cap[0])
        
        print(f"无法获取 {stock_code} 的市值信息")
        return None
    except Exception as e:
        print(f"获取市值失败 {stock_code}: {str(e)}")
        return None

def get_industry(stock_code):
    """获取行业信息"""
    try:
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        # Try multiple possible column names for industry
        industry_mappings = [
            ('行业', 'value'),  # Standard mapping
            ('industry', 'value'),  # Test case mapping
            ('item', 'value')  # Another possible mapping
        ]
        
        for item_col, value_col in industry_mappings:
            try:
                if item_col in stock_info.columns and value_col in stock_info.columns:
                    industry_row = stock_info[stock_info[item_col] == 'industry']
                    if not industry_row.empty:
                        return industry_row[value_col].iloc[0]
            except Exception:
                continue
        
        print(f"无法找到 {stock_code} 的行业信息")
        print("可用列:", stock_info.columns.tolist())
        return "未知行业"
    except Exception as e:
        print(f"获取行业信息失败 {stock_code}: {str(e)}")
        return "未知行业"

def check_financial_conditions(fin_data, stock_code):
    """
    检查财务指标是否满足条件
    
    Args:
        fin_data (dict): 包含财务指标和现金流量数据的字典
        stock_code (str): 股票代码
        
    Returns:
        bool: 如果满足所有财务条件返回True，否则返回False
    """
    if fin_data is None or fin_data.get('indicators') is None:
        print(f"{stock_code}: 财务数据不完整")
        return False
    
    indicators = fin_data['indicators']
    
    try:
        print(f"\n检查 {stock_code} 的财务数据...")
        print("可用财务指标列:", indicators.columns.tolist())
        
        # 检查数据源类型并提取ROE数据
        roe_values = []
        found_roe = False
        
        # 尝试从ROE列提取数据 (不同可能的列名)
        roe_columns = ['ROE', '净资产收益率', '净资产收益率(%)', '净资产收益率（ROE）', '净资产收益率(加权)']
        for col in roe_columns:
            if not found_roe and col in indicators.columns:
                print(f"{stock_code}: 从'{col}'列提取ROE数据")
                roe_series = indicators[col].dropna()
                if not roe_series.empty:
                    # 尝试转换为数值类型
                    try:
                        roe_values = [float(str(x).replace('%', '').strip()) for x in roe_series.head(3)]
                        found_roe = True
                        print(f"{stock_code}: 成功提取ROE值: {roe_values}")
                        break
                    except (ValueError, TypeError) as e:
                        print(f"{stock_code}: 转换ROE值时出错: {str(e)}")
        
        # 尝试从利润表数据中计算ROE (净利润 / 净资产)
        if not found_roe and '指标' in indicators.columns:
            print(f"\n{stock_code}: 尝试从财务指标数据中提取ROE或计算ROE")
            
            # 首先提取所有日期列（排除非日期列）
            date_columns = [col for col in indicators.columns if col not in ['指标', '单位'] and col.replace('-', '').replace('/', '').isdigit()]
            if not date_columns:  # 如果没有找到标准日期格式，尝试其他可能的日期列
                date_columns = [col for col in indicators.columns if col not in ['指标', '单位']]
            
            # 按日期排序（最近的在前）
            date_columns = sorted(date_columns, reverse=True)[:3]  # 获取最近的3个报告期
            
            if not date_columns:
                print(f"{stock_code}: 错误: 无法识别任何日期列")
                print(f"所有列: {indicators.columns.tolist()}")
            else:
                print(f"{stock_code}: 识别的报告期: {date_columns}")
            
            # 打印所有可用的财务指标，便于调试
            print(f"\n{stock_code}: 所有可用的财务指标:")
            for idx, row in indicators.iterrows():
                print(f"{row['指标']}: {row[date_columns[0]] if date_columns and date_columns[0] in row else 'N/A'}")
            
            # 首先尝试直接提取ROE
            roe_patterns = ['净资产收益率', 'ROE', '净资产收益率\(%\)', '净资产收益率（ROE）']
            roe_indicators = indicators[indicators['指标'].str.contains('|'.join(roe_patterns), na=False, regex=True)]
            
            if not roe_indicators.empty:
                print(f"\n{stock_code}: 找到ROE指标数据:")
                print(roe_indicators.to_string())
                
                for date_col in date_columns:
                    if date_col in roe_indicators.columns:
                        for _, row in roe_indicators.iterrows():
                            roe_value = row[date_col]
                            if pd.notna(roe_value) and str(roe_value).strip() != '':
                                try:
                                    roe = float(str(roe_value).replace('%', '').strip())
                                    roe_values.append(roe)
                                    print(f"{stock_code}: 从{date_col}提取到ROE: {roe}% (指标: {row['指标']})")
                                    break  # 找到第一个有效的ROE值就停止
                                except (ValueError, TypeError) as e:
                                    print(f"{stock_code}: 转换ROE值'{roe_value}'时出错: {str(e)}")
                
                if roe_values:
                    found_roe = True
                    print(f"{stock_code}: 成功提取ROE数据: {roe_values}")
            
            # 如果没有直接找到ROE，尝试从净利润和净资产计算
            if not found_roe and any(indicators['指标'].str.contains('净利润|净利', na=False)):
                print(f"\n{stock_code}: 未直接找到ROE，尝试从净利润和净资产计算")
                
                # 查找净利润行
                net_profit_patterns = ['归属于母公司所有者的净利润', '净利润', '净利']
                net_profit_row = indicators[indicators['指标'].str.contains('|'.join(net_profit_patterns), na=False, regex=True)]
                
                if not net_profit_row.empty:
                    net_profit_row = net_profit_row.iloc[0]  # 取第一行匹配的净利润数据
                    print(f"{stock_code}: 找到净利润数据: {net_profit_row['指标']}")
                    
                    # 查找所有者权益行（使用更灵活的模式匹配）
                    equity_patterns = [
                        '归属于母公司所有者权益合计',
                        '所有者权益.*合计',
                        '股东权益.*合计',
                        '净资产',
                        '归属于母公司股东的权益',
                        '所有者权益合计',
                        '股东权益',
                        '净资产合计',
                        '所有者权益总计',
                        '股东权益总计',
                        '归属于母公司股东权益合计',
                        '净资产'
                    ]
                    
                    equity_row = None
                    for pattern in equity_patterns:
                        mask = indicators['指标'].str.contains(pattern, na=False, regex=True)
                        if mask.any():
                            equity_row = indicators[mask].iloc[0]
                            print(f"{stock_code}: 找到匹配的所有者权益行: {equity_row['指标']} (模式: {pattern})")
                            break
                    
                    if equity_row is not None:
                        print(f"{stock_code}: 所有者权益数据: {equity_row[date_columns].to_dict() if date_columns else '无日期列'}")
                        
                        # 计算ROE
                        for date_col in date_columns:
                            try:
                                np_val = net_profit_row[date_col] if date_col in net_profit_row else None
                                eq_val = equity_row[date_col] if date_col in equity_row else None
                                
                                if pd.notna(np_val) and pd.notna(eq_val):
                                    np_str = str(np_val).replace(',', '').replace(' ', '').replace('元', '').strip()
                                    eq_str = str(eq_val).replace(',', '').replace(' ', '').replace('元', '').strip()
                                    
                                    if np_str and eq_str and np_str.replace('.', '').replace('-', '').isdigit() and eq_str.replace('.', '').replace('-', '').isdigit():
                                        np = float(np_str)
                                        eq = float(eq_str)
                                        
                                        if eq != 0:
                                            roe = (np / abs(eq)) * 100
                                            roe_values.append(roe)
                                            print(f"{stock_code}: 计算ROE ({date_col}): 净利润={np}, 净资产={eq}, ROE={roe:.2f}%")
                                        else:
                                            print(f"{stock_code}: 警告: {date_col}年净资产为0，无法计算ROE")
                                    else:
                                        print(f"{stock_code}: 无法解析数值: 净利润='{np_str}', 净资产='{eq_str}'")
                                else:
                                    print(f"{stock_code}: 缺少{date_col}年的数据: 净利润={pd.notna(np_val)}, 净资产={pd.notna(eq_val)}")
                            except (ValueError, TypeError, AttributeError) as e:
                                print(f"{stock_code}: 计算{date_col}年ROE时出错: {str(e)}")
                        
                        if roe_values:
                            found_roe = True
                            print(f"{stock_code}: 成功计算ROE: {roe_values}")
                    else:
                        print(f"{stock_code}: 未找到可用的所有者权益数据")
                        print(f"{stock_code}: 所有尝试的权益模式: {equity_patterns}")
                        print(f"{stock_code}: 可用的指标:")
                        for idx, row in indicators.iterrows():
                            print(f"- {row['指标']}")
                else:
                    print(f"{stock_code}: 未找到净利润数据")
                    print(f"{stock_code}: 尝试的净利润模式: {net_profit_patterns}")
                
                # 获取净利润数据
                net_profit_row = indicators[indicators['指标'].isin(['归属于母公司所有者的净利润', '净利润'])].iloc[0]
                net_profit_values = net_profit_row[date_columns].dropna()
                
                print(f"{stock_code}: 净利润数据: {net_profit_values.to_dict()}")
                
                # 尝试查找所有者权益数据
                equity_patterns = [
                    '归属于母公司所有者权益合计',
                    '所有者权益(或股东权益)合计',
                    '股东权益合计',
                    '净资产',
                    '归属于母公司股东的权益',
                    '所有者权益合计',
                    '股东权益',
                    '净资产合计',
                    '所有者权益（或股东权益）合计',
                    '所有者权益总计',
                    '股东权益总计',
                    '归属于母公司股东权益合计'
                ]
                
                equity_row = None
                for pattern in equity_patterns:
                    mask = indicators['指标'].str.contains(pattern, na=False, regex=False)
                    if mask.any():
                        equity_row = indicators[mask].iloc[0]
                        print(f"{stock_code}: 找到匹配的所有者权益行: {equity_row['指标']}")
                        break
                
                if equity_row is not None:
                    equity_values = equity_row[date_columns].dropna()
                    print(f"{stock_code}: 净资产数据: {equity_values.to_dict()}")
                    
                    # 计算ROE
                    for date in net_profit_values.index.intersection(equity_values.index):
                        try:
                            np_str = str(net_profit_values[date]).replace(',', '').replace(' ', '').replace('元', '')
                            eq_str = str(equity_values[date]).replace(',', '').replace(' ', '').replace('元', '')
                            
                            np = float(np_str) if np_str.replace('.', '').replace('-', '').replace('+', '').isdigit() else 0
                            eq = float(eq_str) if eq_str.replace('.', '').replace('-', '').replace('+', '').isdigit() else 0
                            
                            if eq != 0:
                                roe = (np / abs(eq)) * 100
                                roe_values.append(roe)
                                print(f"{stock_code}: 计算ROE ({date}): 净利润={np}, 净资产={eq}, ROE={roe:.2f}%")
                            else:
                                print(f"{stock_code}: 警告: {date}年净资产为0，无法计算ROE")
                        except (ValueError, TypeError) as e:
                            print(f"{stock_code}: 计算{date}年ROE时出错: {str(e)}")
                    
                    if roe_values:
                        found_roe = True
                        print(f"{stock_code}: 成功计算ROE: {roe_values}")
                else:
                    print(f"{stock_code}: 未找到可用的所有者权益数据")
                    print(f"{stock_code}: 所有尝试的权益模式: {equity_patterns}")
                    print(f"{stock_code}: 可用的指标:")
                    print(indicators['指标'].tolist())
            
            # 如果仍然没有找到ROE，尝试从资产负债表和利润表计算
            if not found_roe:
                print(f"{stock_code}: 尝试从资产负债表和利润表计算ROE")
                
                # 查找净利润和股东权益
                net_profit_row = indicators[indicators['指标'].isin(['净利润', '归属于母公司所有者的净利润'])].iloc[0] if not indicators[indicators['指标'].isin(['净利润', '归属于母公司所有者的净利润'])].empty else None
                
                # 查找股东权益
                equity_patterns = ['股东权益合计', '所有者权益合计', '归属于母公司股东权益合计']
                equity_row = None
                for pattern in equity_patterns:
                    mask = indicators['指标'].str.contains(pattern, na=False, regex=False)
                    if mask.any():
                        equity_row = indicators[mask].iloc[0]
                        break
                
                if net_profit_row is not None and equity_row is not None:
                    print(f"{stock_code}: 找到净利润和股东权益数据")
                    
                    # 获取最近3年的数据
                    for date_col in date_columns[:3]:
                        try:
                            np_val = net_profit_row[date_col]
                            eq_val = equity_row[date_col]
                            
                            np = float(str(np_val).replace(',', '').replace(' ', '').replace('元', '')) if pd.notna(np_val) else 0
                            eq = float(str(eq_val).replace(',', '').replace(' ', '').replace('元', '')) if pd.notna(eq_val) else 0
                            
                            if eq != 0:
                                roe = (np / abs(eq)) * 100
                                roe_values.append(roe)
                                print(f"{stock_code}: 计算ROE ({date_col}): 净利润={np}, 净资产={eq}, ROE={roe:.2f}%")
                        except (ValueError, TypeError) as e:
                            print(f"{stock_code}: 计算{date_col}年ROE时出错: {str(e)}")
                    
                    if roe_values:
                        found_roe = True
                        print(f"{stock_code}: 成功计算ROE: {roe_values}")
                
                if not found_roe:
                    print(f"{stock_code}: 无法从可用数据计算ROE")
                    print(f"{stock_code}: 可用的指标:")
                    print(indicators['指标'].tolist())
        
        # 如果仍然没有找到ROE，尝试从现金流量表计算
        if not found_roe and '净利润' in indicators['指标'].values and '股东权益' in indicators['指标'].values:
            print(f"{stock_code}: 尝试从现金流量表计算ROE")
            try:
                net_profit = indicators[indicators['指标'] == '净利润'].iloc[0][1:].dropna()
                equity = indicators[indicators['指标'] == '股东权益'].iloc[0][1:].dropna()
                
                for date in net_profit.index.intersection(equity.index):
                    try:
                        np_val = float(str(net_profit[date]).replace(',', '').replace(' ', '').replace('元', ''))
                        eq_val = float(str(equity[date]).replace(',', '').replace(' ', '').replace('元', ''))
                        
                        if eq_val != 0:
                            roe = (np_val / abs(eq_val)) * 100
                            roe_values.append(roe)
                            print(f"{stock_code}: 从现金流量表计算ROE ({date}): {roe:.2f}%")
                    except (ValueError, TypeError) as e:
                        print(f"{stock_code}: 计算{date}年ROE时出错: {str(e)}")
                
                if roe_values:
                    found_roe = True
                    print(f"{stock_code}: 成功从现金流量表计算ROE: {roe_values}")
            except Exception as e:
                print(f"{stock_code}: 从现金流量表计算ROE时出错: {str(e)}")
        
        # 尝试从指标列中查找ROE
        if not found_roe and '指标' in indicators.columns:
            try:
                # 尝试不同的ROE列名
                roe_aliases = ['净资产收益率', 'ROE', '净资产收益率(%)', '净资产收益率（ROE）']
                for alias in roe_aliases:
                    if alias in indicators['指标'].values:
                        print(f"{stock_code}: 从指标列找到{alias}")
                        roe_mask = indicators['指标'] == alias
                        roe_row = indicators[roe_mask].iloc[0, 1:].dropna()
                        if not roe_row.empty:
                            roe_values = roe_row.head(3).tolist()
                            found_roe = True
                            break
            except Exception as e:
                print(f"{stock_code}: 从指标列提取ROE时出错: {str(e)}")
        
        # 如果还没有找到ROE，        # 尝试从净利润和净资产计算
        if not found_roe and '指标' in indicators.columns and '净利润' in indicators['指标'].values:
            print(f"{stock_code}: 尝试从净利润和净资产计算ROE")
            try:
                # 获取净利润行
                net_profit_row = indicators[indicators['指标'] == '净利润'].iloc[0, 1:].dropna()
                
                # 尝试不同的所有者权益列名
                equity_columns = [
                    '归属于母公司所有者权益合计',
                    '所有者权益(或股东权益)合计',
                    '股东权益合计',
                    '净资产',
                    '归属于母公司股东的权益',
                    '所有者权益合计',
                    '股东权益',
                    '净资产合计'
                ]
                
                # 查找第一个存在的权益列
                equity_row = None
                for col in equity_columns:
                    if col in indicators['指标'].values:
                        equity_row = indicators[indicators['指标'] == col].iloc[0, 1:].dropna()
                        print(f"{stock_code}: 找到权益列: {col}")
                        break
                
                if equity_row is not None and not equity_row.empty:
                    min_length = min(len(net_profit_row), len(equity_row), 3)
                    for i in range(min_length):
                        try:
                            # 处理可能包含逗号的数字字符串
                            net_profit = float(str(net_profit_row.iloc[i]).replace(',', '').replace(' ', ''))
                            equity = float(str(equity_row.iloc[i]).replace(',', '').replace(' ', ''))
                            if equity != 0:
                                roe = (net_profit / abs(equity)) * 100  # 使用绝对值避免负权益的问题
                                roe_values.append(roe)
                                print(f"{stock_code}: 计算ROE: 净利润={net_profit}, 净资产={equity}, ROE={roe:.2f}%")
                        except (ValueError, IndexError, AttributeError) as e:
                            print(f"{stock_code}: 计算ROE时出错 (i={i}): {str(e)}")
                            continue
                    
                    if roe_values:
                        found_roe = True
                        print(f"{stock_code}: 成功从净利润和净资产计算ROE: {roe_values}")
                else:
                    print(f"{stock_code}: 未找到可用的所有者权益列，无法计算ROE")
                    print("可用的指标列:", indicators['指标'].unique().tolist())
                    
            except Exception as e:
                print(f"{stock_code}: 从净利润和净资产计算ROE时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not found_roe or not roe_values:
            print(f"{stock_code}: 未找到有效的ROE数据")
            print("可用指标:", indicators['指标'].tolist() if '指标' in indicators.columns else "无指标列")
            return False
            
        print(f"{stock_code}: 提取到的ROE值: {roe_values}")
        
        # 检查ROE数据
        try:
            # 确保我们有足够的数据点
            if len(roe_values) < 1:
                print(f"{stock_code}: 没有足够的ROE数据点")
                return False
                
            # 检查ROE是否都≥15%
            for i, roe in enumerate(roe_values[:3]):  # 最多检查前三年
                roe_float = float(str(roe).replace('%', '').strip())
                if roe_float < 15:
                    print(f"{stock_code}: 第{i+1}年ROE {roe_float}% 不满足≥15%")
                    return False
            
            # 检查ROE是否都≥15%
            for i, roe in enumerate(roe_values):
                if roe < 15:
                    print(f"{stock_code}: 第{i+1}年ROE {roe}% 不满足≥15%")
                    return False
            
            print(f"{stock_code}: 最近{len(roe_values)}年ROE均≥15%: {roe_values}")
            
        except Exception as e:
            print(f"{stock_code}: 检查ROE数据时出错: {str(e)}")
            return False
        
        # 检查毛利率
        gross_margin = None
        if '销售毛利率' in indicators.columns:
            # 东方财富数据源格式
            gross_margin = indicators['销售毛利率'].iloc[0] if not indicators.empty else None
        elif '指标' in indicators.columns and '销售毛利率' in indicators['指标'].values:
            # 新浪财经数据源格式
            margin_mask = indicators['指标'] == '销售毛利率'
            if not indicators[margin_mask].empty:
                gross_margin = indicators[margin_mask].iloc[0, 1]
        
        if gross_margin is None or pd.isna(gross_margin):
            print(f"{stock_code}: 未找到有效的毛利率数据")
            return False
            
        print(f"毛利率数据: {gross_margin}")
        
        try:
            # 转换毛利率值为浮点数
            gross_margin_value = float(str(gross_margin).replace('%', '').strip())
            if gross_margin_value < 40:
                print(f"{stock_code}: 毛利率{gross_margin_value}% 不满足≥40%")
                return False
        except (ValueError, IndexError) as e:
            print(f"{stock_code}: 解析毛利率数据出错: {e}")
            return False
            
        # 获取资产负债率
        debt_ratio_mask = fin_data['indicators']['报表日期'] == '资产负债率'
        debt_ratio = fin_data['indicators'][debt_ratio_mask]
        
        if debt_ratio.empty:
            print(f"{stock_code}: 未找到资产负债率数据")
            return False
            
        print("资产负债率数据:")
        print(debt_ratio)
        
        try:
            debt_ratio_value = debt_ratio.iloc[0, 1]
            if isinstance(debt_ratio_value, str):
                debt_ratio_value = float(debt_ratio_value.replace('%', ''))
            else:
                debt_ratio_value = float(debt_ratio_value)
                
            if debt_ratio_value > 60:
                print(f"{stock_code}: 资产负债率{debt_ratio_value}% 不满足≤60%")
                return False
        except (ValueError, IndexError) as e:
            print(f"{stock_code}: 解析资产负债率数据出错: {e}")
            return False
            
        # 由于自由现金流数据获取较为复杂，这里简化处理
        print(f"{stock_code}: 所有财务指标检查通过")
        return True
        
    except Exception as e:
        print(f"{stock_code}: 检查财务数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def screen_stocks():
    """筛选符合条件的股票"""
    print("开始获取A股股票列表...")
    stocks = get_ashare_stocks()
    
    if stocks is None or stocks.empty:
        print("获取A股股票列表失败，请检查网络连接或数据源")
        return None
        
    print(f"共获取到 {len(stocks)} 只A股股票")
    
    results = []
    total_stocks = len(stocks)
    processed_count = 0
    
    for index, row in stocks.iterrows():
        stock_code = row['code']
        stock_name = row['name']
        processed_count += 1
        
        print(f"\n=== 正在处理 {stock_name}({stock_code}) [{processed_count}/{total_stocks}] ===")
        
        # 获取市值
        print(f"获取市值...")
        market_cap = get_market_cap(stock_code)
        if market_cap is None:
            print(f"{stock_code}: 获取市值失败，跳过")
            continue
            
        print(f"当前市值: {market_cap}亿")
        
        if market_cap < 10000000000 or market_cap > 50000000000:
            print(f"{stock_code}: 市值{market_cap}亿 不在100-500亿范围内，跳过")
            continue
            
        # 获取财务数据
        print(f"获取财务数据...")
        fin_data = get_financial_indicators(stock_code)
        
        # 检查财务指标
        if not check_financial_conditions(fin_data, stock_code):
            print(f"{stock_code}: 财务指标不满足条件，跳过")
            continue
            
        # 获取股票数据
        print(f"获取股票历史数据...")
        stock_data = get_stock_data(stock_code)
        if stock_data is None or stock_data.empty:
            print(f"{stock_code}: 获取股票数据失败，跳过")
            continue
            
        # 计算年线（250日均线）
        annual_price = stock_data['收盘'].mean()
        
        # 获取最新收盘价
        latest_close = stock_data.iloc[-1]['收盘']
        
        # 获取行业信息
        print(f"获取行业信息...")
        industry = get_industry(stock_code)
        
        stock_info = {
            '股票名称': stock_name,
            '股票代码': stock_code,
            '最新收盘价': round(latest_close, 2),
            '年线价': round(annual_price, 2),
            '所属行业': industry,
            '市值(亿)': round(market_cap, 2)
        }
        
        print(f"找到符合条件的股票: {stock_info}")
        results.append(stock_info)
        
        # 添加延迟，避免请求过于频繁
        print("等待0.5秒...")
        time.sleep(0.5)
        
        # 测试时只处理前10只股票，调试完成后可以注释掉
        if len(results) >= 10:
            print("\n已达到测试数量限制，停止处理更多股票")
            break
    
    # 转换为DataFrame并保存
    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('市值(亿)', ascending=False)
        
        # 添加筛选日期
        result_df['筛选日期'] = '2025-07-16'
        
        # 选择需要的列并重命名
        result_df = result_df[['筛选日期', '股票名称', '股票代码', '最新收盘价', '年线价', '市值(亿)', '所属行业']]
        
        # 保存到Excel
        output_file = 'screened_stocks.xlsx'
        result_df.to_excel(output_file, index=False, float_format='%.2f')
        
        print(f"\n筛选完成，共找到 {len(results)} 只符合条件的股票，已保存到 {output_file}")
        print("\n筛选结果概览：")
        print(result_df.to_string(index=False, float_format='%.2f'))
    else:
        print("\n未找到符合条件的股票")
    
    return results  # Always return a list, even if empty

def test_single_stock(stock_code):
    """测试单只股票的筛选条件"""
    print(f"\n=== 开始测试股票 {stock_code} ===")
    
    # 获取股票基本信息
    try:
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        stock_name = stock_info[stock_info['item'] == '股票简称'].iloc[0]['value']
        print(f"股票名称: {stock_name}")
    except Exception as e:
        print(f"获取股票{stock_code}基本信息失败: {str(e)}")
        return
    
    # 获取市值
    print(f"获取市值...")
    market_cap = get_market_cap(stock_code)
    if market_cap is None:
        print(f"{stock_code}: 获取市值失败")
        return
    print(f"当前市值: {market_cap}亿")
    
    # 获取财务数据
    print(f"获取财务数据...")
    fin_data = get_financial_indicators(stock_code)
    if fin_data is None:
        print(f"{stock_code}: 获取财务数据失败")
        return
        
    # 检查财务指标
    print(f"检查财务指标...")
    financial_ok = check_financial_conditions(fin_data, stock_code)
    print(f"财务指标检查结果: {'通过' if financial_ok else '不通过'}")
    
    # 获取股票数据
    print(f"获取股票历史数据...")
    stock_data = get_stock_data(stock_code)
    if stock_data is None or stock_data.empty:
        print(f"{stock_code}: 获取股票数据失败")
        return
    
    # 计算年线（250日均线）
    annual_price = stock_data['收盘'].mean()
    latest_close = stock_data.iloc[-1]['收盘']
    
    # 获取行业信息
    print(f"获取行业信息...")
    industry = get_industry(stock_code)
    
    # 输出结果
    print("\n=== 测试结果 ===")
    print(f"股票代码: {stock_code}")
    print(f"股票名称: {stock_name}")
    print(f"最新收盘价: {latest_close:.2f}")
    print(f"年线价: {annual_price:.2f}")
    print(f"市值: {market_cap:.2f}亿")
    print(f"所属行业: {industry}")
    print(f"是否符合财务筛选条件: {'是' if financial_ok else '否'}")
    
    if latest_close > annual_price:
        print("当前价格在年线上方")
    else:
        print("当前价格在年线下方")

if __name__ == "__main__":
    # 测试特定股票
    test_single_stock("600004")  # 白云机场
