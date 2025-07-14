# A股量化投资工具

一个基于Python的A股量化投资工具，主要关注均线指标（5/10/20/60/200天）和换手率指标，帮助投资者进行技术分析和策略回测。

## 功能特点

- **数据获取**：自动获取A股历史行情数据
- **技术指标**：计算多种均线（5/10/20/60/200天）和换手率指标
- **策略回测**：基于均线和换手率的交易策略回测
- **可视化分析**：提供丰富的图表展示，包括价格走势、交易信号、换手率等
- **绩效评估**：计算收益率、夏普比率、最大回撤等关键指标

## 安装指南

1. 克隆本仓库到本地：
   ```bash
   git clone https://github.com/yourusername/a_share_quant_tool.git
   cd a_share_quant_tool
   ```

2. 创建并激活虚拟环境（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   .\venv\Scripts\activate  # Windows
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 快速开始

1. 运行主程序：
   ```bash
   python main.py
   ```

2. 程序会：
   - 自动下载指定股票的历史数据
   - 计算技术指标
   - 执行策略回测
   - 显示分析结果和图表

## 自定义配置

### 修改股票代码
在`main.py`中修改以下代码：
```python
stock_code = '000001'  # 修改为你想分析的股票代码
```

### 调整时间范围
在`main.py`中修改以下代码：
```python
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')  # 3年数据
```

### 调整策略参数
在`main.py`中修改以下代码：
```python
strategy = MATurnoverStrategy(
    ma_windows=[5, 10, 20, 60, 200],  # 均线周期
    turnover_ma_windows=[5, 10],      # 换手率均线周期
    turnover_threshold=1.5            # 换手率阈值
)
```

## 项目结构

```
a_share_quant_tool/
├── data/                    # 数据存储目录
├── strategies/              # 策略模块
│   └── ma_turnover_strategy.py  # 均线+换手率策略
├── utils/                   # 工具模块
│   ├── data_fetcher.py      # 数据获取与处理
│   └── visualization.py     # 可视化工具
├── main.py                 # 主程序
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明
```

## 策略说明

### 买入信号
1. 收盘价在5日均线之上
2. 收盘价在10日均线之上
3. 5日均线向上
4. 均线呈多头排列（5>10>20>60>200）
5. 换手率超过其5日均线的1.5倍

### 卖出信号
1. 收盘价跌破10日均线
2. 或收盘价跌破20日均线

## 回测指标说明

- **总收益率**：投资期间的总收益率
- **年化收益率**：年化后的收益率
- **夏普比率**：风险调整后的收益指标，越高越好
- **最大回撤**：投资期间最大亏损幅度，越小越好
- **交易次数**：策略发出的交易信号数量

## 注意事项

1. 本工具仅供学习参考，不构成投资建议
2. 实际交易需考虑交易成本、滑点等因素
3. 历史表现不代表未来收益
4. 请遵守相关法律法规

## 依赖库

- pandas
- numpy
- matplotlib
- akshare
- tushare
- backtrader
- seaborn

## 许可证

MIT License
