"""
DCF (Discounted Cash Flow) Calculator v2

This module implements a DCF valuation model based on projected free cash flows,
discount rate, and terminal growth rate. The model calculates the present value
of future cash flows and terminal value to estimate the enterprise value.
"""
from dataclasses import dataclass
from typing import List, Tuple

import akshare as ak
import numpy as np
import pandas as pd

from stock_screener import get_financial_indicators


@dataclass
class DCFResult:
    """Container for DCF calculation results."""
    enterprise_value: float
    present_values: List[float]
    terminal_value: float
    discounted_terminal_value: float
    discount_rate: float
    perpetual_growth_rate: float
    current_price: float = 0.0
    shares_outstanding: float = 0.0
    net_debt: float = 0.0
    
    @property
    def equity_value(self) -> float:
        """Calculate equity value from enterprise value."""
        return max(self.enterprise_value - self.net_debt, self.enterprise_value * 0.1)  # At least 10% of EV
    
    @property
    def intrinsic_value_per_share(self) -> float:
        """Calculate intrinsic value per share."""
        if self.shares_outstanding > 0:
            return self.equity_value * 100 / self.shares_outstanding
        return self.equity_value * 10  # Fallback if shares outstanding is missing
    
    @property
    def margin_of_safety(self) -> float:
        """Calculate margin of safety percentage."""
        if self.current_price <= 0 or self.intrinsic_value_per_share <= 0:
            return 0.0
        return ((self.intrinsic_value_per_share - self.current_price) / 
                self.intrinsic_value_per_share * 100)
    
    @property
    def recommendation(self) -> str:
        """Generate investment recommendation based on margin of safety."""
        mos = self.margin_of_safety
        if mos > 30:
            return "强烈买入 (Strong Buy)"
        elif mos > 15:
            return "买入 (Buy)"
        elif mos > 0:
            return "持有 (Hold)"
        else:
            return "卖出 (Sell)"
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for better visualization."""
        years = list(range(1, len(self.present_values) + 1))
        data = {
            'Year': years,
            'FCF': [pv * (1 + self.discount_rate) ** year for year, pv in zip(years, self.present_values)],
            'Present Value': self.present_values,
            'Cumulative PV': np.cumsum(self.present_values).tolist()
        }
        
        # Add terminal value row
        df = pd.DataFrame(data)
        terminal_row = pd.DataFrame({
            'Year': ['Terminal'],
            'FCF': [np.nan],
            'Present Value': [self.discounted_terminal_value],
            'Cumulative PV': [self.enterprise_value]
        })
        
        return pd.concat([df, terminal_row], ignore_index=True)


def calculate_dcf(
    free_cash_flows: List[float],
    discount_rate: float = 0.10,
    perpetual_growth_rate: float = 0.02,
    verbose: bool = True
) -> DCFResult:
    """
    Calculate the DCF valuation based on projected free cash flows.
    
    Args:
        free_cash_flows: List of projected free cash flows for each year
        discount_rate: Discount rate (WACC) as a decimal (e.g., 0.10 for 10%)
        perpetual_growth_rate: Perpetual growth rate for terminal value as a decimal (e.g., 0.02 for 2%)
        verbose: Whether to print detailed calculation steps
        
    Returns:
        DCFResult object containing the valuation results
        
    Raises:
        ValueError: If perpetual_growth_rate >= discount_rate (violates Gordon Growth Model)
    """
    if perpetual_growth_rate >= discount_rate:
        raise ValueError("Perpetual growth rate must be less than the discount rate")
    
    n_years = len(free_cash_flows)
    
    if verbose:
        print(f"Calculating DCF with {n_years} years of projections")
        print(f"Discount rate: {discount_rate:.1%}")
        print(f"Perpetual growth rate: {perpetual_growth_rate:.1%}\n")
    
    # 1. Calculate present value of projected FCFs
    present_values = []
    for t, fcf in enumerate(free_cash_flows, 1):
        pv = fcf / ((1 + discount_rate) ** t)
        present_values.append(pv)
        
        if verbose:
            print(f"Year {t} FCF: {fcf:,.2f}, PV: {pv:,.2f}")
    
    # 2. Calculate terminal value using Gordon Growth Model
    final_year_fcf = free_cash_flows[-1]
    terminal_value = (final_year_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
    
    # 3. Discount terminal value to present
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** n_years)
    
    if verbose:
        print(f"\nTerminal Value: {terminal_value:,.2f}")
        print(f"Discounted Terminal Value: {discounted_terminal_value:,.2f}")
    
    # 4. Calculate enterprise value
    enterprise_value = sum(present_values) + discounted_terminal_value
    
    if verbose:
        print(f"\nPresent value of FCFs: {sum(present_values):,.2f}")
        print(f"Enterprise Value: {enterprise_value:,.2f}")
    
    return DCFResult(
        enterprise_value=enterprise_value,
        present_values=present_values,
        terminal_value=terminal_value,
        discounted_terminal_value=discounted_terminal_value,
        discount_rate=discount_rate,
        perpetual_growth_rate=perpetual_growth_rate
    )


def dcf_sensitivity_analysis(
    base_fcfs: List[float],
    base_discount_rate: float = 0.10,
    base_growth_rate: float = 0.02,
    discount_rate_range: Tuple[float, float, float] = (0.08, 0.12, 0.01),
    growth_rate_range: Tuple[float, float, float] = (0.01, 0.03, 0.005)
) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying discount and growth rates.
    
    Args:
        base_fcfs: Base case free cash flow projections
        base_discount_rate: Base discount rate
        base_growth_rate: Base perpetual growth rate
        discount_rate_range: (start, stop, step) for discount rate variations
        growth_rate_range: (start, stop, step) for growth rate variations
        
    Returns:
        DataFrame with sensitivity analysis results
    """
    discount_rates = np.arange(*discount_rate_range)
    growth_rates = np.arange(*growth_rate_range)
    
    results = []
    
    for dr in discount_rates:
        row = {}
        for gr in growth_rates:
            try:
                result = calculate_dcf(base_fcfs, dr, gr, verbose=False)
                row[f"{gr:.1%} Growth"] = result.enterprise_value
            except ValueError:
                row[f"{gr:.1%} Growth"] = np.nan
        results.append(row)
    
    df = pd.DataFrame(results, index=[f"{dr:.0%} DR" for dr in discount_rates])
    return df


def main():
    """Example usage of the DCF calculator."""
    # Example FCF projections (in millions)
    fcf_projections = [100, 120, 140, 160, 180]
    
    print("=== DCF Valuation Example ===\n")
    
    # Calculate base case
    result = calculate_dcf(
        free_cash_flows=fcf_projections,
        discount_rate=0.10,
        perpetual_growth_rate=0.02
    )
    
    # Display results
    print("\n=== DCF Valuation Results ===")
    print(f"Enterprise Value: {result.enterprise_value:,.2f}")
    
    # Show detailed breakdown
    print("\n=== Detailed Calculation ===")
    df = result.to_dataframe()
    print("\nYearly Cash Flows and Present Values:")
    print(df.to_string(index=False))
    
    # Perform sensitivity analysis
    print("\n=== Sensitivity Analysis ===")
    print("Varying discount rates (DR) and perpetual growth rates (G):")
    sensitivity = dcf_sensitivity_analysis(
        fcf_projections,
        base_discount_rate=0.10,
        base_growth_rate=0.02,
        discount_rate_range=(0.08, 0.13, 0.01),
        growth_rate_range=(0.01, 0.035, 0.005)
    )
    print("\nEnterprise Value under different scenarios:")
    print(sensitivity.style.format("{:,.0f}"))


def fetch_stock_data(stock_code: str, n_years: int = 5):
    """
    通过 akshare 获取指定股票代码的近 n 年自由现金流（FCF）数据。
    优先使用经营活动现金流净额和资本性支出（如有），否则回退到净利润等。
    返回: List[float]，按年份从旧到新
    """
    fin_data = get_financial_indicators(stock_code)
    if fin_data is None or fin_data.get("cash_flow") is None or fin_data["cash_flow"].empty:
        print(f"未能获取到 {stock_code} 的现金流量数据")
        return None

    cash_flow = fin_data["cash_flow"]
    # 兼容不同列名
    fcf_candidates = [
        ["自由现金流", "FCF"],
        ["经营活动产生的现金流量净额", "经营现金流净额", "NetCashFlowsFromOperatingActivities"],
        ["净利润", "NetProfit"]
    ]
    # 优先找自由现金流
    for cols in fcf_candidates:
        for col in cols:
            if col in cash_flow.columns:
                fcf_series = cash_flow[col].dropna().astype(float)
                if len(fcf_series) >= n_years:
                    # 取最近 n 年，按时间升序
                    return fcf_series.iloc[:n_years][::-1].tolist()

    # 自动计算FCF：经营活动现金流净额 - CapEx
    op_cols = ["经营活动产生的现金流量净额", "经营现金流净额", "NetCashFlowsFromOperatingActivities"]
    capex_cols = [
        "购建固定资产、无形资产和其他长期资产所支付的现金",
        "购建固定资产支付的现金",
        "购建无形资产支付的现金",
        "购建长期资产支付的现金",
        "投资支付的现金",
        "NetCashPaidForAcquisitionOfFixedIntangibleAndOtherLongTermAssets",
    ]
    op_col = next((col for col in op_cols if col in cash_flow.columns), None)
    capex_col = next((col for col in capex_cols if col in cash_flow.columns), None)
    if op_col and capex_col:
        op_series = cash_flow[op_col].dropna().astype(float)
        capex_series = cash_flow[capex_col].dropna().astype(float)
        # 对齐索引（报告期）
        fcf_df = pd.DataFrame({"op": op_series, "capex": capex_series})
        fcf_df = fcf_df.dropna()
        if len(fcf_df) >= n_years:
            fcf_calc = (fcf_df["op"] - abs(fcf_df["capex"]))
            print(f"自动计算自由现金流（经营现金流-资本开支），列: {op_col} - abs({capex_col})")
            return fcf_calc.iloc[:n_years][::-1].tolist()
    print(f"未找到可用的自由现金流列，尝试经营现金流或净利润")
    return None

def get_stock_price(stock_code: str) -> float:
    """Get current stock price using akshare."""
    try:
        # Try to get real-time data first
        stock_zh_a_spot = ak.stock_zh_a_spot()
        stock_data = stock_zh_a_spot[stock_zh_a_spot['代码'] == stock_code]
        
        if not stock_data.empty and '最新价' in stock_data.columns:
            return float(stock_data['最新价'].iloc[0])
            
        # Fallback to historical data if real-time fails
        hist_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
        if not hist_data.empty and '收盘' in hist_data.columns:
            return float(hist_data['收盘'].iloc[-1])
            
    except Exception as e:
        print(f"获取股票价格时出错: {e}")
    
    print("无法获取股票价格，请手动输入当前价格:")
    while True:
        try:
            return float(input("当前价格 (元): "))
        except ValueError:
            print("请输入有效的数字")

def dcf_from_stock_code(stock_code: str, n_years: int = 5, discount_rate: float = 0.10, 
                       perpetual_growth_rate: float = 0.02, current_price: float = None):
    """
    输入股票代码，自动获取 FCF 并计算 DCF 估值。
    
    Args:
        stock_code: 股票代码，例如 '600519'
        n_years: 使用最近几年的自由现金流数据
        discount_rate: 折现率
        perpetual_growth_rate: 永续增长率
        current_price: 当前股价（如不提供则自动获取）
    """
    print(f"\n=== 股票 {stock_code} DCF 估值 ===")
    
    # 获取自由现金流数据
    fcfs = fetch_stock_data(stock_code, n_years=n_years)
    if not fcfs:
        print("无法获取到有效的自由现金流数据，无法估值。")
        return None
    print(f"自由现金流序列（最近{n_years}年，旧→新）: {fcfs}")
    
    # 计算DCF
    result = calculate_dcf(fcfs, discount_rate, perpetual_growth_rate, verbose=True)
    
    # 获取当前股价
    if current_price is None:
        current_price = get_stock_price(stock_code)
    result.current_price = current_price
    
    # 设置其他必要参数（实际项目中应从财务数据中获取）
    result.shares_outstanding = 100  # 示例值，实际应从财务数据获取
    result.net_debt = 0  # 示例值，实际应从财务数据获取
    
    # 打印估值结果
    print("\n=== 估值结果 ===")
    print(f"当前股价: {current_price:.2f} 元")
    print(f"每股内在价值: {result.intrinsic_value_per_share:,.2f} 元")
    print(f"安全边际: {result.margin_of_safety:.1f}%")
    print(f"投资建议: {result.recommendation}")
    
    print(f"\n企业价值（Enterprise Value）: {result.enterprise_value:,.2f} 万元")
    print(f"股权价值（Equity Value）: {result.equity_value:,.2f} 万元")
    
    print("\n详细分年现值:")
    print(result.to_dataframe().to_string(index=False))
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DCF估值计算器 v2: 支持股票代码自动抓取数据")
    parser.add_argument("--stock", type=str, help="A股股票代码，如 600519")
    parser.add_argument("--years", type=int, default=5, help="使用最近几年自由现金流，默认5年")
    parser.add_argument("--discount", type=float, default=0.10, help="贴现率，默认0.10")
    parser.add_argument("--growth", type=float, default=0.02, help="永续增长率，默认0.02")
    args = parser.parse_args()

    if args.stock:
        dcf_from_stock_code(args.stock, n_years=args.years, discount_rate=args.discount, perpetual_growth_rate=args.growth)
    else:
        main()
