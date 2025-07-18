"""
Convertible Bond (可转债) Investment Strategy

This module implements a convertible bond investment strategy with the following rules:
- Buy when price first drops below 90
- Strong buy when price drops below 85
- Sell when price rises above 120
- Strong sell when price rises above 130
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import pandas as pd
import akshare as ak

@dataclass
class BondSignal:
    """Class to track signals for a single convertible bond"""
    symbol: str
    name: str
    price: float
    signal: str  # 'buy', 'strong_buy', 'sell', 'strong_sell'
    timestamp: datetime = field(default_factory=datetime.now)

class ConvertibleBondStrategy:
    """
    Convertible Bond Investment Strategy
    
    This strategy monitors convertible bonds and generates signals based on price levels:
    - Buy when price first drops below 90
    - Strong buy when price drops below 85
    - Sell when price rises above 120
    - Strong sell when price rises above 130
    """
    
    def __init__(self):
        """Initialize the strategy with default price thresholds"""
        self.buy_threshold = 90.0
        self.strong_buy_threshold = 85.0
        self.sell_threshold = 120.0
        self.strong_sell_threshold = 130.0
        self.bond_states: Dict[str, dict] = {}
        self.signals: List[BondSignal] = []
    
    def fetch_convertible_bonds(self) -> pd.DataFrame:
        """
        Fetch current convertible bond data from AKShare
        
        Returns:
            DataFrame with convertible bond data
        """
        try:
            # Get convertible bond data
            df = ak.bond_zh_hs_cov_spot()
            
            # Map available columns to our expected column names
            # Note: Some expected columns may not be available in the current AKShare response
            column_mapping = {
                'symbol': 'symbol',
                'name': 'name',
                'trade': 'price',
                'changepercent': 'change_pct',
                'volume': 'volume',
                'amount': 'amount',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'settlement': 'settlement',
                'ticktime': 'update_time'
            }
            
            # Only keep columns that exist in the DataFrame
            existing_columns = [col for col in column_mapping.keys() if col in df.columns]
            df = df[existing_columns].copy()
            
            # Rename columns to our standard names
            df = df.rename(columns=column_mapping)
            
            # Convert numeric columns
            numeric_cols = ['price', 'change_pct', 'volume', 'amount', 'open', 'high', 'low', 'settlement']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error fetching convertible bond data: {e}")
            return pd.DataFrame()
    
    def analyze_bonds(self, df: pd.DataFrame) -> List[BondSignal]:
        """
        Analyze convertible bonds and generate signals
        
        Args:
            df: DataFrame with convertible bond data
            
        Returns:
            List of BondSignal objects with trading signals
        """
        signals = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            price = row['price']
            name = row['name']
            
            # Initialize bond state if not exists
            if symbol not in self.bond_states:
                self.bond_states[symbol] = {
                    'below_90': False,
                    'below_85': False,
                    'above_120': False,
                    'above_130': False,
                    'last_signal': None
                }
            
            state = self.bond_states[symbol]
            signal = None
            
            # Check for buy signals (price decreasing)
            if price < self.strong_buy_threshold and not state['below_85']:
                signal = 'strong_buy'
                state.update({
                    'below_85': True,
                    'below_90': True,
                    'last_signal': signal
                })
            elif price < self.buy_threshold and not state['below_90']:
                signal = 'buy'
                state.update({
                    'below_90': True,
                    'last_signal': signal
                })
            
            # Check for sell signals (price increasing)
            elif price > self.strong_sell_threshold and not state['above_130']:
                signal = 'strong_sell'
                state.update({
                    'above_130': True,
                    'above_120': True,
                    'last_signal': signal
                })
            elif price > self.sell_threshold and not state['above_120']:
                signal = 'sell'
                state.update({
                    'above_120': True,
                    'last_signal': signal
                })
            
            # Reset states when price moves back to normal range
            if price >= self.buy_threshold:
                state['below_90'] = False
            if price >= self.strong_buy_threshold:
                state['below_85'] = False
            if price <= self.sell_threshold:
                state['above_120'] = False
            if price <= self.strong_sell_threshold:
                state['above_130'] = False
            
            # Create signal if any condition was met
            if signal:
                bond_signal = BondSignal(
                    symbol=symbol,
                    name=name,
                    price=price,
                    signal=signal
                )
                signals.append(bond_signal)
                self.signals.append(bond_signal)
        
        return signals
    
    def get_cheap_bonds(self, df: pd.DataFrame, max_price: float = 90.0) -> pd.DataFrame:
        """
        Get convertible bonds priced below a certain threshold
        
        Args:
            df: DataFrame with convertible bond data
            max_price: Maximum price to include (default: 90.0)
            
        Returns:
            Filtered DataFrame with cheap convertible bonds
        """
        if df.empty:
            return pd.DataFrame()
        
        return df[df['price'] < max_price].sort_values('price')
    
    def run_strategy(self) -> Dict:
        """
        Run the convertible bond strategy
        
        Returns:
            Dictionary with strategy results including signals and cheap bonds
        """
        # Fetch current convertible bond data
        df = self.fetch_convertible_bonds()
        
        if df.empty:
            return {
                'success': False,
                'error': 'Failed to fetch convertible bond data',
                'signals': [],
                'cheap_bonds': pd.DataFrame()
            }
        
        # Generate signals
        signals = self.analyze_bonds(df)
        
        # Get cheap bonds (priced below 90)
        cheap_bonds = self.get_cheap_bonds(df, self.buy_threshold)
        
        return {
            'success': True,
            'signals': signals,
            'cheap_bonds': cheap_bonds,
            'timestamp': datetime.now()
        }


def example_usage():
    """Example usage of the ConvertibleBondStrategy"""
    print("Running Convertible Bond Strategy...")
    
    # Initialize strategy
    strategy = ConvertibleBondStrategy()
    
    # Run strategy
    result = strategy.run_strategy()
    
    if not result['success']:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print("\n=== Trading Signals ===")
    if not result['signals']:
        print("No new signals generated.")
    else:
        for signal in result['signals']:
            print(f"{signal.timestamp} - {signal.symbol} ({signal.name}): {signal.signal.upper()} at {signal.price}")
    
    # Print cheap bonds (price < 90)
    print("\n=== Cheap Bonds (Price < 90) ===")
    if result['cheap_bonds'].empty:
        print("No cheap bonds found.")
    else:
        # Select only available columns from the DataFrame
        available_columns = [col for col in ['symbol', 'name', 'price', 'change_pct'] 
                           if col in result['cheap_bonds'].columns]
        
        if not available_columns:
            print("No data available for cheap bonds.")
        else:
            # Format the output with price and change percentage if available
            if 'price' in available_columns:
                result['cheap_bonds'] = result['cheap_bonds'].sort_values('price')
            
            # Format the price and change percentage for better readability
            display_df = result['cheap_bonds'][available_columns].copy()
            
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
            
            if 'change_pct' in display_df.columns:
                display_df['change_pct'] = display_df['change_pct'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"
                )
            
            print(display_df.to_string(index=False))


if __name__ == "__main__":
    example_usage()
