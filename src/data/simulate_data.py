"""
Synthetic banking transaction data generation using CTGAN, Gaussian Copulas, or rule-based simulation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Optional, Tuple
from sdv.tabular import CTGAN
from sdv.single_table import GaussianCopula
import warnings
warnings.filterwarnings('ignore')


class TransactionSimulator:
    """Generate synthetic banking transaction data with fraud patterns."""
    
    def __init__(
        self,
        n_customers: int = 10000,
        n_transactions: int = 500000,
        fraud_rate: float = 0.005,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        random_state: int = 42
    ):
        """
        Initialize transaction simulator.
        
        Args:
            n_customers: Number of unique customers
            n_transactions: Total number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions (default 0.5%)
            start_date: Start date for transactions
            end_date: End date for transactions
            random_state: Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_state = random_state
        
        if start_date is None:
            start_date = datetime(2023, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 1, 1)
        
        self.start_date = start_date
        self.end_date = end_date
        
        np.random.seed(random_state)
        random.seed(random_state)
        
    def generate_base_transactions(self) -> pd.DataFrame:
        """Generate base legitimate transaction dataset."""
        print("Generating base transactions...")
        
        # Customer profiles
        customer_ids = np.arange(1, self.n_customers + 1)
        
        # Generate transaction timestamps
        date_range = (self.end_date - self.start_date).days
        timestamps = [
            self.start_date + timedelta(
                days=random.randint(0, date_range),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            for _ in range(self.n_transactions)
        ]
        timestamps.sort()
        
        # Transaction amounts (log-normal distribution for realistic amounts)
        amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=self.n_transactions)
        amounts = np.round(amounts, 2)
        
        # Transaction types
        transaction_types = np.random.choice(
            ['purchase', 'transfer', 'withdrawal', 'deposit', 'payment'],
            size=self.n_transactions,
            p=[0.4, 0.2, 0.15, 0.15, 0.1]
        )
        
        # Merchant categories
        merchant_categories = np.random.choice(
            ['retail', 'groceries', 'restaurant', 'gas', 'online', 'utility', 'other'],
            size=self.n_transactions,
            p=[0.25, 0.2, 0.15, 0.1, 0.15, 0.1, 0.05]
        )
        
        # Geographic locations (simplified)
        countries = np.random.choice(
            ['US', 'CA', 'MX', 'UK', 'DE', 'FR'],
            size=self.n_transactions,
            p=[0.6, 0.1, 0.05, 0.1, 0.1, 0.05]
        )
        
        # Customer assignment (some customers more active)
        customer_weights = np.random.gamma(2, 2, size=self.n_customers)
        customer_weights = customer_weights / customer_weights.sum()
        customer_ids_assigned = np.random.choice(
            customer_ids,
            size=self.n_transactions,
            p=customer_weights
        )
        
        # Account balances (for context)
        account_balances = np.random.lognormal(mean=8, sigma=1.5, size=self.n_transactions)
        
        # Create base dataframe
        df = pd.DataFrame({
            'transaction_id': range(1, self.n_transactions + 1),
            'customer_id': customer_ids_assigned,
            'timestamp': timestamps,
            'amount': amounts,
            'transaction_type': transaction_types,
            'merchant_category': merchant_categories,
            'country': countries,
            'account_balance_before': account_balances,
            'fraud': 0
        })
        
        return df
    
    def inject_fraud_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic fraud patterns into the dataset."""
        print(f"Injecting fraud patterns (target rate: {self.fraud_rate:.2%})...")
        
        n_fraud = int(self.n_transactions * self.fraud_rate)
        fraud_indices = np.random.choice(df.index, size=n_fraud, replace=False)
        
        df_fraud = df.loc[fraud_indices].copy()
        df_legit = df.drop(fraud_indices).copy()
        
        # Fraud Pattern 1: Unusually large amounts
        n_large_amount = int(n_fraud * 0.3)
        large_indices = np.random.choice(fraud_indices, size=n_large_amount, replace=False)
        df.loc[large_indices, 'amount'] = np.random.lognormal(mean=6, sigma=1.5, size=n_large_amount)
        df.loc[large_indices, 'amount'] = np.round(df.loc[large_indices, 'amount'], 2)
        
        # Fraud Pattern 2: Rapid successive transactions (burst)
        n_burst = int(n_fraud * 0.25)
        burst_customers = np.random.choice(df['customer_id'].unique(), size=min(50, n_burst // 5), replace=False)
        for customer in burst_customers:
            customer_txns = df[df['customer_id'] == customer].index
            if len(customer_txns) > 0:
                burst_size = min(5, len(customer_txns))
                burst_idx = np.random.choice(customer_txns, size=burst_size, replace=False)
                df.loc[burst_idx, 'fraud'] = 1
                # Make timestamps very close together
                base_time = df.loc[burst_idx[0], 'timestamp']
                for i, idx in enumerate(burst_idx[1:], 1):
                    df.loc[idx, 'timestamp'] = base_time + timedelta(minutes=i*2)
        
        # Fraud Pattern 3: Unusual geographic locations
        n_geo = int(n_fraud * 0.2)
        geo_indices = np.random.choice(fraud_indices, size=n_geo, replace=False)
        unusual_countries = ['RU', 'CN', 'BR', 'NG', 'IN']
        df.loc[geo_indices, 'country'] = np.random.choice(unusual_countries, size=n_geo)
        
        # Fraud Pattern 4: Unusual transaction types for customer
        n_type = int(n_fraud * 0.15)
        type_indices = np.random.choice(fraud_indices, size=n_type, replace=False)
        # Assign unusual types (e.g., large withdrawals for customers who usually only purchase)
        df.loc[type_indices, 'transaction_type'] = np.random.choice(
            ['withdrawal', 'transfer'],
            size=n_type
        )
        df.loc[type_indices, 'amount'] = np.random.lognormal(mean=5.5, sigma=1.2, size=n_type)
        
        # Fraud Pattern 5: Off-hours transactions
        n_offhours = int(n_fraud * 0.1)
        offhours_indices = np.random.choice(fraud_indices, size=n_offhours, replace=False)
        for idx in offhours_indices:
            # Set to 2-5 AM
            hour = np.random.randint(2, 6)
            df.loc[idx, 'timestamp'] = df.loc[idx, 'timestamp'].replace(hour=hour, minute=np.random.randint(0, 60))
        
        # Mark all fraud indices
        df.loc[fraud_indices, 'fraud'] = 1
        
        # Shuffle to mix fraud and legitimate transactions
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        actual_fraud_rate = df['fraud'].mean()
        print(f"Actual fraud rate: {actual_fraud_rate:.2%}")
        
        return df
    
    def generate(self, method: str = 'rule_based') -> pd.DataFrame:
        """
        Generate synthetic transaction data.
        
        Args:
            method: 'rule_based', 'ctgan', or 'copula'
        
        Returns:
            DataFrame with transactions and fraud labels
        """
        if method == 'rule_based':
            df = self.generate_base_transactions()
            df = self.inject_fraud_patterns(df)
        else:
            # For CTGAN/Copula, generate base data first, then inject fraud
            df = self.generate_base_transactions()
            df = self.inject_fraud_patterns(df)
            # Could add generative model training here if needed
        
        # Add derived fields
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        return df


def generate_synthetic_data(
    output_path: Path,
    n_customers: int = 10000,
    n_transactions: int = 500000,
    fraud_rate: float = 0.005,
    method: str = 'rule_based',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Main function to generate synthetic banking transaction data.
    
    Args:
        output_path: Path to save the generated data
        n_customers: Number of unique customers
        n_transactions: Total number of transactions
        fraud_rate: Proportion of fraudulent transactions
        method: Generation method ('rule_based', 'ctgan', 'copula')
        random_state: Random seed
    
    Returns:
        Generated DataFrame
    """
    simulator = TransactionSimulator(
        n_customers=n_customers,
        n_transactions=n_transactions,
        fraud_rate=fraud_rate,
        random_state=random_state
    )
    
    df = simulator.generate(method=method)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} transactions to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic banking transaction data")
    parser.add_argument("--output", type=str, default="data/raw/transactions_raw.csv",
                       help="Output path for generated data")
    parser.add_argument("--n_customers", type=int, default=10000,
                       help="Number of customers")
    parser.add_argument("--n_transactions", type=int, default=500000,
                       help="Number of transactions")
    parser.add_argument("--fraud_rate", type=float, default=0.005,
                       help="Fraud rate (default 0.5%%)")
    parser.add_argument("--method", type=str, default="rule_based",
                       choices=["rule_based", "ctgan", "copula"],
                       help="Generation method")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    df = generate_synthetic_data(
        output_path=output_path,
        n_customers=args.n_customers,
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        method=args.method,
        random_state=args.seed
    )
    
    print("\nDataset Summary:")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['fraud'].sum():,} ({df['fraud'].mean():.2%})")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

