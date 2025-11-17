"""
Feature engineering for fraud detection: rolling windows, aggregates, velocity features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path


class FeatureEngineer:
    """Feature engineering pipeline for transaction data."""
    
    def __init__(self):
        self.feature_names = []
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        return df
    
    def create_rolling_window_features(
        self,
        df: pd.DataFrame,
        windows: List[str] = ['1H', '6H', '24H', '7D']
    ) -> pd.DataFrame:
        """
        Create rolling window aggregations per customer.
        
        Args:
            df: Transaction dataframe (must be sorted by timestamp)
            windows: List of window sizes (pandas offset strings)
        
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        df = df.sort_values(['customer_id', 'timestamp'])
        
        # Group by customer
        grouped = df.groupby('customer_id')
        
        for window in windows:
            # Rolling count
            df[f'txn_count_{window}'] = grouped['transaction_id'].transform(
                lambda x: x.rolling(window=window, on=df.loc[x.index, 'timestamp']).count()
            )
            
            # Rolling sum of amounts
            df[f'amount_sum_{window}'] = grouped['amount'].transform(
                lambda x: x.rolling(window=window, on=df.loc[x.index, 'timestamp']).sum()
            )
            
            # Rolling mean
            df[f'amount_mean_{window}'] = grouped['amount'].transform(
                lambda x: x.rolling(window=window, on=df.loc[x.index, 'timestamp']).mean()
            )
            
            # Rolling max
            df[f'amount_max_{window}'] = grouped['amount'].transform(
                lambda x: x.rolling(window=window, on=df.loc[x.index, 'timestamp']).max()
            )
        
        # Fill NaN with 0 for first transactions
        rolling_cols = [col for col in df.columns if any(w in col for w in windows)]
        df[rolling_cols] = df[rolling_cols].fillna(0)
        
        return df
    
    def create_customer_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregate features."""
        df = df.copy()
        df = df.sort_values(['customer_id', 'timestamp'])
        
        grouped = df.groupby('customer_id')
        
        # Historical aggregates (up to current transaction)
        df['customer_txn_count'] = grouped.cumcount() + 1
        df['customer_total_amount'] = grouped['amount'].transform('cumsum')
        df['customer_avg_amount'] = df['customer_total_amount'] / df['customer_txn_count']
        
        # Days since first transaction
        df['days_since_first_txn'] = (
            df['timestamp'] - grouped['timestamp'].transform('first')
        ).dt.total_seconds() / 86400
        
        # Transaction frequency (transactions per day)
        df['customer_txn_frequency'] = df['customer_txn_count'] / (df['days_since_first_txn'] + 1)
        
        return df
    
    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity features (rate of transactions/amounts)."""
        df = df.copy()
        df = df.sort_values(['customer_id', 'timestamp'])
        
        grouped = df.groupby('customer_id')
        
        # Time since last transaction
        df['time_since_last_txn'] = grouped['timestamp'].diff().dt.total_seconds() / 3600  # hours
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(24)  # Default for first transaction
        
        # Amount velocity (change in amount)
        df['amount_velocity'] = grouped['amount'].diff()
        df['amount_velocity'] = df['amount_velocity'].fillna(0)
        
        # Transaction burst indicator (multiple transactions in short time)
        df['is_burst'] = (df['time_since_last_txn'] < 1).astype(int)
        
        # Burst size (consecutive transactions within 1 hour)
        df['burst_size'] = (
            df.groupby(['customer_id', (df['time_since_last_txn'] >= 1).cumsum()])
            .cumcount() + 1
        )
        
        return df
    
    def create_risk_heuristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rule-based risk score features."""
        df = df.copy()
        
        # Large amount relative to customer average
        df['amount_vs_avg'] = df['amount'] / (df['customer_avg_amount'] + 1)
        df['is_large_amount'] = (df['amount_vs_avg'] > 3).astype(int)
        
        # Unusual transaction type for customer
        # (simplified - in practice would check historical patterns)
        df['is_unusual_type'] = 0  # Placeholder
        
        # Unusual country (simplified)
        customer_countries = df.groupby('customer_id')['country'].apply(set)
        df['customer_countries'] = df['customer_id'].map(customer_countries)
        df['is_unusual_country'] = df.apply(
            lambda x: 1 if x['country'] not in x['customer_countries'] else 0,
            axis=1
        )
        df = df.drop('customer_countries', axis=1)
        
        # Risk score (simple heuristic)
        df['risk_score'] = (
            df['is_large_amount'] * 2 +
            df['is_unusual_country'] * 1.5 +
            df['is_burst'] * 1 +
            df['is_night'] * 0.5 +
            (df['amount'] > df['amount'].quantile(0.95)).astype(int) * 1
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        df = df.copy()
        
        # Amount × transaction type interaction
        type_encoding = pd.get_dummies(df['transaction_type'], prefix='type')
        for col in type_encoding.columns:
            df[f'{col}_amount'] = type_encoding[col] * df['amount']
        
        # Hour × day interaction
        df['hour_day_interaction'] = df['hour'] * df['day_of_week']
        
        # Amount × country risk (simplified)
        high_risk_countries = ['RU', 'CN', 'BR', 'NG', 'IN']
        df['is_high_risk_country'] = df['country'].isin(high_risk_countries).astype(int)
        df['amount_risk_country'] = df['amount'] * df['is_high_risk_country']
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        # One-hot encode transaction type
        transaction_type_dummies = pd.get_dummies(df['transaction_type'], prefix='txn_type')
        df = pd.concat([df, transaction_type_dummies], axis=1)
        
        # One-hot encode merchant category
        merchant_dummies = pd.get_dummies(df['merchant_category'], prefix='merchant')
        df = pd.concat([df, merchant_dummies], axis=1)
        
        # Label encode country (or use one-hot for smaller cardinality)
        country_dummies = pd.get_dummies(df['country'], prefix='country')
        df = pd.concat([df, country_dummies], axis=1)
        
        # Drop original categorical columns
        df = df.drop(['transaction_type', 'merchant_category', 'country'], axis=1)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw transaction dataframe
        
        Returns:
            DataFrame with engineered features
        """
        print("Starting feature engineering...")
        
        # Time features
        df = self.create_time_features(df)
        print("  ✓ Time features created")
        
        # Customer aggregates (needed before rolling windows)
        df = self.create_customer_aggregates(df)
        print("  ✓ Customer aggregates created")
        
        # Rolling windows
        df = self.create_rolling_window_features(df)
        print("  ✓ Rolling window features created")
        
        # Velocity features
        df = self.create_velocity_features(df)
        print("  ✓ Velocity features created")
        
        # Risk heuristics
        df = self.create_risk_heuristics(df)
        print("  ✓ Risk heuristic features created")
        
        # Interaction features
        df = self.create_interaction_features(df)
        print("  ✓ Interaction features created")
        
        # Encode categoricals
        df = self.encode_categorical_features(df)
        print("  ✓ Categorical features encoded")
        
        # Store feature names (exclude metadata columns)
        exclude_cols = ['transaction_id', 'customer_id', 'timestamp', 'fraud']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Total features created: {len(self.feature_names)}")
        
        return df


def engineer_features_from_raw(
    input_path: Path,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load raw data and engineer features.
    
    Args:
        input_path: Path to raw transaction CSV
        output_path: Optional path to save feature matrix
    
    Returns:
        Feature matrix DataFrame
    """
    from src.data.preprocess import load_raw_data, clean_data
    
    # Load and clean
    df = load_raw_data(input_path)
    df = clean_data(df)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        print(f"\nSaved feature matrix to {output_path}")
        print(f"Shape: {df_features.shape}")
        print(f"Features: {len(engineer.feature_names)}")
        print(f"Fraud rate: {df_features['fraud'].mean():.2%}")
    
    return df_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Engineer features from raw transaction data")
    parser.add_argument("--input", type=str, default="data/raw/transactions_raw.csv",
                       help="Input raw data path")
    parser.add_argument("--output", type=str, default="data/processed/features.csv",
                       help="Output feature matrix path")
    
    args = parser.parse_args()
    
    engineer_features_from_raw(
        input_path=Path(args.input),
        output_path=Path(args.output)
    )

