"""
Data preprocessing utilities for transaction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data: handle missing values, outliers, etc.
    
    Args:
        df: Raw transaction dataframe
    
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    
    # Remove transactions with negative amounts (unless it's a deposit/refund)
    df = df[(df['amount'] > 0) | (df['transaction_type'].isin(['deposit', 'refund']))]
    
    # Handle missing values
    df['merchant_category'] = df['merchant_category'].fillna('other')
    df['country'] = df['country'].fillna('US')
    
    # Cap extreme outliers (99.9th percentile)
    amount_99_9 = df['amount'].quantile(0.999)
    df.loc[df['amount'] > amount_99_9, 'amount'] = amount_99_9
    
    return df


def create_interim_dataset(df: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create interim cleaned dataset.
    
    Args:
        df: Raw dataframe
        output_path: Optional path to save interim data
    
    Returns:
        Cleaned dataframe
    """
    df_clean = clean_data(df)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"Saved interim data to {output_path}")
    
    return df_clean

