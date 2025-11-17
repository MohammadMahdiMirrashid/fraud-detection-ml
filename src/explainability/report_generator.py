"""
Generate audit-style reports for fraud detection explanations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


def generate_audit_report(
    transaction_id: str,
    customer_id: str,
    amount: float,
    timestamp: str,
    fraud_probability: float,
    top_features: pd.DataFrame,
    shap_values: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate an audit-style report for a flagged transaction.
    
    Args:
        transaction_id: Transaction ID
        customer_id: Customer ID
        amount: Transaction amount
        timestamp: Transaction timestamp
        fraud_probability: Predicted fraud probability
        top_features: DataFrame with top contributing features
        shap_values: Optional SHAP values for detailed explanation
        threshold: Alert threshold
        output_path: Optional path to save report
    
    Returns:
        Report as string
    """
    is_flagged = fraud_probability >= threshold
    
    report = f"""
{'='*80}
FRAUD DETECTION AUDIT REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRANSACTION DETAILS
{'-'*80}
Transaction ID: {transaction_id}
Customer ID:    {customer_id}
Amount:         ${amount:,.2f}
Timestamp:      {timestamp}
Fraud Score:    {fraud_probability:.4f}
Alert Status:   {'FLAGGED' if is_flagged else 'CLEAR'}
Threshold:      {threshold:.4f}

{'='*80}
TOP CONTRIBUTING FACTORS
{'-'*80}
"""
    
    for idx, row in top_features.head(10).iterrows():
        report += f"{row['feature']:30s} : {row['importance']:10.4f}\n"
    
    report += f"""
{'='*80}
RECOMMENDATION
{'-'*80}
"""
    
    if is_flagged:
        report += f"""
This transaction has been FLAGGED for manual review.
Fraud probability ({fraud_probability:.2%}) exceeds threshold ({threshold:.2%}).

Recommended Actions:
1. Review customer transaction history
2. Verify transaction details with customer
3. Check for unusual patterns in recent activity
4. Consider temporary account hold if risk is high
"""
    else:
        report += f"""
This transaction appears LEGITIMATE.
Fraud probability ({fraud_probability:.2%}) is below threshold ({threshold:.2%}).

No immediate action required.
"""
    
    report += f"\n{'='*80}\n"
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Audit report saved to {output_path}")
    
    return report


def generate_batch_report(
    flagged_transactions: pd.DataFrame,
    output_path: Path
):
    """
    Generate batch audit report for multiple flagged transactions.
    
    Args:
        flagged_transactions: DataFrame with flagged transactions and explanations
        output_path: Path to save report
    """
    report = f"""
{'='*80}
BATCH FRAUD DETECTION AUDIT REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Flagged Transactions: {len(flagged_transactions)}

{'='*80}
SUMMARY STATISTICS
{'-'*80}
Average Fraud Score: {flagged_transactions['fraud_probability'].mean():.4f}
Median Fraud Score:  {flagged_transactions['fraud_probability'].median():.4f}
Min Fraud Score:     {flagged_transactions['fraud_probability'].min():.4f}
Max Fraud Score:     {flagged_transactions['fraud_probability'].max():.4f}

Total Amount Flagged: ${flagged_transactions['amount'].sum():,.2f}
Average Amount:       ${flagged_transactions['amount'].mean():,.2f}

{'='*80}
FLAGGED TRANSACTIONS
{'-'*80}
"""
    
    for idx, row in flagged_transactions.iterrows():
        report += f"""
Transaction ID: {row.get('transaction_id', 'N/A')}
Customer ID:    {row.get('customer_id', 'N/A')}
Amount:         ${row.get('amount', 0):,.2f}
Fraud Score:    {row.get('fraud_probability', 0):.4f}
Timestamp:      {row.get('timestamp', 'N/A')}
{'-'*80}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Batch audit report saved to {output_path}")

