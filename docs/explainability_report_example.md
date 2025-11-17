# Example Explainability Report

## Sample Audit Report

```
================================================================================
FRAUD DETECTION AUDIT REPORT
================================================================================
Generated: 2024-01-15 14:30:00

TRANSACTION DETAILS
--------------------------------------------------------------------------------
Transaction ID: TXN_00012345
Customer ID:    CUST_004567
Amount:         $5,234.56
Timestamp:      2024-01-15 14:30:00
Fraud Score:    0.8523
Alert Status:   FLAGGED
Threshold:     0.5000

================================================================================
TOP CONTRIBUTING FACTORS
--------------------------------------------------------------------------------
amount_vs_avg                    :    0.2345
is_burst                         :    0.1890
is_unusual_country               :    0.1567
txn_count_1H                     :    0.1234
amount_sum_1H                    :    0.0987
risk_score                       :    0.0876
time_since_last_txn              :    0.0654
is_night                         :    0.0432
customer_txn_frequency           :    0.0321
amount_max_24H                   :    0.0210

================================================================================
RECOMMENDATION
--------------------------------------------------------------------------------
This transaction has been FLAGGED for manual review.
Fraud probability (85.23%) exceeds threshold (50.00%).

Recommended Actions:
1. Review customer transaction history
2. Verify transaction details with customer
3. Check for unusual patterns in recent activity
4. Consider temporary account hold if risk is high

================================================================================
```

## SHAP Summary Interpretation

The SHAP summary plot shows:
- **Red dots**: High feature values
- **Blue dots**: Low feature values
- **Horizontal position**: Impact on prediction (left = lower fraud prob, right = higher)
- **Vertical position**: Feature importance

Key insights:
- `amount_vs_avg` has the highest impact - large amounts relative to customer average
- `is_burst` indicates rapid successive transactions
- `is_unusual_country` shows geographic anomaly

