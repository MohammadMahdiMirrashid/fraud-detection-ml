# Feature Dictionary

## Transaction-Level Features

### Time Features
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `day_of_month`: Day of month (1-31)
- `month`: Month (1-12)
- `is_weekend`: Binary indicator (1 if Saturday/Sunday)
- `is_night`: Binary indicator (1 if 10 PM - 6 AM)

### Amount Features
- `amount`: Transaction amount
- `amount_vs_avg`: Amount relative to customer average
- `is_large_amount`: Binary (1 if amount > 3x customer average)

### Rolling Window Features
- `txn_count_1H`: Transaction count in last 1 hour
- `txn_count_6H`: Transaction count in last 6 hours
- `txn_count_24H`: Transaction count in last 24 hours
- `txn_count_7D`: Transaction count in last 7 days
- `amount_sum_1H`: Sum of amounts in last 1 hour
- `amount_mean_1H`: Mean amount in last 1 hour
- `amount_max_1H`: Max amount in last 1 hour
- (Similar for 6H, 24H, 7D windows)

## Customer-Level Features

### Historical Aggregates
- `customer_txn_count`: Total transactions for customer
- `customer_total_amount`: Cumulative transaction amount
- `customer_avg_amount`: Average transaction amount
- `days_since_first_txn`: Days since customer's first transaction
- `customer_txn_frequency`: Transactions per day

### Velocity Features
- `time_since_last_txn`: Hours since last transaction
- `amount_velocity`: Change in amount from previous transaction
- `is_burst`: Binary (1 if transaction within 1 hour of previous)
- `burst_size`: Number of consecutive transactions within 1 hour

## Risk Heuristic Features

- `is_unusual_country`: Binary (1 if country not in customer history)
- `risk_score`: Composite risk score (0-10 scale)

## Categorical Encodings

### Transaction Type (One-Hot)
- `txn_type_purchase`
- `txn_type_transfer`
- `txn_type_withdrawal`
- `txn_type_deposit`
- `txn_type_payment`

### Merchant Category (One-Hot)
- `merchant_retail`
- `merchant_groceries`
- `merchant_restaurant`
- `merchant_gas`
- `merchant_online`
- `merchant_utility`
- `merchant_other`

### Country (One-Hot)
- `country_US`
- `country_CA`
- `country_MX`
- (etc.)

## Interaction Features

- `type_*_amount`: Transaction type × amount interactions
- `hour_day_interaction`: Hour × day of week
- `amount_risk_country`: Amount × high-risk country indicator

