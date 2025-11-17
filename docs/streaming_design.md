# Streaming Design

## Architecture

The real-time scoring system processes transaction events as they arrive:

```
Event Stream → Feature Extraction → Model Scoring → Alert Generation
```

## Components

### 1. Event Stream
- **Mock Stream**: Simulated events for testing (`mock_stream.py`)
- **Kafka Stream**: Production-ready Kafka integration (`kafka_producer.py`, `kafka_consumer.py`)

### 2. Feature Extraction
- Sliding window feature cache
- Real-time rolling aggregations
- Customer-level feature updates

### 3. Model Scoring
- Load trained model from disk
- Generate fraud probability
- Apply threshold for alerts

### 4. Alert Generation
- Flag high-risk transactions
- Generate audit reports
- Log for investigation

## Performance Considerations

- **Latency**: Target < 100ms per transaction
- **Throughput**: Handle 1000+ transactions/second
- **Feature Cache**: Maintain rolling windows efficiently
- **Model Loading**: Pre-load models to avoid cold starts

## Scalability

- Horizontal scaling: Multiple scoring workers
- Load balancing: Distribute events across workers
- State management: Shared feature cache (Redis/DB)
- Monitoring: Track latency, throughput, alert rates

## Example Usage

```python
from src.streaming.real_time_scoring import RealTimeScorer
from src.streaming.mock_stream import EventStream

# Initialize
scorer = RealTimeScorer(
    model_path="models/best_model.pkl",
    model_type="lightgbm",
    threshold=0.5
)

# Process stream
stream = EventStream(rate_per_second=10)
scorer.process_stream(stream, max_events=1000)
```

