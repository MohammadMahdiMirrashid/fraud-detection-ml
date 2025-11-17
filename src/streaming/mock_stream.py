"""
Mock event stream for real-time fraud detection demo.
"""

import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Iterator, Dict, Optional
import json


class EventStream:
    """Simulate a stream of transaction events."""
    
    def __init__(
        self,
        rate_per_second: float = 5.0,
        n_events: Optional[int] = None,
        fraud_rate: float = 0.005,
        random_state: int = 42
    ):
        """
        Initialize event stream.
        
        Args:
            rate_per_second: Events per second
            n_events: Total number of events (None = infinite)
            fraud_rate: Proportion of fraudulent events
            random_state: Random seed
        """
        self.rate_per_second = rate_per_second
        self.n_events = n_events
        self.fraud_rate = fraud_rate
        self.random_state = random_state
        self.event_count = 0
        
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Generate customer pool
        self.customer_ids = list(range(1, 10001))
        
    def _generate_event(self) -> Dict:
        """Generate a single transaction event."""
        event = {
            'transaction_id': f"TXN_{self.event_count:08d}",
            'customer_id': random.choice(self.customer_ids),
            'timestamp': datetime.now().isoformat(),
            'amount': np.random.lognormal(mean=3.5, sigma=1.2),
            'transaction_type': random.choice(['purchase', 'transfer', 'withdrawal', 'deposit', 'payment']),
            'merchant_category': random.choice(['retail', 'groceries', 'restaurant', 'gas', 'online', 'utility', 'other']),
            'country': random.choice(['US', 'CA', 'MX', 'UK', 'DE', 'FR']),
            'fraud': 0
        }
        
        # Inject fraud occasionally
        if random.random() < self.fraud_rate:
            event['fraud'] = 1
            # Make it suspicious
            event['amount'] = np.random.lognormal(mean=6, sigma=1.5)
            if random.random() < 0.3:
                event['country'] = random.choice(['RU', 'CN', 'BR', 'NG', 'IN'])
        
        event['amount'] = round(event['amount'], 2)
        self.event_count += 1
        
        return event
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over events."""
        interval = 1.0 / self.rate_per_second
        
        while True:
            if self.n_events is not None and self.event_count >= self.n_events:
                break
            
            event = self._generate_event()
            yield event
            
            time.sleep(interval)
    
    def get_batch(self, n: int) -> pd.DataFrame:
        """Get a batch of n events."""
        events = [self._generate_event() for _ in range(n)]
        return pd.DataFrame(events)


class KafkaProducer:
    """Mock Kafka producer for transaction events."""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'transactions'):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic name
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        print(f"Mock Kafka Producer initialized (topic: {topic})")
        print("Note: This is a mock implementation. Install kafka-python for real Kafka support.")
    
    def send(self, event: Dict):
        """Send event to Kafka topic."""
        # Mock implementation
        print(f"[Kafka] Sending event to topic '{self.topic}': {event.get('transaction_id', 'N/A')}")
    
    def flush(self):
        """Flush pending messages."""
        pass
    
    def close(self):
        """Close producer."""
        pass


class KafkaConsumer:
    """Mock Kafka consumer for transaction events."""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'transactions'):
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic name
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.stream = EventStream()
        print(f"Mock Kafka Consumer initialized (topic: {topic})")
        print("Note: This is a mock implementation. Install kafka-python for real Kafka support.")
    
    def __iter__(self):
        """Iterate over messages from Kafka."""
        for event in self.stream:
            yield event
    
    def close(self):
        """Close consumer."""
        pass

