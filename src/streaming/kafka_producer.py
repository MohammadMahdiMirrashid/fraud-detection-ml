"""
Kafka producer for real transaction events (requires kafka-python).
"""

try:
    from kafka import KafkaProducer
    import json
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("kafka-python not installed. Using mock implementation.")


if KAFKA_AVAILABLE:
    class TransactionProducer:
        """Kafka producer for transaction events."""
        
        def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'transactions'):
            """
            Initialize Kafka producer.
            
            Args:
                bootstrap_servers: Kafka broker addresses
                topic: Topic name
            """
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.topic = topic
        
        def send(self, event: dict):
            """Send event to Kafka topic."""
            self.producer.send(self.topic, event)
        
        def flush(self):
            """Flush pending messages."""
            self.producer.flush()
        
        def close(self):
            """Close producer."""
            self.producer.close()
else:
    # Fallback to mock
    from src.streaming.mock_stream import KafkaProducer as TransactionProducer

