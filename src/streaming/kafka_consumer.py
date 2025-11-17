"""
Kafka consumer for real transaction events (requires kafka-python).
"""

try:
    from kafka import KafkaConsumer
    import json
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("kafka-python not installed. Using mock implementation.")


if KAFKA_AVAILABLE:
    class TransactionConsumer:
        """Kafka consumer for transaction events."""
        
        def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'transactions'):
            """
            Initialize Kafka consumer.
            
            Args:
                bootstrap_servers: Kafka broker addresses
                topic: Topic name
            """
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            self.topic = topic
        
        def __iter__(self):
            """Iterate over messages from Kafka."""
            for message in self.consumer:
                yield message.value
        
        def close(self):
            """Close consumer."""
            self.consumer.close()
else:
    # Fallback to mock
    from src.streaming.mock_stream import KafkaConsumer as TransactionConsumer

