"""
Real-time scoring pipeline for fraud detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import joblib
import time

from src.data.feature_engineering import FeatureEngineer
from src.streaming.mock_stream import EventStream


class RealTimeScorer:
    """Real-time fraud detection scorer."""
    
    def __init__(
        self,
        model_path: Path,
        model_type: str = 'lightgbm',
        threshold: float = 0.5,
        feature_cache_size: int = 1000
    ):
        """
        Initialize real-time scorer.
        
        Args:
            model_path: Path to trained model
            model_type: Model type ('lightgbm', 'xgboost', 'catboost', 'isolation_forest', 'autoencoder')
            threshold: Fraud detection threshold
            feature_cache_size: Size of feature cache for rolling windows
        """
        self.model_path = model_path
        self.model_type = model_type
        self.threshold = threshold
        self.feature_cache_size = feature_cache_size
        
        # Load model
        self.model = self._load_model()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Feature cache for rolling windows
        self.feature_cache = []
        
        print(f"RealTimeScorer initialized with {model_type} model")
        print(f"Threshold: {threshold:.4f}")
    
    def _load_model(self):
        """Load trained model."""
        if self.model_type == 'lightgbm':
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(self.model_path))
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(self.model_path))
        elif self.model_type == 'catboost':
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(self.model_path))
        elif self.model_type == 'isolation_forest':
            from src.models.isolation_forest import load_model
            model = load_model(self.model_path)
        elif self.model_type == 'autoencoder':
            from src.models.autoencoder import load_model
            model = load_model(self.model_path)
        else:
            model = joblib.load(self.model_path)
        
        return model
    
    def _extract_features(self, event: Dict) -> pd.DataFrame:
        """
        Extract features from a single transaction event.
        
        Args:
            event: Transaction event dictionary
        
        Returns:
            Feature vector as DataFrame
        """
        # Convert event to DataFrame
        df_event = pd.DataFrame([event])
        df_event['timestamp'] = pd.to_datetime(df_event['timestamp'])
        
        # Add to cache
        self.feature_cache.append(df_event)
        if len(self.feature_cache) > self.feature_cache_size:
            self.feature_cache.pop(0)
        
        # Combine cache for rolling features
        if len(self.feature_cache) > 1:
            df_all = pd.concat(self.feature_cache, ignore_index=True)
        else:
            df_all = df_event.copy()
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df_all)
        
        # Return features for most recent event
        return df_features.iloc[[-1]]
    
    def score(self, event: Dict) -> Dict:
        """
        Score a single transaction event.
        
        Args:
            event: Transaction event dictionary
        
        Returns:
            Dictionary with fraud probability and alert status
        """
        # Extract features
        features = self._extract_features(event)
        
        # Remove metadata columns
        metadata_cols = ['transaction_id', 'customer_id', 'timestamp', 'fraud']
        X = features.drop(columns=[col for col in metadata_cols if col in features.columns])
        
        # Predict
        if self.model_type == 'lightgbm':
            proba = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            proba = self.model.predict(dtest)[0]
        elif self.model_type == 'catboost':
            proba = self.model.predict_proba(X)[0, 1]
        elif self.model_type == 'isolation_forest':
            from src.models.isolation_forest import predict_fraud_probability
            proba = predict_fraud_probability(self.model, X)[0]
        elif self.model_type == 'autoencoder':
            from src.models.autoencoder import predict_fraud_probability
            proba = predict_fraud_probability(self.model, X)[0]
        else:
            proba = self.model.predict_proba(X)[0, 1]
        
        # Determine alert
        is_alert = proba >= self.threshold
        
        return {
            'transaction_id': event.get('transaction_id', 'N/A'),
            'customer_id': event.get('customer_id', 'N/A'),
            'amount': event.get('amount', 0),
            'fraud_probability': float(proba),
            'is_alert': is_alert,
            'timestamp': event.get('timestamp', 'N/A')
        }
    
    def process_stream(self, stream: EventStream, max_events: Optional[int] = None):
        """
        Process a stream of events.
        
        Args:
            stream: Event stream
            max_events: Maximum number of events to process
        """
        event_count = 0
        
        print("Starting real-time scoring...")
        print(f"Threshold: {self.threshold:.4f}\n")
        
        for event in stream:
            if max_events is not None and event_count >= max_events:
                break
            
            result = self.score(event)
            event_count += 1
            
            # Print alert
            if result['is_alert']:
                print(f"🚨 ALERT #{event_count}")
                print(f"   Transaction: {result['transaction_id']}")
                print(f"   Customer:    {result['customer_id']}")
                print(f"   Amount:      ${result['amount']:,.2f}")
                print(f"   Fraud Score: {result['fraud_probability']:.4f}")
                print(f"   Timestamp:   {result['timestamp']}")
                print()
            else:
                if event_count % 10 == 0:
                    print(f"Processed {event_count} events (no alerts)")
            
            time.sleep(0.1)  # Small delay for readability
        
        print(f"\nProcessing complete. Total events: {event_count}")

