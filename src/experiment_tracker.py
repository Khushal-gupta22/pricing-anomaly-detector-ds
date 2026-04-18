"""
Experiment Tracker
==================
Simple file-based experiment tracking for anomaly detection runs.
Logs hyperparameters, metrics, and metadata to a JSON Lines file.

This is a lightweight alternative to MLflow for personal projects.
Each experiment is a single JSON line, making it easy to load
into pandas for analysis.

Usage:
    tracker = ExperimentTracker()
    tracker.log_experiment(
        method='isolation_forest',
        params={'contamination': 0.10, 'n_estimators': 200},
        metrics={'precision': 0.85, 'recall': 0.42, 'f1': 0.56},
        notes='Increased contamination from 0.05'
    )
    
    # Load all experiments
    history = tracker.load_history()
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
from loguru import logger


class ExperimentTracker:
    """
    File-based experiment tracking using JSON Lines format.
    
    Each experiment record contains:
    - experiment_id: auto-incrementing ID
    - timestamp: when the experiment was run
    - method: detection method name
    - params: dict of hyperparameters
    - metrics: dict of evaluation metrics (precision, recall, F1, etc.)
    - data_info: dict describing the dataset used
    - notes: free-text notes
    - duration_seconds: how long the run took
    """
    
    def __init__(self, log_path: str = "data/experiments.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
    
    def log_experiment(
        self,
        method: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        data_info: Optional[Dict[str, Any]] = None,
        notes: str = "",
        duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Log a single experiment run.
        
        Args:
            method: Name of the detection method (e.g., 'isolation_forest', 'ensemble')
            params: Hyperparameters used (e.g., {'contamination': 0.10})
            metrics: Evaluation metrics (e.g., {'precision': 0.85, 'recall': 0.42})
            data_info: Dataset info (e.g., {'n_events': 100000, 'anomaly_rate': 0.12})
            notes: Free-text notes about this experiment
            duration_seconds: Runtime in seconds
            
        Returns:
            The logged experiment record as a dict.
        """
        experiment_id = self._next_id()
        
        record = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'params': params,
            'metrics': metrics,
            'data_info': data_info or {},
            'notes': notes,
            'duration_seconds': round(duration_seconds, 2),
        }
        
        # Append to JSON Lines file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
        
        logger.info(
            f"Logged experiment #{experiment_id}: {method} | "
            f"F1={metrics.get('f1', 'N/A')} | "
            f"{notes[:50] if notes else 'no notes'}"
        )
        
        return record
    
    def log_pipeline_run(
        self,
        pipeline_results: Dict[str, Any],
        config: Dict[str, Any],
        notes: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Log all method results from a pipeline run.
        
        Args:
            pipeline_results: The results dict from AnomalyDetectionPipeline.run()
            config: The configuration dict used
            notes: Free-text notes
            
        Returns:
            List of logged experiment records.
        """
        records = []
        evaluation = pipeline_results.get('evaluation', {})
        data_info = {
            'n_events': pipeline_results.get('total_events', 0),
            'n_anomalies_detected': pipeline_results.get('total_anomalies_detected', 0),
        }
        duration = pipeline_results.get('pipeline_duration_seconds', 0)
        
        # Log each method's metrics
        for method_name in ['ensemble', 'business_rules', 'combined']:
            if method_name in evaluation:
                method_metrics = evaluation[method_name]
                
                # Extract relevant params
                params = {}
                if method_name == 'ensemble':
                    ad_config = config.get('anomaly_detection', {})
                    params = {
                        'contamination': ad_config.get('isolation_forest', {}).get('contamination', 0.10),
                        'n_estimators': ad_config.get('isolation_forest', {}).get('n_estimators', 200),
                        'zscore_threshold': ad_config.get('zscore_threshold', 2.5),
                        'ensemble_threshold': ad_config.get('ensemble_threshold', 0.4),
                    }
                elif method_name == 'business_rules':
                    br_config = config.get('business_rules', {})
                    params = {
                        'max_surge': br_config.get('max_surge_multiplier', 5.0),
                        'min_price_floor': br_config.get('min_price_floor', 1.0),
                        'max_price_per_mile': br_config.get('max_price_per_mile', 30.0),
                    }
                elif method_name == 'combined':
                    params = {'method': 'ensemble + business_rules'}
                
                record = self.log_experiment(
                    method=method_name,
                    params=params,
                    metrics=method_metrics,
                    data_info=data_info,
                    notes=notes,
                    duration_seconds=duration,
                )
                records.append(record)
        
        return records
    
    def load_history(self) -> pd.DataFrame:
        """
        Load all experiment records into a DataFrame.
        
        Flattens nested dicts (params.*, metrics.*) into columns
        for easy analysis.
        """
        if not os.path.exists(self.log_path):
            logger.warning(f"No experiment log at {self.log_path}")
            return pd.DataFrame()
        
        records = []
        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not records:
            return pd.DataFrame()
        
        # Flatten nested dicts
        flat_records = []
        for r in records:
            flat = {
                'experiment_id': r['experiment_id'],
                'timestamp': r['timestamp'],
                'method': r['method'],
                'notes': r.get('notes', ''),
                'duration_seconds': r.get('duration_seconds', 0),
            }
            # Flatten params
            for k, v in r.get('params', {}).items():
                flat[f'param_{k}'] = v
            # Flatten metrics
            for k, v in r.get('metrics', {}).items():
                flat[f'metric_{k}'] = v
            # Flatten data_info
            for k, v in r.get('data_info', {}).items():
                flat[f'data_{k}'] = v
            flat_records.append(flat)
        
        df = pd.DataFrame(flat_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    def get_best_by_metric(
        self, metric: str = 'f1', method: Optional[str] = None
    ) -> Optional[Dict]:
        """Get the experiment with the best value for a given metric."""
        history = self.load_history()
        if history.empty:
            return None
        
        metric_col = f'metric_{metric}'
        if metric_col not in history.columns:
            return None
        
        if method:
            history = history[history['method'] == method]
        
        if history.empty:
            return None
        
        best_idx = history[metric_col].idxmax()
        return history.loc[best_idx].to_dict()
    
    def compare_experiments(
        self, experiment_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments side by side.
        If no IDs specified, compares the latest run of each method.
        """
        history = self.load_history()
        if history.empty:
            return pd.DataFrame()
        
        if experiment_ids:
            return history[history['experiment_id'].isin(experiment_ids)]
        
        # Latest run of each method
        return history.drop_duplicates(subset=['method'], keep='first')
    
    def _next_id(self) -> int:
        """Get the next experiment ID."""
        if not os.path.exists(self.log_path):
            return 1
        
        max_id = 0
        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    max_id = max(max_id, record.get('experiment_id', 0))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return max_id + 1
    
    def clear(self):
        """Clear the experiment log (for testing)."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
            logger.info(f"Cleared experiment log at {self.log_path}")
