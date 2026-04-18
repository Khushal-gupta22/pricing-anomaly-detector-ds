"""
Pipeline Orchestrator
=====================
End-to-end pipeline that:
1. Generates or loads pricing data
2. Stores in database
3. Fits all detection models
4. Runs detection
5. Classifies and stores anomalies
6. Evaluates performance against ground truth
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Tuple
from loguru import logger
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import DynamicPricingDataGenerator
from src.database import PricingDatabase
from src.anomaly_detector import EnsembleAnomalyDetector
from src.business_rules import BusinessRuleValidator
from src.time_series import TimeSeriesForecaster
from src.alerting import AlertClassifier, prepare_alerts_for_db
from src.experiment_tracker import ExperimentTracker


class AnomalyDetectionPipeline:
    """
    Production-grade anomaly detection pipeline.
    
    Stages:
    1. DATA INGESTION: Generate synthetic data or load from CSV/DB
    2. PREPROCESSING: Clean, validate, feature engineering
    3. MODEL FITTING: Train Isolation Forest, compute baselines, fit Prophet
    4. DETECTION: Run all detectors on the data
    5. ALERTING: Classify severities, generate descriptions
    6. STORAGE: Persist results to database
    7. EVALUATION: Compare against ground truth (if available)
    
    Design decisions:
    - Pipeline is idempotent: running it twice produces the same results
    - Each stage logs progress for debugging
    - Graceful degradation: if Prophet fails, other methods still work
    - Results are stored in SQLite for dashboard consumption
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.db = PricingDatabase(
            self.config.get('database', {}).get('path', 'data/pricing_anomalies.db')
        )
        self.ensemble = EnsembleAnomalyDetector(self.config)
        self.rule_validator = BusinessRuleValidator(self.config)
        self.forecaster = TimeSeriesForecaster(self.config)
        self.alert_classifier = AlertClassifier()
        self.experiment_tracker = ExperimentTracker(
            os.path.join(PROJECT_ROOT, "data", "experiments.jsonl")
        )
        
        self.pricing_df: Optional[pd.DataFrame] = None
        self.anomaly_ground_truth: Optional[pd.DataFrame] = None
        self.results: Dict = {}
    
    def _load_config(self, config_path: str) -> dict:
        for path in [config_path, os.path.join(PROJECT_ROOT, config_path)]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        logger.warning("Config not found, using defaults")
        return {}
    
    def run(
        self,
        generate_data: bool = True,
        use_prophet: bool = True,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        Execute the full anomaly detection pipeline.
        
        Args:
            generate_data: If True, generate synthetic data. If False, load from DB.
            use_prophet: If True, include Prophet time series forecasting.
            sample_size: If set, subsample data for faster execution.
        
        Returns:
            Dictionary with pipeline results and metrics.
        """
        pipeline_start = time.time()
        run_id = self.db.start_detection_run(
            methods=['isolation_forest', 'statistical', 'contextual', 
                     'business_rules', 'prophet']
        )
        
        try:
            # =============================================
            # STAGE 1: Data Ingestion
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 1: Data Ingestion")
            logger.info("=" * 60)
            
            if generate_data:
                generator = DynamicPricingDataGenerator(
                    os.path.join(PROJECT_ROOT, "config/settings.yaml")
                )
                self.pricing_df, self.anomaly_ground_truth = generator.generate()
                
                # Save to CSV
                data_dir = os.path.join(PROJECT_ROOT, "data")
                os.makedirs(data_dir, exist_ok=True)
                self.pricing_df.to_csv(
                    os.path.join(data_dir, "pricing_events.csv"), index=False
                )
                self.anomaly_ground_truth.to_csv(
                    os.path.join(data_dir, "anomaly_ground_truth.csv"), index=False
                )
            else:
                data_dir = os.path.join(PROJECT_ROOT, "data")
                self.pricing_df = pd.read_csv(
                    os.path.join(data_dir, "pricing_events.csv")
                )
                gt_path = os.path.join(data_dir, "anomaly_ground_truth.csv")
                if os.path.exists(gt_path):
                    self.anomaly_ground_truth = pd.read_csv(gt_path)
            
            # Load into database
            self.db.load_pricing_data(self.pricing_df)
            if self.anomaly_ground_truth is not None and not self.anomaly_ground_truth.empty:
                self.db.load_ground_truth(self.anomaly_ground_truth)
            
            logger.info(f"  Loaded {len(self.pricing_df):,} pricing events")
            
            # Optional subsampling
            if sample_size and len(self.pricing_df) > sample_size:
                logger.info(f"  Subsampling to {sample_size:,} events")
                self.pricing_df = self.pricing_df.sample(
                    n=sample_size, random_state=42
                ).reset_index(drop=True)
            
            # =============================================
            # STAGE 2: Preprocessing
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 2: Preprocessing")
            logger.info("=" * 60)
            
            self.pricing_df['timestamp'] = pd.to_datetime(self.pricing_df['timestamp'])
            
            # Basic data quality checks
            null_counts = self.pricing_df.isnull().sum()
            if null_counts.any():
                logger.warning(f"  Null values found: {null_counts[null_counts > 0].to_dict()}")
            
            n_dupes = self.pricing_df.duplicated(subset=['event_id']).sum()
            if n_dupes > 0:
                logger.warning(f"  {n_dupes} duplicate event_ids found")
            
            logger.info(
                f"  Date range: {self.pricing_df['timestamp'].min()} to "
                f"{self.pricing_df['timestamp'].max()}"
            )
            logger.info(f"  Categories: {self.pricing_df['category'].unique().tolist()}")
            logger.info(
                f"  Price range: ${self.pricing_df['final_price'].min():.2f} to "
                f"${self.pricing_df['final_price'].max():.2f}"
            )
            
            # =============================================
            # STAGE 3: Model Fitting
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 3: Model Fitting")
            logger.info("=" * 60)
            
            # Use first 80% for training, remaining 20% for evaluation.
            # Detection runs on the full dataset (as it would in production),
            # but metrics are computed ONLY on the held-out test set to
            # avoid data leakage / inflated scores.
            train_size = int(len(self.pricing_df) * 0.8)
            train_df = self.pricing_df.iloc[:train_size]
            self._test_indices = self.pricing_df.index[train_size:]
            
            logger.info(f"  Training on {len(train_df):,} events (80%)")
            
            fit_start = time.time()
            self.ensemble.fit(train_df)
            logger.info(f"  Ensemble fitted in {time.time() - fit_start:.1f}s")
            
            if use_prophet:
                ts_start = time.time()
                self.forecaster.fit(train_df)
                logger.info(f"  Prophet fitted in {time.time() - ts_start:.1f}s")
            
            # =============================================
            # STAGE 4: Detection
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 4: Anomaly Detection")
            logger.info("=" * 60)
            
            detect_start = time.time()
            
            # Run ensemble (Isolation Forest + Statistical + Contextual)
            ensemble_results = self.ensemble.predict(self.pricing_df)
            logger.info(
                f"  Ensemble: {ensemble_results['is_anomaly'].sum():,} anomalies "
                f"in {time.time() - detect_start:.1f}s"
            )
            
            # Run business rules
            rule_start = time.time()
            rule_results = self.rule_validator.validate(self.pricing_df)
            logger.info(
                f"  Business rules: {rule_results['rule_anomaly'].sum():,} violations "
                f"in {time.time() - rule_start:.1f}s"
            )
            
            # Run time series
            ts_results = None
            if use_prophet:
                ts_start = time.time()
                ts_results = self.forecaster.predict(self.pricing_df)
                logger.info(
                    f"  Time series: {ts_results['ts_anomaly'].sum():,} anomalies "
                    f"in {time.time() - ts_start:.1f}s"
                )
            
            # =============================================
            # STAGE 5: Alert Classification
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 5: Alert Classification")
            logger.info("=" * 60)
            
            enriched_anomalies = self.alert_classifier.classify_and_enrich(
                self.pricing_df, ensemble_results, rule_results, ts_results
            )
            
            if not enriched_anomalies.empty:
                logger.info(f"  Total alerts: {len(enriched_anomalies):,}")
                severity_counts = enriched_anomalies['severity'].value_counts()
                for sev, count in severity_counts.items():
                    logger.info(f"    {sev}: {count}")
            
            # =============================================
            # STAGE 6: Storage
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 6: Persisting Results")
            logger.info("=" * 60)
            
            self.db.clear_detected_anomalies()
            
            if not enriched_anomalies.empty:
                db_records = prepare_alerts_for_db(enriched_anomalies)
                self.db.save_detected_anomalies(db_records)
            
            # =============================================
            # STAGE 7: Evaluation
            # =============================================
            logger.info("=" * 60)
            logger.info("STAGE 7: Evaluation")
            logger.info("=" * 60)
            
            eval_metrics = self._evaluate(ensemble_results, rule_results, run_id)
            
            # =============================================
            # Summary
            # =============================================
            total_time = time.time() - pipeline_start
            
            self.db.complete_detection_run(
                run_id=run_id,
                events_processed=len(self.pricing_df),
                anomalies_detected=len(enriched_anomalies),
                status="completed"
            )
            
            self.results = {
                'run_id': run_id,
                'total_events': len(self.pricing_df),
                'total_anomalies_detected': len(enriched_anomalies),
                'ensemble_anomalies': int(ensemble_results['is_anomaly'].sum()),
                'rule_violations': int(rule_results['rule_anomaly'].sum()),
                'ts_anomalies': int(ts_results['ts_anomaly'].sum()) if ts_results is not None else 0,
                'severity_distribution': enriched_anomalies['severity'].value_counts().to_dict() if not enriched_anomalies.empty else {},
                'evaluation': eval_metrics,
                'pipeline_duration_seconds': round(total_time, 2),
            }
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info(f"  Total time: {total_time:.1f}s")
            logger.info(f"  Events processed: {len(self.pricing_df):,}")
            logger.info(f"  Anomalies detected: {len(enriched_anomalies):,}")
            if eval_metrics:
                logger.info(f"  Precision: {eval_metrics.get('precision', 'N/A')}")
                logger.info(f"  Recall: {eval_metrics.get('recall', 'N/A')}")
                logger.info(f"  F1 Score: {eval_metrics.get('f1', 'N/A')}")
            logger.info("=" * 60)
            
            # Log to experiment tracker
            if eval_metrics:
                try:
                    self.experiment_tracker.log_pipeline_run(
                        pipeline_results=self.results,
                        config=self.config,
                        notes=f"Pipeline run: {len(self.pricing_df)} events, "
                              f"prophet={'yes' if use_prophet else 'no'}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log experiment: {e}")
            
            return self.results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.db.complete_detection_run(
                run_id=run_id,
                events_processed=0,
                anomalies_detected=0,
                status="failed",
                error=str(e)
            )
            raise
    
    def _evaluate(
        self,
        ensemble_results: pd.DataFrame,
        rule_results: pd.DataFrame,
        run_id: int
    ) -> Dict:
        """
        Evaluate detection performance against ground truth.
        
        Computes precision, recall, and F1 for:
        - Each individual method
        - The ensemble
        - Business rules
        """
        if self.pricing_df is None or 'is_anomaly' not in self.pricing_df.columns:
            logger.warning("No ground truth available for evaluation")
            return {}
        
        # Evaluate ONLY on held-out test set (20%) to avoid data leakage.
        # The model was trained on the first 80%, so metrics on training
        # data would be artificially inflated.
        test_idx = self._test_indices if hasattr(self, '_test_indices') else self.pricing_df.index
        ground_truth = self.pricing_df.loc[test_idx, 'is_anomaly'].astype(bool)
        
        if ground_truth.sum() == 0:
            logger.warning("No anomalies in ground truth")
            return {}
        
        metrics = {}
        
        # Ensemble evaluation (test set only)
        predicted = ensemble_results.loc[test_idx, 'is_anomaly'].astype(bool)
        p, r, f1 = self._precision_recall_f1(ground_truth, predicted)
        metrics['ensemble'] = {'precision': p, 'recall': r, 'f1': f1}
        self.db.save_model_performance(run_id, 'ensemble', p, r, f1)
        
        # Business rules evaluation (test set only)
        rule_predicted = rule_results.loc[test_idx, 'rule_anomaly'].astype(bool)
        p, r, f1 = self._precision_recall_f1(ground_truth, rule_predicted)
        metrics['business_rules'] = {'precision': p, 'recall': r, 'f1': f1}
        self.db.save_model_performance(run_id, 'business_rules', p, r, f1)
        
        # Combined (any method, test set only)
        combined = predicted | rule_predicted
        p, r, f1 = self._precision_recall_f1(ground_truth, combined)
        metrics['combined'] = {'precision': p, 'recall': r, 'f1': f1}
        metrics['precision'] = round(p, 4)
        metrics['recall'] = round(r, 4)
        metrics['f1'] = round(f1, 4)
        self.db.save_model_performance(run_id, 'combined', p, r, f1)
        
        # Log detailed metrics
        logger.info(f"  Ground truth anomalies: {ground_truth.sum():,}")
        logger.info(f"  Ensemble - P: {metrics['ensemble']['precision']:.3f}, "
                    f"R: {metrics['ensemble']['recall']:.3f}, "
                    f"F1: {metrics['ensemble']['f1']:.3f}")
        logger.info(f"  Rules   - P: {metrics['business_rules']['precision']:.3f}, "
                    f"R: {metrics['business_rules']['recall']:.3f}, "
                    f"F1: {metrics['business_rules']['f1']:.3f}")
        logger.info(f"  Combined- P: {metrics['combined']['precision']:.3f}, "
                    f"R: {metrics['combined']['recall']:.3f}, "
                    f"F1: {metrics['combined']['f1']:.3f}")
        
        return metrics
    
    @staticmethod
    def _precision_recall_f1(
        y_true: pd.Series, y_pred: pd.Series
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, F1 without sklearn dependency."""
        tp = ((y_true == True) & (y_pred == True)).sum()
        fp = ((y_true == False) & (y_pred == True)).sum()
        fn = ((y_true == True) & (y_pred == False)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0.0)
        
        return round(precision, 4), round(recall, 4), round(f1, 4)


def main():
    """Run the full pipeline from command line."""
    import argparse
    
    print("[pipeline] Parsing arguments...", flush=True)
    
    parser = argparse.ArgumentParser(description="Dynamic Pricing Anomaly Detection Pipeline")
    parser.add_argument('--no-generate', action='store_true',
                        help='Load existing data instead of generating')
    parser.add_argument('--no-prophet', action='store_true',
                        help='Skip Prophet time series (faster)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Subsample N events for faster testing')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    print(f"[pipeline] Config: generate={not args.no_generate}, "
          f"prophet={not args.no_prophet}, sample={args.sample}", flush=True)
    
    pipeline = AnomalyDetectionPipeline(config_path=args.config)
    
    print("[pipeline] Running pipeline...", flush=True)
    results = pipeline.run(
        generate_data=not args.no_generate,
        use_prophet=not args.no_prophet,
        sample_size=args.sample
    )
    
    print("\n" + "=" * 60, flush=True)
    print("PIPELINE RESULTS", flush=True)
    print("=" * 60, flush=True)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:", flush=True)
            for k, v in value.items():
                print(f"  {k}: {v}", flush=True)
        else:
            print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
