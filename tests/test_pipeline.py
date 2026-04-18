"""
Test Suite for Dynamic Pricing Anomaly Detector
================================================
Tests the full pipeline end-to-end plus individual component tests.

Run: python -m pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────
# Data Generator Tests
# ─────────────────────────────────────────────────────────
class TestDataGenerator:
    """Tests for the synthetic data generator."""
    
    def test_generates_data(self):
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        # Generate just 3 days for speed
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=3)
        pricing_df, anomaly_df = gen.generate(start_date=start, end_date=end, seed=42)
        
        assert len(pricing_df) > 100, "Should generate substantial data"
        assert len(anomaly_df) > 0, "Should inject at least some anomalies"
    
    def test_required_columns(self):
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=2)
        pricing_df, _ = gen.generate(start_date=start, end_date=end, seed=42)
        
        required = [
            'timestamp', 'category', 'base_price', 'surge_multiplier',
            'final_price', 'demand_level', 'supply_level',
            'distance_miles', 'duration_minutes', 'region',
            'is_anomaly', 'event_id'
        ]
        for col in required:
            assert col in pricing_df.columns, f"Missing column: {col}"
    
    def test_all_categories_present(self):
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=5)
        pricing_df, _ = gen.generate(start_date=start, end_date=end, seed=42)
        
        expected_cats = {'ride_standard', 'ride_premium', 'ride_pool',
                        'delivery_food', 'delivery_grocery'}
        actual_cats = set(pricing_df['category'].unique())
        assert expected_cats == actual_cats
    
    def test_anomaly_injection(self):
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=10)
        pricing_df, anomaly_df = gen.generate(start_date=start, end_date=end, seed=42)
        
        # Should inject ~3% anomalies (with clustering, could be more)
        anomaly_rate = pricing_df['is_anomaly'].mean()
        assert 0.01 < anomaly_rate < 0.15, f"Anomaly rate {anomaly_rate:.2%} out of expected range"
    
    def test_clustered_anomalies(self):
        """Verify that anomaly clustering produces consecutive anomaly runs."""
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=15)
        pricing_df, _ = gen.generate(start_date=start, end_date=end, seed=42)
        
        # Check for runs of consecutive anomalies (evidence of clustering)
        anom_flags = pricing_df['is_anomaly'].values
        max_run = 0
        current_run = 0
        for flag in anom_flags:
            if flag:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        assert max_run >= 3, f"Expected clustered anomalies but max run was {max_run}"
    
    def test_prices_mostly_positive(self):
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=5)
        pricing_df, _ = gen.generate(start_date=start, end_date=end, seed=42)
        
        # Non-anomalous prices should all be positive
        normal = pricing_df[pricing_df['is_anomaly'] == False]
        assert (normal['final_price'] > 0).all(), "Non-anomalous prices should be positive"
    
    def test_deterministic_with_seed(self):
        from src.data_generator import DynamicPricingDataGenerator
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=2)
        
        gen1 = DynamicPricingDataGenerator()
        df1, _ = gen1.generate(start_date=start, end_date=end, seed=99)
        
        gen2 = DynamicPricingDataGenerator()
        df2, _ = gen2.generate(start_date=start, end_date=end, seed=99)
        
        assert len(df1) == len(df2)
        pd.testing.assert_frame_equal(df1, df2)


# ─────────────────────────────────────────────────────────
# Database Tests
# ─────────────────────────────────────────────────────────
class TestDatabase:
    """Tests for the SQLite database layer."""
    
    def test_init_creates_db(self, tmp_path):
        from src.database import PricingDatabase
        db_path = str(tmp_path / "test.db")
        db = PricingDatabase(db_path)
        assert os.path.exists(db_path)
    
    def test_load_and_query_events(self, tmp_path):
        from src.database import PricingDatabase
        db = PricingDatabase(str(tmp_path / "test.db"))
        
        df = pd.DataFrame({
            'event_id': [1, 2, 3],
            'timestamp': ['2024-06-01 10:00', '2024-06-01 11:00', '2024-06-01 12:00'],
            'category': ['ride_standard', 'ride_premium', 'ride_pool'],
            'base_price': [12.5, 25.0, 8.0],
            'surge_multiplier': [1.2, 1.5, 1.0],
            'final_price': [18.0, 42.0, 10.0],
            'demand_level': [0.6, 0.8, 0.3],
            'supply_level': [0.4, 0.3, 0.7],
            'distance_miles': [5.0, 8.0, 3.0],
            'duration_minutes': [15.0, 25.0, 10.0],
            'region': ['downtown', 'airport', 'suburbs'],
            'is_anomaly': [False, False, True],
            'anomaly_type': [None, None, 'flash_crash'],
        })
        
        db.load_pricing_data(df)
        result = db.get_pricing_events()
        assert len(result) == 3
    
    def test_detection_run_lifecycle(self, tmp_path):
        from src.database import PricingDatabase
        db = PricingDatabase(str(tmp_path / "test.db"))
        
        run_id = db.start_detection_run(methods=['test_method'])
        assert run_id is not None
        
        db.complete_detection_run(run_id, events_processed=100, anomalies_detected=5)
        # Should not raise


# ─────────────────────────────────────────────────────────
# Anomaly Detector Tests
# ─────────────────────────────────────────────────────────
class TestAnomalyDetector:
    """Tests for the multi-method anomaly detection engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate a small sample dataset for testing."""
        from src.data_generator import DynamicPricingDataGenerator
        gen = DynamicPricingDataGenerator()
        end = datetime(2024, 6, 15, 12, 0)
        start = end - timedelta(days=5)
        pricing_df, _ = gen.generate(start_date=start, end_date=end, seed=42)
        return pricing_df
    
    def test_isolation_forest_fit_predict(self, sample_data):
        from src.anomaly_detector import IsolationForestDetector
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(sample_data)
        results = detector.predict(sample_data)
        
        assert 'if_anomaly' in results.columns
        assert 'if_score' in results.columns
        assert len(results) == len(sample_data)
        assert results['if_score'].between(0, 1).all()
    
    def test_statistical_detector(self, sample_data):
        from src.anomaly_detector import StatisticalDetector
        detector = StatisticalDetector(zscore_threshold=3.0)
        detector.fit(sample_data)
        results = detector.predict(sample_data)
        
        assert 'stat_anomaly' in results.columns
        assert 'stat_score' in results.columns
        assert len(results) == len(sample_data)
    
    def test_contextual_detector(self, sample_data):
        from src.anomaly_detector import ContextualDetector
        detector = ContextualDetector()
        detector.fit(sample_data)
        results = detector.predict(sample_data)
        
        assert 'ctx_anomaly' in results.columns
        assert 'ctx_score' in results.columns
        assert len(results) == len(sample_data)
    
    def test_ensemble_detector(self, sample_data):
        from src.anomaly_detector import EnsembleAnomalyDetector
        detector = EnsembleAnomalyDetector()
        detector.fit(sample_data)
        results = detector.predict(sample_data)
        
        required = ['anomaly_score', 'is_anomaly', 'n_methods_flagged',
                    'severity', 'detection_methods']
        for col in required:
            assert col in results.columns, f"Missing: {col}"
        
        # Should detect some anomalies
        assert results['is_anomaly'].any(), "Should detect at least one anomaly"
    
    def test_ensemble_detects_extreme_outlier(self):
        """An extreme outlier should be caught by the ensemble."""
        from src.anomaly_detector import EnsembleAnomalyDetector
        
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-06-01', periods=n, freq='h'),
            'category': ['ride_standard'] * n,
            'base_price': [12.5] * n,
            'surge_multiplier': np.random.uniform(1.0, 2.0, n),
            'final_price': np.random.normal(20, 3, n),
            'demand_level': np.random.uniform(0.2, 0.8, n),
            'supply_level': np.random.uniform(0.3, 0.7, n),
            'distance_miles': np.random.uniform(2, 10, n),
            'duration_minutes': np.random.uniform(5, 30, n),
            'region': ['downtown'] * n,
        })
        
        # Inject a massive outlier
        df.loc[500, 'final_price'] = 500.0
        df.loc[500, 'surge_multiplier'] = 25.0
        
        detector = EnsembleAnomalyDetector()
        detector.fit(df)
        results = detector.predict(df)
        
        assert results.loc[500, 'is_anomaly'], "Should detect $500 outlier"
        assert results.loc[500, 'anomaly_score'] > 0.5, "Outlier should have high score"


# ─────────────────────────────────────────────────────────
# Business Rules Tests
# ─────────────────────────────────────────────────────────
class TestBusinessRules:
    """Tests for the business logic validator."""
    
    def _make_event(self, **overrides):
        """Create a single-row DataFrame representing one pricing event."""
        defaults = {
            'final_price': 20.0,
            'base_price': 12.5,
            'surge_multiplier': 1.5,
            'demand_level': 0.5,
            'supply_level': 0.5,
            'distance_miles': 5.0,
            'duration_minutes': 15.0,
            'category': 'ride_standard',
            'region': 'downtown',
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])
    
    def test_catches_negative_price(self):
        from src.business_rules import BusinessRuleValidator
        validator = BusinessRuleValidator()
        df = self._make_event(final_price=-5.0)
        result = validator.validate(df)
        
        assert result['rule_anomaly'].iloc[0] == True
        assert 'negative_price' in result['rule_violations'].iloc[0]
    
    def test_catches_zero_surge(self):
        from src.business_rules import BusinessRuleValidator
        validator = BusinessRuleValidator()
        df = self._make_event(surge_multiplier=-0.5)
        result = validator.validate(df)
        
        assert result['rule_anomaly'].iloc[0] == True
        assert 'zero_surge' in result['rule_violations'].iloc[0]
    
    def test_catches_surge_cap(self):
        from src.business_rules import BusinessRuleValidator
        validator = BusinessRuleValidator()
        df = self._make_event(surge_multiplier=20.0, distance_miles=5.0)
        result = validator.validate(df)
        
        assert result['rule_anomaly'].iloc[0] == True
        assert 'surge_cap_exceeded' in result['rule_violations'].iloc[0]
    
    def test_normal_event_passes(self):
        from src.business_rules import BusinessRuleValidator
        validator = BusinessRuleValidator()
        df = self._make_event()
        result = validator.validate(df)
        
        assert result['rule_anomaly'].iloc[0] == False
    
    def test_demand_surge_mismatch(self):
        from src.business_rules import BusinessRuleValidator
        validator = BusinessRuleValidator()
        df = self._make_event(demand_level=0.95, supply_level=0.1, surge_multiplier=1.0)
        result = validator.validate(df)
        
        assert result['rule_anomaly'].iloc[0] == True
        assert 'demand_surge_mismatch' in result['rule_violations'].iloc[0]


# ─────────────────────────────────────────────────────────
# Alerting Tests
# ─────────────────────────────────────────────────────────
class TestAlerting:
    """Tests for the alert classification system."""
    
    def test_classify_and_enrich(self):
        from src.alerting import AlertClassifier
        
        events = pd.DataFrame({
            'event_id': [1],
            'timestamp': [datetime(2024, 6, 1, 10, 0)],
            'category': ['ride_standard'],
            'final_price': [-5.0],
            'surge_multiplier': [-0.5],
            'region': ['downtown'],
            'base_price': [12.5],
            'anomaly_type': ['negative_price'],
        })
        
        ensemble = pd.DataFrame({
            'is_anomaly': [True],
            'anomaly_score': [0.95],
            'n_methods_flagged': [3],
        })
        
        rules = pd.DataFrame({
            'rule_anomaly': [True],
            'rule_score': [1.0],
            'rule_violations': ['negative_price,zero_surge'],
            'rule_severity': ['critical'],
        })
        
        classifier = AlertClassifier()
        result = classifier.classify_and_enrich(events, ensemble, rules)
        
        assert len(result) == 1
        assert 'severity' in result.columns
        assert 'description' in result.columns
        assert result['severity'].iloc[0] in ['critical', 'high']


# ─────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────
class TestPipelineIntegration:
    """End-to-end integration test for the full pipeline."""
    
    def test_full_pipeline_runs(self, tmp_path):
        """Run the complete pipeline on a small dataset and verify outputs."""
        from src.pipeline import AnomalyDetectionPipeline
        
        # Use a temp database
        os.environ['PIPELINE_DB_PATH'] = str(tmp_path / "test_pipeline.db")
        
        pipeline = AnomalyDetectionPipeline()
        # Override db path
        pipeline.db = __import__('src.database', fromlist=['PricingDatabase']).PricingDatabase(
            str(tmp_path / "test_pipeline.db")
        )
        
        results = pipeline.run(
            generate_data=True,
            use_prophet=False,  # Skip Prophet for speed
            sample_size=5000    # Small sample for fast test
        )
        
        assert results['total_events'] == 5000
        assert results['total_anomalies_detected'] > 0
        assert 'evaluation' in results
        assert results['pipeline_duration_seconds'] > 0
        
        # Verify database was populated
        events = pipeline.db.get_pricing_events(limit=10)
        assert len(events) > 0
        
        anomalies = pipeline.db.get_detected_anomalies(limit=10)
        assert len(anomalies) > 0


# ─────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────
class TestEdgeCases:
    """Tests for edge cases and robustness of the ML pipeline."""
    
    def _make_minimal_df(self, n=100, categories=None, include_anomaly=False):
        """Helper: create a minimal valid DataFrame."""
        rng = np.random.default_rng(42)
        cats = categories or ['ride_standard']
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-06-01', periods=n, freq='h'),
            'category': rng.choice(cats, n),
            'base_price': 12.5,
            'surge_multiplier': rng.uniform(1.0, 2.0, n),
            'final_price': rng.normal(20, 3, n),
            'demand_level': rng.uniform(0.2, 0.8, n),
            'supply_level': rng.uniform(0.3, 0.7, n),
            'distance_miles': rng.uniform(2, 10, n),
            'duration_minutes': rng.uniform(5, 30, n),
            'region': 'downtown',
        })
        if include_anomaly:
            df.loc[50, 'final_price'] = 500.0
            df.loc[50, 'surge_multiplier'] = 25.0
        return df
    
    def test_single_category_data(self):
        """Detectors should work with only one category."""
        from src.anomaly_detector import EnsembleAnomalyDetector
        df = self._make_minimal_df(n=200, categories=['ride_standard'])
        
        detector = EnsembleAnomalyDetector()
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        assert 'anomaly_score' in results.columns
    
    def test_all_same_prices(self):
        """Detectors should handle constant prices (zero variance)."""
        from src.anomaly_detector import StatisticalDetector
        df = self._make_minimal_df(n=100)
        df['final_price'] = 20.0  # all identical
        
        detector = StatisticalDetector(zscore_threshold=3.0)
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        # With zero variance, nothing should be flagged (or at least no crash)
        assert results['stat_score'].notna().all()
    
    def test_nan_in_prices(self):
        """Detectors should handle NaN values gracefully."""
        from src.anomaly_detector import IsolationForestDetector
        df = self._make_minimal_df(n=200)
        df.loc[10, 'final_price'] = np.nan
        df.loc[20, 'surge_multiplier'] = np.nan
        df.loc[30, 'distance_miles'] = np.nan
        
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        # Should not have NaN in output scores
        assert results['if_score'].notna().all()
    
    def test_negative_prices_in_data(self):
        """Detectors should handle negative prices (injected anomalies)."""
        from src.anomaly_detector import EnsembleAnomalyDetector
        df = self._make_minimal_df(n=200)
        df.loc[50, 'final_price'] = -10.0
        df.loc[51, 'final_price'] = -5.0
        
        detector = EnsembleAnomalyDetector()
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        # Negative prices should get high anomaly scores
        assert results.loc[50, 'anomaly_score'] > 0.3
    
    def test_empty_dataframe_business_rules(self):
        """Business rules should handle empty DataFrames."""
        from src.business_rules import BusinessRuleValidator
        validator = BusinessRuleValidator()
        df = pd.DataFrame({
            'final_price': pd.Series(dtype=float),
            'base_price': pd.Series(dtype=float),
            'surge_multiplier': pd.Series(dtype=float),
            'demand_level': pd.Series(dtype=float),
            'supply_level': pd.Series(dtype=float),
            'distance_miles': pd.Series(dtype=float),
            'duration_minutes': pd.Series(dtype=float),
            'category': pd.Series(dtype=str),
            'region': pd.Series(dtype=str),
        })
        result = validator.validate(df)
        assert len(result) == 0
    
    def test_very_small_dataset(self):
        """Ensemble should handle very small datasets (< 20 events)."""
        from src.anomaly_detector import EnsembleAnomalyDetector
        df = self._make_minimal_df(n=15)
        
        detector = EnsembleAnomalyDetector()
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
    
    def test_extreme_values(self):
        """Detectors should handle extreme but valid values."""
        from src.anomaly_detector import IsolationForestDetector
        df = self._make_minimal_df(n=200)
        # Add extreme but finite values
        df.loc[0, 'final_price'] = 1e6
        df.loc[1, 'final_price'] = 1e-6
        df.loc[2, 'surge_multiplier'] = 100.0
        df.loc[3, 'distance_miles'] = 500.0
        
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        assert np.isfinite(results['if_score']).all()
    
    def test_inf_values_handled(self):
        """Detectors should handle inf values without crashing."""
        from src.anomaly_detector import IsolationForestDetector
        df = self._make_minimal_df(n=200)
        df.loc[10, 'final_price'] = np.inf
        df.loc[11, 'distance_miles'] = 0.0  # causes inf in price_per_mile
        
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        assert np.isfinite(results['if_score']).all()
    
    def test_duplicate_timestamps(self):
        """Should handle duplicate timestamps (multiple events at same time)."""
        from src.anomaly_detector import EnsembleAnomalyDetector
        df = self._make_minimal_df(n=100)
        # Make many timestamps identical
        df['timestamp'] = pd.Timestamp('2024-06-01 10:00:00')
        
        detector = EnsembleAnomalyDetector()
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
    
    def test_contextual_detector_sparse_contexts(self):
        """Contextual detector should handle contexts with few data points."""
        from src.anomaly_detector import ContextualDetector
        df = self._make_minimal_df(n=50, categories=['ride_standard', 'ride_premium',
                                                       'ride_pool', 'delivery_food',
                                                       'delivery_grocery'])
        # With 50 events spread across 5 categories and many hour blocks,
        # some contexts will have <5 points
        
        detector = ContextualDetector()
        detector.fit(df)
        results = detector.predict(df)
        
        assert len(results) == len(df)
        assert results['ctx_score'].notna().all()
    
    def test_lof_detector_basic(self):
        """LOF detector should work on basic data."""
        from src.anomaly_detector import LocalOutlierFactorDetector
        df = self._make_minimal_df(n=200, include_anomaly=True)
        
        detector = LocalOutlierFactorDetector(n_neighbors=10, contamination=0.10)
        results = detector.fit_predict(df)
        
        assert 'lof_anomaly' in results.columns
        assert 'lof_score' in results.columns
        assert len(results) == len(df)
        assert results['lof_score'].between(0, 1).all()
    
    def test_ocsvm_detector_basic(self):
        """One-Class SVM detector should work on basic data."""
        from src.anomaly_detector import OneClassSVMDetector
        df = self._make_minimal_df(n=200, include_anomaly=True)
        
        detector = OneClassSVMDetector(nu=0.10, max_train_size=200)
        detector.fit(df)
        results = detector.predict(df)
        
        assert 'svm_anomaly' in results.columns
        assert 'svm_score' in results.columns
        assert len(results) == len(df)
    
    def test_dbscan_detector_basic(self):
        """DBSCAN detector should work on basic data."""
        from src.anomaly_detector import DBSCANDetector
        df = self._make_minimal_df(n=200, include_anomaly=True)
        
        detector = DBSCANDetector(min_samples=5, auto_tune_eps=True)
        results = detector.fit_predict(df)
        
        assert 'dbscan_anomaly' in results.columns
        assert 'dbscan_score' in results.columns
        assert 'dbscan_cluster' in results.columns
        assert len(results) == len(df)


# ─────────────────────────────────────────────────────────
# Experiment Tracker Tests
# ─────────────────────────────────────────────────────────
class TestExperimentTracker:
    """Tests for the JSON Lines experiment tracking system."""
    
    def test_log_and_load(self, tmp_path):
        from src.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(str(tmp_path / "exp.jsonl"))
        
        record = tracker.log_experiment(
            method='test_method',
            params={'contamination': 0.10, 'n_estimators': 200},
            metrics={'precision': 0.85, 'recall': 0.42, 'f1': 0.56},
            notes='test run'
        )
        
        assert record['experiment_id'] == 1
        assert record['method'] == 'test_method'
        
        history = tracker.load_history()
        assert len(history) == 1
        assert history.iloc[0]['method'] == 'test_method'
        assert history.iloc[0]['metric_f1'] == 0.56
    
    def test_auto_increment_id(self, tmp_path):
        from src.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(str(tmp_path / "exp.jsonl"))
        
        r1 = tracker.log_experiment(method='a', params={}, metrics={'f1': 0.5})
        r2 = tracker.log_experiment(method='b', params={}, metrics={'f1': 0.6})
        r3 = tracker.log_experiment(method='c', params={}, metrics={'f1': 0.7})
        
        assert r1['experiment_id'] == 1
        assert r2['experiment_id'] == 2
        assert r3['experiment_id'] == 3
    
    def test_get_best_by_metric(self, tmp_path):
        from src.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(str(tmp_path / "exp.jsonl"))
        
        tracker.log_experiment(method='a', params={}, metrics={'f1': 0.3})
        tracker.log_experiment(method='b', params={}, metrics={'f1': 0.9})
        tracker.log_experiment(method='c', params={}, metrics={'f1': 0.6})
        
        best = tracker.get_best_by_metric('f1')
        assert best is not None
        assert best['metric_f1'] == 0.9
    
    def test_clear(self, tmp_path):
        from src.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(str(tmp_path / "exp.jsonl"))
        
        tracker.log_experiment(method='a', params={}, metrics={'f1': 0.5})
        tracker.clear()
        
        history = tracker.load_history()
        assert len(history) == 0
    
    def test_empty_history(self, tmp_path):
        from src.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(str(tmp_path / "nonexistent.jsonl"))
        
        history = tracker.load_history()
        assert len(history) == 0


# ─────────────────────────────────────────────────────────
# Shared Feature Engineering Tests
# ─────────────────────────────────────────────────────────
class TestFeatureEngineering:
    """Tests for the shared engineer_anomaly_features() function."""
    
    def _make_df(self, n=100):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-06-01', periods=n, freq='h'),
            'category': ['ride_standard'] * n,
            'base_price': 12.5,
            'surge_multiplier': rng.uniform(1.0, 2.0, n),
            'final_price': rng.normal(20, 3, n),
            'demand_level': rng.uniform(0.2, 0.8, n),
            'supply_level': rng.uniform(0.3, 0.7, n),
            'distance_miles': rng.uniform(2, 10, n),
            'duration_minutes': rng.uniform(5, 30, n),
            'region': 'downtown',
        })
    
    def test_feature_count(self):
        from src.anomaly_detector import engineer_anomaly_features
        df = self._make_df()
        features, names = engineer_anomaly_features(df)
        
        assert len(names) == 15  # all 15 features
        assert len(features.columns) == 15
    
    def test_no_nans_or_infs(self):
        from src.anomaly_detector import engineer_anomaly_features
        df = self._make_df()
        df.loc[0, 'distance_miles'] = 0  # would cause inf in price_per_mile
        df.loc[1, 'final_price'] = np.nan
        
        features, _ = engineer_anomaly_features(df)
        assert features.isna().sum().sum() == 0
        assert np.isinf(features.values).sum() == 0
    
    def test_rush_hour_correctness(self):
        """Verify rush hour flag is correct after precedence fix."""
        from src.anomaly_detector import engineer_anomaly_features
        hours_to_test = {
            0: False, 3: False, 6: False,
            7: True, 8: True, 9: True,
            10: False, 12: False, 15: False, 16: False,
            17: True, 18: True, 19: True,
            20: False, 23: False,
        }
        for hour, expected in hours_to_test.items():
            df = pd.DataFrame({
                'timestamp': [pd.Timestamp(f'2024-06-01 {hour:02d}:00:00')],
                'base_price': [12.5], 'surge_multiplier': [1.5],
                'final_price': [20.0], 'demand_level': [0.5],
                'supply_level': [0.5], 'distance_miles': [5.0],
                'duration_minutes': [15.0],
            })
            features, _ = engineer_anomaly_features(df)
            actual = bool(features['is_rush_hour'].iloc[0])
            assert actual == expected, f"Hour {hour}: expected {expected}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
