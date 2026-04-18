"""
Anomaly Detection Engine
========================
Multi-method anomaly detection combining:
1. Isolation Forest (unsupervised ML)
2. Statistical methods (Z-score, IQR, rolling statistics)
3. Contextual detection (time-aware anomalies)

Design philosophy: No single method catches everything. We ensemble
multiple detectors and aggregate their signals. A pricing event flagged
by 3/3 methods is much more likely to be a true anomaly than 1/3.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ─────────────────────────────────────────────────────────
# Shared Feature Engineering
# ─────────────────────────────────────────────────────────
# Extracted to a single function to avoid duplication across
# the 4 detector classes (IF, LOF, SVM, DBSCAN).

def engineer_anomaly_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features for anomaly detection from raw pricing events.
    
    Returns a clean DataFrame (NaN/inf replaced with 0) and a list of
    feature names. Used by IsolationForest, LOF, One-Class SVM, and
    DBSCAN for fair, consistent comparison.
    
    Features (15 total):
    - Price ratios: surge, price-to-base, price-per-mile, price-per-minute
    - Demand-supply: gap, ratio, surge-demand ratio
    - Time: cyclical hour encoding, rush hour flag, late night, weekend
    - Absolute: final_price, log_price
    - Trip: implied speed (mph)
    """
    features = pd.DataFrame(index=df.index)
    
    # Price ratios (category-agnostic anomaly signals)
    features['surge_multiplier'] = df['surge_multiplier']
    features['price_to_base_ratio'] = df['final_price'] / df['base_price'].clip(lower=0.01)
    features['price_per_mile'] = df['final_price'] / df['distance_miles'].clip(lower=0.1)
    features['price_per_minute'] = df['final_price'] / df['duration_minutes'].clip(lower=1.0)
    
    # Demand-supply dynamics
    features['demand_supply_gap'] = df['demand_level'] - df['supply_level']
    features['demand_supply_ratio'] = (
        df['demand_level'] / df['supply_level'].clip(lower=0.01)
    )
    features['surge_demand_ratio'] = (
        df['surge_multiplier'] / df['demand_level'].clip(lower=0.01)
    )
    
    # Time features (cyclical encoding)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        hour = ts.dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_rush_hour'] = (((hour >= 7) & (hour <= 9)) | 
                                     ((hour >= 17) & (hour <= 19))).astype(int)
        features['is_late_night'] = ((hour >= 0) & (hour <= 5)).astype(int)
        dow = ts.dt.dayofweek
        features['is_weekend'] = (dow >= 5).astype(int)
    
    # Absolute price (still useful for catching extremes)
    features['final_price'] = df['final_price']
    features['log_price'] = np.log1p(df['final_price'].clip(lower=0))
    
    # Distance-duration relationship
    features['speed_implied'] = (
        df['distance_miles'] / (df['duration_minutes'].clip(lower=1) / 60)
    )
    
    feature_names = list(features.columns)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    
    return features, feature_names


class IsolationForestDetector:
    """
    Isolation Forest for multivariate anomaly detection.
    
    Why Isolation Forest?
    - Works well with high-dimensional data
    - Doesn't assume data distribution (unlike Z-score)
    - Handles mixed feature types
    - Fast training and inference
    - Good at detecting global outliers (extreme prices)
    
    Features engineered for pricing:
    - Price relative to base (surge ratio)
    - Demand-supply gap
    - Price per mile / per minute
    - Hour-of-day encoded cyclically
    - Category-specific deviations
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to shared engineer_anomaly_features()."""
        features, self.feature_names = engineer_anomaly_features(df)
        return features
    
    def fit(self, df: pd.DataFrame) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest on historical (mostly normal) pricing data.
        """
        logger.info(f"Fitting Isolation Forest on {len(df)} events")
        
        features = self._engineer_features(df)
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.fit_transform(features)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self._is_fitted = True
        
        logger.info("Isolation Forest fitted successfully")
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly scores for new pricing events.
        
        Returns DataFrame with columns:
        - if_anomaly: bool
        - if_score: float [0, 1] (higher = more anomalous)
        - if_raw_score: float (raw Isolation Forest score)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features = self._engineer_features(df)
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(features)
        
        # Raw scores: negative = anomaly, positive = normal
        raw_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Normalize to [0, 1] where 1 = most anomalous
        # decision_function returns negative for outliers
        normalized_scores = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )
        
        results = pd.DataFrame({
            'if_anomaly': predictions == -1,
            'if_score': normalized_scores,
            'if_raw_score': raw_scores
        }, index=df.index)
        
        return results


class StatisticalDetector:
    """
    Statistical anomaly detection using multiple methods.
    
    Methods:
    1. Z-score: Detects values far from the mean (assumes ~normal distribution)
    2. Modified Z-score (MAD): Robust to outliers in the training data
    3. IQR method: Non-parametric, good for skewed distributions
    4. Rolling statistics: Detects sudden changes relative to recent history
    
    Why multiple statistical methods?
    - Z-score fails with heavy-tailed distributions (common in pricing)
    - IQR misses subtle anomalies in the "normal" range
    - Rolling stats catch temporal anomalies (sudden changes)
    - Ensemble of methods reduces false positives
    """
    
    def __init__(self, zscore_threshold: float = 3.0,
                 rolling_windows: List[int] = None):
        self.zscore_threshold = zscore_threshold
        self.rolling_windows = rolling_windows or [1, 6, 24]
        self.category_stats: Dict[str, Dict] = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'StatisticalDetector':
        """
        Compute per-category statistics from historical data.
        """
        logger.info(f"Computing statistical baselines from {len(df)} events")
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            prices = cat_data['final_price']
            surges = cat_data['surge_multiplier']
            
            self.category_stats[category] = {
                'price_mean': prices.mean(),
                'price_std': prices.std(),
                'price_median': prices.median(),
                'price_mad': np.median(np.abs(prices - prices.median())),
                'price_q1': prices.quantile(0.25),
                'price_q3': prices.quantile(0.75),
                'price_iqr': prices.quantile(0.75) - prices.quantile(0.25),
                'surge_mean': surges.mean(),
                'surge_std': surges.std(),
                'surge_p95': surges.quantile(0.95),
                'surge_p99': surges.quantile(0.99),
                'count': len(cat_data),
            }
        
        self._is_fitted = True
        logger.info(f"Computed statistics for {len(self.category_stats)} categories")
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply statistical anomaly detection methods — fully vectorized.
        
        Returns DataFrame with:
        - stat_anomaly: bool (any method flagged it)
        - stat_score: float [0, 1] (max score across methods)
        - stat_zscore: Z-score anomaly flag
        - stat_mad: Modified Z-score anomaly flag
        - stat_iqr: IQR anomaly flag
        - stat_rolling: Rolling statistics anomaly flag
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        results = pd.DataFrame(index=df.index)
        
        # Build lookup arrays from category_stats for vectorized ops
        stat_df = pd.DataFrame(self.category_stats).T
        prices = df['final_price'].values
        categories = df['category'].values
        
        # Map per-category stats onto every row
        cat_mean = df['category'].map(
            {c: s['price_mean'] for c, s in self.category_stats.items()}
        ).fillna(0).values
        cat_std = df['category'].map(
            {c: max(s['price_std'], 0.01) for c, s in self.category_stats.items()}
        ).fillna(1).values
        cat_median = df['category'].map(
            {c: s['price_median'] for c, s in self.category_stats.items()}
        ).fillna(0).values
        cat_mad = df['category'].map(
            {c: max(s['price_mad'], 0.01) for c, s in self.category_stats.items()}
        ).fillna(1).values
        cat_q1 = df['category'].map(
            {c: s['price_q1'] for c, s in self.category_stats.items()}
        ).fillna(0).values
        cat_q3 = df['category'].map(
            {c: s['price_q3'] for c, s in self.category_stats.items()}
        ).fillna(0).values
        cat_iqr = df['category'].map(
            {c: s['price_iqr'] for c, s in self.category_stats.items()}
        ).fillna(1).values
        
        # --- Method 1: Standard Z-score (vectorized) ---
        zscore_vals = np.abs((prices - cat_mean) / cat_std)
        zscore_flags = zscore_vals > self.zscore_threshold
        
        # --- Method 2: Modified Z-score / MAD (vectorized) ---
        mad_zscores = np.abs(0.6745 * (prices - cat_median) / cat_mad)
        mad_flags = mad_zscores > self.zscore_threshold
        
        # --- Method 3: IQR (vectorized) ---
        lower_bounds = cat_q1 - 2.0 * cat_iqr
        upper_bounds = cat_q3 + 2.0 * cat_iqr
        iqr_flags = (prices < lower_bounds) | (prices > upper_bounds)
        
        iqr_scores = np.where(
            prices < lower_bounds,
            (lower_bounds - prices) / np.maximum(cat_iqr, 0.01),
            np.where(
                prices > upper_bounds,
                (prices - upper_bounds) / np.maximum(cat_iqr, 0.01),
                0.0
            )
        )
        
        # --- Method 4: Rolling statistics ---
        rolling_flags = self._rolling_anomaly_detection(df)
        
        # Combine all methods
        results['stat_zscore'] = zscore_flags
        results['stat_mad'] = mad_flags
        results['stat_iqr'] = iqr_flags
        results['stat_rolling'] = rolling_flags
        
        results['stat_anomaly'] = (
            results['stat_zscore'] | results['stat_mad'] | 
            results['stat_iqr'] | results['stat_rolling']
        )
        
        # Composite score: normalized combination
        max_zscore = zscore_vals.max() or 1
        max_mad = mad_zscores.max() or 1
        max_iqr = iqr_scores.max() or 1
        
        results['stat_score'] = np.clip(
            0.35 * (zscore_vals / max_zscore) +
            0.35 * (mad_zscores / max_mad) +
            0.20 * (iqr_scores / max_iqr) +
            0.10 * results['stat_rolling'].astype(float),
            0.0, 1.0
        )
        
        return results
    
    def _rolling_anomaly_detection(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect anomalies based on sudden price changes relative to
        recent rolling statistics.
        """
        flags = pd.Series(False, index=df.index)
        
        for category in df['category'].unique():
            mask = df['category'] == category
            cat_data = df[mask].sort_values('timestamp') if 'timestamp' in df.columns else df[mask]
            
            if len(cat_data) < 10:
                continue
            
            prices = cat_data['final_price']
            
            for window in self.rolling_windows:
                window_size = max(window * 5, 10)  # At least 10 data points
                if len(prices) < window_size:
                    window_size = len(prices)
                
                rolling_mean = prices.rolling(window=window_size, min_periods=3).mean()
                rolling_std = prices.rolling(window=window_size, min_periods=3).std()
                
                deviation = abs(prices - rolling_mean) / rolling_std.clip(lower=0.01)
                window_flags = deviation > self.zscore_threshold
                
                flags.loc[window_flags[window_flags].index] = True
        
        return flags


class ContextualDetector:
    """
    Contextual anomaly detection that considers time-of-day, day-of-week,
    and category relationships.
    
    A price that's normal at 6pm during rush hour might be anomalous at 3am.
    This detector builds per-context baselines and flags deviations.
    """
    
    def __init__(self, n_std: float = 2.5):
        self.n_std = n_std
        self.context_baselines: Dict[str, Dict] = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'ContextualDetector':
        """Build per-context baselines (category x hour x day_type)."""
        logger.info("Building contextual baselines")
        
        df_work = df.copy()
        df_work['timestamp'] = pd.to_datetime(df_work['timestamp'])
        df_work['hour'] = df_work['timestamp'].dt.hour
        df_work['is_weekend'] = (df_work['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Build baseline for each context: (category, hour_block, day_type)
        df_work['hour_block'] = (df_work['hour'] // 4)  # 6 blocks of 4 hours
        
        for (cat, hblock, weekend), group in df_work.groupby(
            ['category', 'hour_block', 'is_weekend']
        ):
            key = f"{cat}_{hblock}_{weekend}"
            self.context_baselines[key] = {
                'mean': group['final_price'].mean(),
                'std': group['final_price'].std(),
                'median': group['final_price'].median(),
                'surge_mean': group['surge_multiplier'].mean(),
                'surge_std': group['surge_multiplier'].std(),
                'count': len(group)
            }
        
        self._is_fitted = True
        logger.info(f"Built {len(self.context_baselines)} context baselines")
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect contextual anomalies — fully vectorized."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        df_work = df.copy()
        df_work['timestamp'] = pd.to_datetime(df_work['timestamp'])
        df_work['hour'] = df_work['timestamp'].dt.hour
        df_work['is_weekend'] = (df_work['timestamp'].dt.dayofweek >= 5).astype(int)
        df_work['hour_block'] = (df_work['hour'] // 4)
        df_work['_ctx_key'] = (
            df_work['category'] + '_' + 
            df_work['hour_block'].astype(str) + '_' + 
            df_work['is_weekend'].astype(str)
        )
        
        # Map baseline stats onto every row
        ctx_mean = df_work['_ctx_key'].map(
            {k: v['mean'] for k, v in self.context_baselines.items() if v['count'] >= 5}
        )
        ctx_std = df_work['_ctx_key'].map(
            {k: max(v['std'], 0.01) for k, v in self.context_baselines.items() if v['count'] >= 5}
        )
        ctx_surge_mean = df_work['_ctx_key'].map(
            {k: v['surge_mean'] for k, v in self.context_baselines.items() if v['count'] >= 5}
        )
        ctx_surge_std = df_work['_ctx_key'].map(
            {k: max(v['surge_std'], 0.01) for k, v in self.context_baselines.items() if v['count'] >= 5}
        )
        
        has_baseline = ctx_mean.notna()
        
        price_dev = np.where(
            has_baseline,
            np.abs(df_work['final_price'].values - ctx_mean.fillna(0).values) / ctx_std.fillna(1).values,
            0.0
        )
        surge_dev = np.where(
            has_baseline,
            np.abs(df_work['surge_multiplier'].values - ctx_surge_mean.fillna(0).values) / ctx_surge_std.fillna(1).values,
            0.0
        )
        
        ctx_anomaly = has_baseline & ((price_dev > self.n_std) | (surge_dev > (self.n_std + 1)))
        ctx_score = np.clip(
            0.6 * (price_dev / (self.n_std * 2)) + 
            0.4 * (surge_dev / (self.n_std * 2)),
            0.0, 1.0
        )
        ctx_score = np.where(has_baseline, ctx_score, 0.0)
        
        return pd.DataFrame({
            'ctx_anomaly': ctx_anomaly.values if hasattr(ctx_anomaly, 'values') else ctx_anomaly,
            'ctx_score': ctx_score
        }, index=df.index)


class EnsembleAnomalyDetector:
    """
    Ensemble detector that combines all methods and produces
    a unified anomaly score with confidence.
    
    Scoring strategy:
    - Each method votes (anomaly or not) and provides a score
    - Final score = weighted average of individual scores
    - Confidence = agreement between methods
    - Severity = f(score, business impact)
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        if_config = self.config.get('anomaly_detection', {}).get('isolation_forest', {})
        zscore_thresh = self.config.get('anomaly_detection', {}).get('zscore_threshold', 2.5)
        rolling_windows = self.config.get('anomaly_detection', {}).get('rolling_windows', [1, 6, 24])
        
        self.isolation_forest = IsolationForestDetector(
            contamination=if_config.get('contamination', 0.10),
            n_estimators=if_config.get('n_estimators', 200),
            random_state=if_config.get('random_state', 42)
        )
        self.statistical = StatisticalDetector(
            zscore_threshold=zscore_thresh,
            rolling_windows=rolling_windows
        )
        self.contextual = ContextualDetector()
        
        # Method weights for ensemble scoring
        self.weights = {
            'isolation_forest': 0.40,
            'statistical': 0.35,
            'contextual': 0.25
        }
        
        # Configurable ensemble threshold (default 0.4 for better recall)
        self.ensemble_threshold = self.config.get(
            'anomaly_detection', {}
        ).get('ensemble_threshold', 0.4)
        
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'EnsembleAnomalyDetector':
        """Fit all detection methods on historical data."""
        logger.info(f"Fitting ensemble detector on {len(df)} events")
        
        self.isolation_forest.fit(df)
        self.statistical.fit(df)
        self.contextual.fit(df)
        
        self._is_fitted = True
        logger.info("Ensemble detector fitted")
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all detectors and produce ensemble results.
        
        Returns DataFrame with:
        - anomaly_score: float [0, 1] (ensemble score)
        - is_anomaly: bool (ensemble decision)
        - n_methods_flagged: int (0-3, agreement count)
        - severity: str (critical/high/medium/low)
        - detection_methods: str (which methods flagged it)
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        logger.info(f"Running ensemble prediction on {len(df)} events")
        
        # Run each detector
        if_results = self.isolation_forest.predict(df)
        stat_results = self.statistical.predict(df)
        ctx_results = self.contextual.predict(df)
        
        # Ensemble scoring
        ensemble = pd.DataFrame(index=df.index)
        
        # Weighted ensemble score
        ensemble['anomaly_score'] = np.clip(
            self.weights['isolation_forest'] * if_results['if_score'] +
            self.weights['statistical'] * stat_results['stat_score'] +
            self.weights['contextual'] * ctx_results['ctx_score'],
            0.0, 1.0
        )
        
        # Count how many methods flagged each event
        ensemble['n_methods_flagged'] = (
            if_results['if_anomaly'].astype(int) +
            stat_results['stat_anomaly'].astype(int) +
            ctx_results['ctx_anomaly'].astype(int)
        )
        
        # Decision: anomaly if score > threshold OR 2+ methods agree.
        # Two independent methods flagging the same event is a strong signal.
        ensemble['is_anomaly'] = (
            (ensemble['anomaly_score'] > self.ensemble_threshold) | 
            (ensemble['n_methods_flagged'] >= 2)
        )
        
        # Build detection method string
        if_flagged = if_results['if_anomaly'].values
        stat_flagged = stat_results['stat_anomaly'].values
        ctx_flagged = ctx_results['ctx_anomaly'].values
        
        methods = []
        for i in range(len(df)):
            flagged = []
            if if_flagged[i]:
                flagged.append('isolation_forest')
            if stat_flagged[i]:
                flagged.append('statistical')
            if ctx_flagged[i]:
                flagged.append('contextual')
            methods.append(','.join(flagged) if flagged else 'none')
        
        ensemble['detection_methods'] = methods
        
        # Severity classification
        ensemble['severity'] = ensemble['anomaly_score'].apply(self._classify_severity)
        
        # Include individual scores for debugging
        ensemble['if_score'] = if_results['if_score']
        ensemble['stat_score'] = stat_results['stat_score']
        ensemble['ctx_score'] = ctx_results['ctx_score']
        
        n_anomalies = ensemble['is_anomaly'].sum()
        logger.info(
            f"Ensemble found {n_anomalies} anomalies "
            f"({n_anomalies/len(df)*100:.2f}%)"
        )
        
        return ensemble
    
    def _classify_severity(self, score: float) -> str:
        """Map anomaly score to severity level."""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.3:
            return "low"
        return "none"


# ─────────────────────────────────────────────────────────
# Alternative Detectors (for model comparison benchmarking)
# ─────────────────────────────────────────────────────────

class LocalOutlierFactorDetector:
    """
    Local Outlier Factor for density-based anomaly detection.
    
    Why LOF?
    - Detects local outliers (points that are outliers relative to their
      neighbors, even if globally they look normal)
    - Good for pricing data where "normal" varies across categories/contexts
    - Complementary to Isolation Forest which finds global outliers
    
    Limitation:
    - LOF in sklearn uses `novelty=True` for predict-on-new-data mode,
      which requires fitting on clean data. We use novelty=False and
      fit+predict in one step (fit_predict), which means we can't
      predict on new unseen data — only score the training set.
      For a monitoring system, we'd retrain periodically.
    """
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.10):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to shared engineer_anomaly_features()."""
        features, self.feature_names = engineer_anomaly_features(df)
        return features
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit LOF and predict anomaly scores in one step.
        
        Returns DataFrame with:
        - lof_anomaly: bool
        - lof_score: float [0, 1] (higher = more anomalous)
        """
        logger.info(f"Running LOF on {len(df)} events (n_neighbors={self.n_neighbors})")
        
        features = self._engineer_features(df)
        X_scaled = self.scaler.fit_transform(features)
        
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        
        predictions = lof.fit_predict(X_scaled)
        raw_scores = lof.negative_outlier_factor_
        
        # Normalize to [0, 1] where 1 = most anomalous
        # negative_outlier_factor_ is negative; more negative = more outlier
        normalized_scores = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )
        
        results = pd.DataFrame({
            'lof_anomaly': predictions == -1,
            'lof_score': normalized_scores,
        }, index=df.index)
        
        n_anomalies = results['lof_anomaly'].sum()
        logger.info(f"LOF found {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        self._is_fitted = True
        return results


class OneClassSVMDetector:
    """
    One-Class SVM for boundary-based anomaly detection.
    
    Why One-Class SVM?
    - Learns a decision boundary around normal data
    - Good at detecting anomalies that fall outside the learned manifold
    - Works well when normal data forms a tight cluster
    - Uses RBF kernel to handle non-linear boundaries
    
    Limitation:
    - Scales O(n^2) to O(n^3) with training size. For large datasets,
      we subsample during training and predict on full data.
    - Sensitive to kernel hyperparameters (gamma, nu)
    """
    
    def __init__(self, nu: float = 0.10, kernel: str = 'rbf',
                 gamma: str = 'scale', max_train_size: int = 50000):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_train_size = max_train_size
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to shared engineer_anomaly_features()."""
        features, self.feature_names = engineer_anomaly_features(df)
        return features
    
    def fit(self, df: pd.DataFrame) -> 'OneClassSVMDetector':
        """
        Fit One-Class SVM. Subsamples if data exceeds max_train_size
        to keep training time reasonable (SVM is O(n^2+)).
        """
        logger.info(f"Fitting One-Class SVM on {len(df)} events")
        
        features = self._engineer_features(df)
        
        X = features.values
        
        # Subsample for SVM scalability
        if len(X) > self.max_train_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), self.max_train_size, replace=False)
            X_train = X[idx]
            logger.info(f"  Subsampled to {self.max_train_size} for SVM training")
        else:
            X_train = X
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
        )
        self.model.fit(X_train_scaled)
        self._is_fitted = True
        
        logger.info("One-Class SVM fitted successfully")
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly scores for pricing events.
        
        Returns DataFrame with:
        - svm_anomaly: bool
        - svm_score: float [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features = self._engineer_features(df)
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(X_scaled)
        raw_scores = self.model.decision_function(X_scaled)
        
        # Normalize: decision_function returns negative for outliers
        normalized_scores = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )
        
        results = pd.DataFrame({
            'svm_anomaly': predictions == -1,
            'svm_score': normalized_scores,
        }, index=df.index)
        
        n_anomalies = results['svm_anomaly'].sum()
        logger.info(f"One-Class SVM found {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return results


class DBSCANDetector:
    """
    DBSCAN for density-based clustering anomaly detection.
    
    Why DBSCAN?
    - Does not require pre-specifying number of clusters
    - Points that don't belong to any cluster are noise (anomalies)
    - Good at finding anomalies in arbitrary-shaped data distributions
    - No assumption about data distribution
    
    Approach:
    - Fit DBSCAN on the feature space
    - Points labeled as noise (label=-1) are anomalies
    - Score is based on distance to nearest core point
    
    Limitation:
    - Sensitive to eps and min_samples. We auto-tune eps using the
      k-distance graph heuristic (knee point of sorted k-distances).
    - Scales O(n log n) with spatial indexing, but can be slow for
      very high-dimensional data.
    """
    
    def __init__(self, eps: Optional[float] = None, min_samples: int = 10,
                 auto_tune_eps: bool = True):
        self.eps = eps
        self.min_samples = min_samples
        self.auto_tune_eps = auto_tune_eps
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to shared engineer_anomaly_features()."""
        features, self.feature_names = engineer_anomaly_features(df)
        return features
    
    def _auto_tune_eps(self, X_scaled: np.ndarray) -> float:
        """
        Auto-tune eps using the k-distance graph heuristic.
        
        Compute k-nearest-neighbor distances, sort them, and pick
        the knee point as eps. This avoids manual tuning.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Subsample for speed
        n = min(len(X_scaled), 10000)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), n, replace=False)
        X_sample = X_scaled[idx]
        
        nn = NearestNeighbors(n_neighbors=self.min_samples)
        nn.fit(X_sample)
        distances, _ = nn.kneighbors(X_sample)
        k_distances = np.sort(distances[:, -1])
        
        # Simple knee detection: find where the second derivative is maximum
        # (steepest change in slope)
        if len(k_distances) < 10:
            return 1.5  # fallback
        
        diffs = np.diff(k_distances)
        diffs2 = np.diff(diffs)
        knee_idx = np.argmax(diffs2) + 2
        
        eps = float(k_distances[min(knee_idx, len(k_distances) - 1)])
        # Clamp to reasonable range
        eps = max(0.5, min(eps, 5.0))
        
        logger.info(f"  DBSCAN auto-tuned eps={eps:.3f}")
        return eps
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run DBSCAN and identify noise points as anomalies.
        
        Returns DataFrame with:
        - dbscan_anomaly: bool (noise point = anomaly)
        - dbscan_score: float [0, 1] (distance to nearest core point)
        - dbscan_cluster: int (cluster label, -1 = noise)
        """
        logger.info(f"Running DBSCAN on {len(df)} events")
        
        features = self._engineer_features(df)
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(features)
        
        # Auto-tune eps if needed
        if self.auto_tune_eps or self.eps is None:
            self.eps = self._auto_tune_eps(X_scaled)
        
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            n_jobs=-1
        )
        labels = dbscan.fit_predict(X_scaled)
        
        is_noise = labels == -1
        
        # Score: distance to nearest core sample
        # For noise points, compute distance to closest non-noise point
        from sklearn.neighbors import NearestNeighbors
        
        core_mask = ~is_noise
        if core_mask.sum() > 0 and is_noise.sum() > 0:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(X_scaled[core_mask])
            distances, _ = nn.kneighbors(X_scaled)
            distances = distances.flatten()
            
            # Normalize distances to [0, 1]
            max_dist = distances.max() if distances.max() > 0 else 1.0
            scores = distances / max_dist
        else:
            scores = np.zeros(len(df))
        
        results = pd.DataFrame({
            'dbscan_anomaly': is_noise,
            'dbscan_score': np.clip(scores, 0, 1),
            'dbscan_cluster': labels,
        }, index=df.index)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_anomalies = is_noise.sum()
        logger.info(
            f"DBSCAN found {n_clusters} clusters, "
            f"{n_anomalies} noise points ({n_anomalies/len(df)*100:.2f}%)"
        )
        
        self._is_fitted = True
        return results
