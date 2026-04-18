"""
Alerting System
===============
Classifies detected anomalies by severity, generates human-readable
alert descriptions, and tracks alert lifecycle.

Combines signals from:
- ML anomaly score
- Business rule violations
- Time series deviation
- Historical context (is this a recurring issue?)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class AlertClassifier:
    """
    Takes raw detection results and produces actionable alerts with:
    - Severity classification (critical/high/medium/low)
    - Human-readable descriptions
    - Recommended actions
    - Business impact estimates
    """
    
    # Average revenue per event by category (for impact estimation)
    AVG_REVENUE = {
        'ride_standard': 18.0,
        'ride_premium': 35.0,
        'ride_pool': 12.0,
        'delivery_food': 8.0,
        'delivery_grocery': 10.0,
    }
    
    def classify_and_enrich(
        self, 
        events_df: pd.DataFrame,
        ensemble_results: pd.DataFrame,
        rule_results: pd.DataFrame,
        ts_results: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Combine all detection signals into enriched alert records.
        
        Returns DataFrame of anomalies ready for storage/display with:
        - All original event data
        - Unified anomaly score
        - Severity classification
        - Human-readable description
        - Estimated business impact
        - Recommended action
        """
        # Merge all results
        combined = events_df.copy()
        combined = combined.join(ensemble_results, rsuffix='_ens')
        combined = combined.join(rule_results, rsuffix='_rule')
        
        if ts_results is not None:
            combined = combined.join(ts_results, rsuffix='_ts')
        
        # Filter to only anomalies (from any method)
        anomaly_mask = (
            combined.get('is_anomaly', pd.Series(False, index=combined.index)) |
            combined.get('rule_anomaly', pd.Series(False, index=combined.index))
        )
        
        if ts_results is not None:
            anomaly_mask = anomaly_mask | combined.get(
                'ts_anomaly', pd.Series(False, index=combined.index)
            )
        
        anomalies = combined[anomaly_mask].copy()
        
        if anomalies.empty:
            logger.info("No anomalies to classify")
            return pd.DataFrame()
        
        # Compute unified severity
        anomalies['unified_score'] = self._compute_unified_score(anomalies)
        anomalies['severity'] = anomalies['unified_score'].apply(
            self._severity_from_score
        )
        
        # Generate descriptions
        anomalies['description'] = anomalies.apply(
            self._generate_description, axis=1
        )
        
        # Estimate business impact
        anomalies['estimated_impact_usd'] = anomalies.apply(
            self._estimate_impact, axis=1
        )
        
        # Recommended action
        anomalies['recommended_action'] = anomalies['severity'].map({
            'critical': 'IMMEDIATE: Pause pricing algorithm for this category. Page on-call engineer.',
            'high': 'URGENT: Investigate within 1 hour. Check recent algorithm deployments.',
            'medium': 'REVIEW: Investigate within 4 hours. Check for regional configuration issues.',
            'low': 'MONITOR: Add to next review cycle. May be normal market fluctuation.',
        })
        
        # Detection method summary
        anomalies['detection_summary'] = anomalies.apply(
            self._summarize_detection, axis=1
        )
        
        logger.info(
            f"Classified {len(anomalies)} anomalies: "
            f"{(anomalies['severity'] == 'critical').sum()} critical, "
            f"{(anomalies['severity'] == 'high').sum()} high, "
            f"{(anomalies['severity'] == 'medium').sum()} medium, "
            f"{(anomalies['severity'] == 'low').sum()} low"
        )
        
        return anomalies
    
    def _compute_unified_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a single anomaly score from all detection methods.
        
        Weights:
        - Business rules get highest weight (deterministic, high confidence)
        - Ensemble ML score (good at subtle patterns)
        - Time series (good at temporal anomalies)
        """
        score = pd.Series(0.0, index=df.index)
        
        # Business rule score (weight: 0.40)
        if 'rule_score' in df.columns:
            score += 0.40 * df['rule_score'].fillna(0)
        
        # Ensemble anomaly score (weight: 0.35)
        if 'anomaly_score' in df.columns:
            score += 0.35 * df['anomaly_score'].fillna(0)
        
        # Time series score (weight: 0.25)
        if 'ts_score' in df.columns:
            score += 0.25 * df['ts_score'].fillna(0)
        
        return np.clip(score, 0.0, 1.0)
    
    def _severity_from_score(self, score: float) -> str:
        if score >= 0.85:
            return 'critical'
        elif score >= 0.65:
            return 'high'
        elif score >= 0.45:
            return 'medium'
        else:
            return 'low'
    
    def _generate_description(self, row: pd.Series) -> str:
        """Generate a human-readable description of the anomaly."""
        parts = []
        
        category = row.get('category', 'unknown')
        price = row.get('final_price', 0)
        surge = row.get('surge_multiplier', 1)
        region = row.get('region', 'unknown')
        
        # Price-based descriptions
        if price < 0:
            parts.append(f"NEGATIVE PRICE (${price:.2f}) for {category} in {region}")
        elif price < 1.0:
            parts.append(f"Near-zero price (${price:.2f}) for {category}")
        elif surge > 10:
            parts.append(f"Extreme surge ({surge:.1f}x) pushing {category} to ${price:.2f}")
        elif surge > 5:
            parts.append(f"High surge ({surge:.1f}x) on {category}, price=${price:.2f}")
        
        # Rule violation descriptions
        violations = row.get('rule_violations', 'none')
        if violations and violations != 'none':
            parts.append(f"Rule violations: {violations}")
        
        # Time series descriptions
        if row.get('ts_anomaly', False):
            expected = row.get('ts_expected', None)
            if expected and not pd.isna(expected):
                parts.append(
                    f"Outside forecast range "
                    f"(expected ~${expected:.2f}, got ${price:.2f})"
                )
        
        # Method agreement
        n_methods = row.get('n_methods_flagged', 0)
        if n_methods >= 3:
            parts.append(f"[HIGH CONFIDENCE: {n_methods}/3 methods agree]")
        
        if not parts:
            parts.append(
                f"Anomalous pricing for {category}: "
                f"${price:.2f} (surge={surge:.2f}x) in {region}"
            )
        
        return " | ".join(parts)
    
    def _estimate_impact(self, row: pd.Series) -> float:
        """
        Estimate financial impact of the anomaly in USD.
        
        For over-pricing: potential refund/chargeback cost
        For under-pricing: lost revenue
        """
        category = row.get('category', 'unknown')
        price = row.get('final_price', 0)
        base_expected = self.AVG_REVENUE.get(category, 15.0)
        
        # Deviation from expected revenue
        deviation = abs(price - base_expected)
        
        # Under-pricing is lost revenue; over-pricing risks chargebacks + reputation
        if price < base_expected:
            # Under-pricing: direct revenue loss
            impact = deviation
        else:
            # Over-pricing: revenue + 2x for chargeback/reputation cost
            impact = deviation * 2.0
        
        return round(impact, 2)
    
    def _summarize_detection(self, row: pd.Series) -> str:
        """Summarize which detection methods flagged this event."""
        methods = []
        
        if row.get('anomaly_score', 0) > 0.5:
            methods.append(f"ML ensemble (score={row['anomaly_score']:.2f})")
        
        if row.get('rule_anomaly', False):
            violations = row.get('rule_violations', 'none')
            methods.append(f"Business rules ({violations})")
        
        if row.get('ts_anomaly', False):
            methods.append(f"Time series (score={row.get('ts_score', 0):.2f})")
        
        return " + ".join(methods) if methods else "Ensemble threshold"


def prepare_alerts_for_db(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform enriched anomaly DataFrame into the format expected
    by the detected_anomalies database table.
    """
    if anomalies_df.empty:
        return pd.DataFrame()
    
    db_records = pd.DataFrame({
        'event_id': anomalies_df.get('event_id'),
        'timestamp': anomalies_df.get('timestamp'),
        'category': anomalies_df.get('category'),
        'detection_method': anomalies_df.get('detection_summary', 'unknown'),
        'anomaly_type': anomalies_df.get('anomaly_type', 'detected'),
        'anomaly_score': anomalies_df.get('unified_score', 0.0),
        'severity': anomalies_df.get('severity', 'low'),
        'final_price': anomalies_df.get('final_price'),
        'expected_price_low': anomalies_df.get('ts_lower'),
        'expected_price_high': anomalies_df.get('ts_upper'),
        'description': anomalies_df.get('description', ''),
    })
    
    return db_records
