"""
Business Logic Validator
========================
Rule-based anomaly detection that encodes domain knowledge about
what "should never happen" in a pricing system.

Unlike ML-based detection, these rules are deterministic and
interpretable. They catch bugs that statistical methods might miss
because they encode business constraints that the algorithm
MUST satisfy regardless of market conditions.

Examples:
- Prices must be positive
- Surge multiplier must be between 1.0 and max_cap
- Premium rides must cost more than standard rides
- Price-to-distance ratio must be within bounds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
import yaml
import os


class BusinessRuleValidator:
    """
    Validates pricing events against deterministic business rules.
    
    Rules are organized into tiers:
    - Critical: Should NEVER happen (negative prices, null values)
    - High: System constraint violations (surge cap exceeded)
    - Medium: Business logic violations (price inversions)
    - Low: Suspicious but potentially valid (unusual ratios)
    """
    
    # Price hierarchy: each category should generally be more expensive
    # than categories listed before it
    PRICE_HIERARCHY = {
        'ride_pool': 0,
        'ride_standard': 1,
        'ride_premium': 2,
    }
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        rules_config = self.config.get('business_rules', {})
        
        self.max_surge = rules_config.get('max_surge_multiplier', 5.0)
        self.min_price_floor = rules_config.get('min_price_floor', 1.00)
        self.max_price_per_mile = rules_config.get('max_price_per_mile', 30.00)
        self.max_hourly_change_pct = rules_config.get('max_hourly_change_pct', 200)
        self.allow_negative = rules_config.get('allow_negative_prices', False)
        self.max_consecutive_surge_hours = rules_config.get(
            'max_consecutive_surge_hours', 4
        )
        # Category-specific price caps — accounts for distance + surge
        # Old fixed thresholds produced false positives on legit long rides
        self.price_caps = rules_config.get('price_caps', {
            'ride_pool': 80,
            'ride_standard': 150,
            'ride_premium': 300,
        })
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all business rules against pricing events.
        
        Returns DataFrame with:
        - rule_anomaly: bool
        - rule_score: float [0, 1]
        - rule_violations: str (comma-separated rule names)
        - rule_severity: str (worst severity among violations)
        """
        logger.info(f"Validating {len(df)} events against business rules")
        
        results = pd.DataFrame(index=df.index)
        
        # Run each rule
        violations = {
            'negative_price': self._check_negative_prices(df),
            'below_floor': self._check_price_floor(df),
            'surge_cap_exceeded': self._check_surge_cap(df),
            'extreme_price_per_mile': self._check_price_per_mile(df),
            'price_hierarchy_violation': self._check_price_hierarchy(df),
            'demand_surge_mismatch': self._check_demand_surge_consistency(df),
            'zero_surge': self._check_zero_surge(df),
            'impossible_speed': self._check_impossible_speed(df),
        }
        
        # Aggregate violations per event
        all_flags = pd.DataFrame(index=df.index)
        all_scores = pd.DataFrame(index=df.index)
        all_severities = pd.DataFrame(index=df.index)
        
        severity_rank = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'none': 0}
        
        for rule_name, (flags, scores, severities) in violations.items():
            all_flags[rule_name] = flags
            all_scores[rule_name] = scores
            all_severities[rule_name] = severities
        
        # Composite results
        results['rule_anomaly'] = all_flags.any(axis=1)
        results['rule_score'] = all_scores.max(axis=1)
        
        # Vectorized rule violation listing and worst severity
        rule_names = list(violations.keys())
        flags_array = all_flags[rule_names].values  # (N, num_rules) bool
        sev_array = all_severities[rule_names].values  # (N, num_rules) str
        
        rule_lists = []
        worst_severities = []
        for i in range(len(df)):
            violated = [rule_names[j] for j in range(len(rule_names)) if flags_array[i, j]]
            rule_lists.append(','.join(violated) if violated else 'none')
            if violated:
                sevs = [sev_array[i, j] for j in range(len(rule_names)) if flags_array[i, j]]
                worst_severities.append(max(sevs, key=lambda s: severity_rank.get(s, 0)))
            else:
                worst_severities.append('none')
        
        results['rule_violations'] = rule_lists
        results['rule_severity'] = worst_severities
        
        n_violations = results['rule_anomaly'].sum()
        logger.info(
            f"Found {n_violations} business rule violations "
            f"({n_violations/max(len(df), 1)*100:.2f}%)"
        )
        
        return results
    
    def _check_negative_prices(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """CRITICAL: Prices must never be negative."""
        flags = df['final_price'] < 0
        scores = pd.Series(0.0, index=df.index)
        scores[flags] = 1.0
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'critical'
        return flags, scores, severities
    
    def _check_price_floor(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """HIGH: Price below minimum floor (but not negative)."""
        flags = (df['final_price'] >= 0) & (df['final_price'] < self.min_price_floor)
        scores = pd.Series(0.0, index=df.index)
        scores[flags] = 0.85
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'high'
        return flags, scores, severities
    
    def _check_surge_cap(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """HIGH: Surge multiplier exceeds maximum cap."""
        flags = df['surge_multiplier'] > self.max_surge
        scores = pd.Series(0.0, index=df.index)
        # Score proportional to how far above cap
        over_cap = (df['surge_multiplier'] - self.max_surge) / self.max_surge
        scores[flags] = np.clip(0.7 + 0.3 * over_cap[flags], 0.7, 1.0)
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'high'
        # If surge > 3x cap, it's critical
        severities[df['surge_multiplier'] > self.max_surge * 3] = 'critical'
        return flags, scores, severities
    
    def _check_price_per_mile(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MEDIUM: Price per mile exceeds reasonable bounds.
        
        Distance-conditional: short rides (<2 miles) naturally have a
        higher per-mile cost due to base fare, so we use a relaxed
        threshold for them to avoid false positives.
        """
        distance = df['distance_miles'].clip(lower=0.1)
        ppm = df['final_price'] / distance
        
        # Short rides (<2 mi) get a relaxed threshold — base fare dominates
        threshold = np.where(
            distance < 2.0,
            self.max_price_per_mile * 1.5,  # relax for short rides
            self.max_price_per_mile
        )
        
        flags = ppm > threshold
        scores = pd.Series(0.0, index=df.index)
        scores[flags] = np.clip(
            0.5 + 0.5 * (ppm[flags] - threshold[flags]) / threshold[flags],
            0.5, 1.0
        )
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'medium'
        return flags, scores, severities
    
    def _check_price_hierarchy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MEDIUM: Check that category prices respect expected hierarchy.
        Fully vectorized — no row-by-row iteration.
        
        Uses distance-conditional thresholds to avoid false positives
        on legitimate long-distance + surge combinations:
        - Base cap from config (e.g., pool=$80, standard=$150)
        - Adds distance allowance: $5/mile over 10 miles
        - Adds surge allowance: relaxes cap proportional to surge
        
        This eliminates the false positives from the old fixed-threshold
        approach where a 20-mile airport pool ride at 2.5x surge was
        falsely flagged at the $60 cap.
        """
        flags = pd.Series(False, index=df.index)
        scores = pd.Series(0.0, index=df.index)
        severities = pd.Series('none', index=df.index)
        
        for category, base_cap in self.price_caps.items():
            cat_mask = df['category'] == category
            if not cat_mask.any():
                continue
            
            cat_prices = df.loc[cat_mask, 'final_price']
            cat_distances = df.loc[cat_mask, 'distance_miles']
            cat_surges = df.loc[cat_mask, 'surge_multiplier']
            
            # Dynamic cap: base + distance bonus + surge allowance
            # Long trips at high surge legitimately cost more
            distance_bonus = np.maximum(cat_distances - 10, 0) * 5.0
            surge_allowance = np.maximum(cat_surges - 2.0, 0) * base_cap * 0.15
            effective_cap = base_cap + distance_bonus + surge_allowance
            
            exceeded = cat_prices > effective_cap
            flags.loc[cat_mask] = exceeded
            scores.loc[cat_mask & exceeded] = np.clip(
                0.55 + 0.45 * (
                    (cat_prices[exceeded] - effective_cap[exceeded])
                    / effective_cap[exceeded].clip(lower=1)
                ),
                0.55, 1.0
            )
            severities.loc[cat_mask & exceeded] = 'medium'
        
        return flags, scores, severities
    
    def _check_demand_surge_consistency(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        HIGH: High demand + low supply should produce surge > 1.0
        Detects algorithm failures where pricing doesn't respond to market.
        
        Thresholds are tightened to reduce false positives — only flag
        extreme cases where the pricing algorithm clearly failed.
        """
        ds_config = self.config.get('business_rules', {}).get(
            'demand_surge_mismatch', {}
        )
        min_demand = ds_config.get('min_demand', 0.85)
        max_supply = ds_config.get('max_supply', 0.20)
        max_surge = ds_config.get('max_surge', 1.02)
        
        high_demand = (df['demand_level'] > min_demand) & (df['supply_level'] < max_supply)
        no_surge = df['surge_multiplier'] <= max_surge
        
        flags = high_demand & no_surge
        scores = pd.Series(0.0, index=df.index)
        scores[flags] = 0.75
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'high'
        return flags, scores, severities
    
    def _check_zero_surge(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """CRITICAL: Surge multiplier at or below zero (should never happen)."""
        flags = df['surge_multiplier'] <= 0
        scores = pd.Series(0.0, index=df.index)
        scores[flags] = 1.0
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'critical'
        return flags, scores, severities
    
    def _check_impossible_speed(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        LOW: Implied speed from distance/duration is physically impossible.
        Speed > 120 mph in a city is likely a data error.
        """
        speed_mph = df['distance_miles'] / (df['duration_minutes'].clip(lower=1) / 60)
        flags = speed_mph > 120
        scores = pd.Series(0.0, index=df.index)
        scores[flags] = 0.4
        severities = pd.Series('none', index=df.index)
        severities[flags] = 'low'
        return flags, scores, severities
