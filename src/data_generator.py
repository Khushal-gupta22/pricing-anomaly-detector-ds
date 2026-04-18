"""
Synthetic Pricing Data Generator
=================================
Generates realistic dynamic pricing data that mimics ride-sharing / e-commerce
pricing algorithms, complete with intentional anomalies for detection testing.

Key design decisions:
- Prices follow time-of-day patterns (rush hour surges, late-night dips)
- Day-of-week effects (weekends vs weekdays)
- Random demand spikes (events, weather)
- Injected anomalies: flash crashes, stuck multipliers, negative prices,
  runaway surges, and correlated failures across categories
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import yaml
import os


@dataclass
class AnomalyEvent:
    """Records metadata about an injected anomaly for ground truth."""
    timestamp: datetime
    category: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    expected_price: float
    actual_price: float
    description: str


@dataclass
class PricingEvent:
    """A single pricing decision by the algorithm."""
    timestamp: datetime
    category: str
    base_price: float
    surge_multiplier: float
    final_price: float
    demand_level: float  # 0-1 normalized
    supply_level: float  # 0-1 normalized
    distance_miles: float
    duration_minutes: float
    region: str
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None


class DynamicPricingDataGenerator:
    """
    Generates synthetic pricing data that mirrors real-world dynamic pricing.
    
    The generator models:
    1. Base price per category
    2. Time-of-day surge patterns (rush hours, late night)
    3. Day-of-week patterns (weekday vs weekend)
    4. Supply-demand dynamics
    5. Regional variation
    6. Random external events (concerts, weather, holidays)
    7. Intentionally injected anomalies for ground truth
    """
    
    REGIONS = ["downtown", "suburbs", "airport", "university", "industrial"]
    
    # Hour-of-day surge profile (0-23): models realistic demand curves
    # Peak at 8am, 12pm, 6pm; trough at 3am
    HOURLY_SURGE_PROFILE = {
        0: 0.7, 1: 0.5, 2: 0.4, 3: 0.3, 4: 0.3, 5: 0.5,
        6: 0.8, 7: 1.2, 8: 1.5, 9: 1.3, 10: 1.0, 11: 1.1,
        12: 1.3, 13: 1.2, 14: 1.0, 15: 1.0, 16: 1.2, 17: 1.6,
        18: 1.8, 19: 1.5, 20: 1.2, 21: 1.1, 22: 1.0, 23: 0.8
    }
    
    # Day-of-week multiplier (Mon=0 ... Sun=6)
    DOW_MULTIPLIER = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.05, 4: 1.15,
        5: 1.3, 6: 1.2  # weekends are busier
    }
    
    # Region-specific demand bias
    REGION_BIAS = {
        "downtown": 1.3, "suburbs": 0.9, "airport": 1.4,
        "university": 1.1, "industrial": 0.7
    }
    
    ANOMALY_TYPES = [
        "surge_stuck_high",       # Multiplier gets stuck at peak value
        "surge_stuck_low",        # Multiplier drops to near zero
        "flash_crash",            # Price suddenly drops to near zero
        "runaway_surge",          # Multiplier escalates without bound
        "negative_price",         # Bug: price goes negative
        "price_inversion",        # Premium cheaper than standard
        "stale_price",            # Price doesn't change for hours (flat line)
        "demand_supply_mismatch", # High demand but no surge (algorithm failure)
        "regional_bleedthrough",  # Airport pricing applied in suburbs
        "rounding_error",         # Prices with weird decimal places
        "gradual_drift",          # Price slowly creeps up over hours (config leak)
    ]
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.rng = np.random.default_rng(seed=42)
        self.anomaly_log: List[AnomalyEvent] = []
        # Cluster state: when an anomaly "outbreak" starts, it persists
        # for N consecutive events (simulates a bug in production)
        self._cluster_remaining: int = 0
        self._cluster_type: Optional[str] = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        # Try multiple paths
        for path in [config_path, os.path.join(os.path.dirname(__file__), '..', config_path)]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        # Return defaults if config not found
        logger.warning(f"Config not found at {config_path}, using defaults")
        return self._default_config()
    
    def _default_config(self) -> dict:
        return {
            'data_generation': {
                'history_days': 90,
                'events_per_hour': 50,
                'categories': ['ride_standard', 'ride_premium', 'ride_pool',
                               'delivery_food', 'delivery_grocery'],
                'base_prices': {
                    'ride_standard': 12.50, 'ride_premium': 25.00,
                    'ride_pool': 8.00, 'delivery_food': 5.99,
                    'delivery_grocery': 7.99
                },
                'anomaly_rate': 0.03
            }
        }
    
    def _generate_demand_supply(
        self, timestamp: datetime, region: str
    ) -> Tuple[float, float]:
        """
        Generate realistic demand/supply levels based on time and region.
        
        Returns (demand, supply) each in [0, 1].
        """
        hour = timestamp.hour
        dow = timestamp.weekday()
        
        # Base demand from time patterns
        base_demand = self.HOURLY_SURGE_PROFILE[hour] / 1.8  # normalize to ~[0,1]
        base_demand *= self.DOW_MULTIPLIER[dow]
        base_demand *= self.REGION_BIAS[region]
        
        # Add stochastic noise
        demand = np.clip(
            base_demand + self.rng.normal(0, 0.1), 0.05, 1.0
        )
        
        # Supply is inversely correlated with demand (with lag and noise)
        # When demand spikes, supply takes time to catch up
        supply = np.clip(
            0.7 - 0.3 * demand + self.rng.normal(0, 0.12), 0.05, 1.0
        )
        
        return float(demand), float(supply)
    
    def _calculate_surge_multiplier(
        self, demand: float, supply: float, hour: int, category: str
    ) -> float:
        """
        Calculate surge multiplier from demand/supply dynamics.
        
        Models a real pricing algorithm:
        - Surge = f(demand/supply ratio)
        - Category-specific sensitivity
        - Smoothed to avoid jarring jumps
        """
        ratio = demand / max(supply, 0.01)
        
        # Category-specific surge sensitivity
        sensitivity = {
            'ride_standard': 1.0,
            'ride_premium': 0.7,   # Premium absorbs surge better
            'ride_pool': 1.3,      # Pool is more price-sensitive
            'delivery_food': 0.8,
            'delivery_grocery': 0.6
        }.get(category, 1.0)
        
        # Sigmoid-like surge curve: smooth transition from 1x to ~4x
        surge = 1.0 + (3.0 * sensitivity) / (1.0 + np.exp(-2.5 * (ratio - 1.2)))
        
        # Add micro-noise (real algorithms have small fluctuations)
        surge += self.rng.normal(0, 0.02)
        
        return float(max(surge, 1.0))
    
    def _generate_trip_attributes(
        self, category: str, region: str
    ) -> Tuple[float, float]:
        """Generate realistic distance and duration for a trip/delivery."""
        distance_params = {
            'ride_standard': (5.0, 3.0),
            'ride_premium': (8.0, 4.0),
            'ride_pool': (4.0, 2.0),
            'delivery_food': (3.0, 1.5),
            'delivery_grocery': (4.0, 2.0)
        }
        
        region_distance_mult = {
            "downtown": 0.7, "suburbs": 1.4, "airport": 2.0,
            "university": 0.8, "industrial": 1.2
        }
        
        mean_dist, std_dist = distance_params.get(category, (5.0, 2.5))
        mean_dist *= region_distance_mult.get(region, 1.0)
        
        distance = max(0.5, self.rng.normal(mean_dist, std_dist))
        # Duration correlates with distance (~3 min/mile + noise)
        duration = max(3.0, distance * 3.0 + self.rng.normal(5, 3))
        
        return float(round(distance, 2)), float(round(duration, 1))
    
    def _calculate_final_price(
        self, base_price: float, surge: float, distance: float, duration: float
    ) -> float:
        """
        Calculate final price with distance/duration components.
        
        price = (base_price + distance_charge + time_charge) * surge_multiplier
        """
        distance_charge = distance * 1.20  # $1.20 per mile
        time_charge = duration * 0.25       # $0.25 per minute
        
        raw_price = (base_price + distance_charge + time_charge) * surge
        return float(round(raw_price, 2))
    
    def _inject_anomaly(
        self, event: PricingEvent, anomaly_type: str
    ) -> PricingEvent:
        """
        Inject a specific anomaly type into a pricing event.
        
        Each anomaly type models a real failure mode in pricing systems.
        """
        original_price = event.final_price
        
        if anomaly_type == "surge_stuck_high":
            # Surge multiplier gets stuck at 4.5x regardless of demand
            event.surge_multiplier = 4.5
            event.final_price = self._calculate_final_price(
                event.base_price, 4.5, event.distance_miles, event.duration_minutes
            )
            severity = "high"
            desc = f"Surge stuck at 4.5x (demand={event.demand_level:.2f})"
            
        elif anomaly_type == "surge_stuck_low":
            # Surge drops to 0.1x — massive under-pricing
            event.surge_multiplier = 0.1
            event.final_price = self._calculate_final_price(
                event.base_price, 0.1, event.distance_miles, event.duration_minutes
            )
            severity = "critical"
            desc = "Surge collapsed to 0.1x — under-pricing"
            
        elif anomaly_type == "flash_crash":
            # Price suddenly drops to $0.01
            event.final_price = 0.01
            event.surge_multiplier = 0.001
            severity = "critical"
            desc = "Flash crash: price dropped to $0.01"
            
        elif anomaly_type == "runaway_surge":
            # Surge escalates to 15x-25x (way beyond normal bounds)
            runaway_mult = self.rng.uniform(15, 25)
            event.surge_multiplier = runaway_mult
            event.final_price = self._calculate_final_price(
                event.base_price, runaway_mult,
                event.distance_miles, event.duration_minutes
            )
            severity = "critical"
            desc = f"Runaway surge at {runaway_mult:.1f}x"
            
        elif anomaly_type == "negative_price":
            # Bug: price goes negative (refund > charge)
            event.final_price = -1 * abs(self.rng.normal(5, 2))
            event.surge_multiplier = -0.5
            severity = "critical"
            desc = f"Negative price: ${event.final_price:.2f}"
            
        elif anomaly_type == "price_inversion":
            # Premium category priced lower than standard
            event.base_price *= 0.3
            event.final_price = self._calculate_final_price(
                event.base_price, event.surge_multiplier,
                event.distance_miles, event.duration_minutes
            )
            severity = "high"
            desc = "Price inversion: premium cheaper than standard"
            
        elif anomaly_type == "stale_price":
            # Price is exactly the same as base (no demand adjustment)
            event.final_price = event.base_price
            event.surge_multiplier = 1.0
            severity = "medium"
            desc = "Stale price: no demand adjustment applied"
            
        elif anomaly_type == "demand_supply_mismatch":
            # High demand but surge is 1.0 (algorithm not responding)
            event.demand_level = 0.95
            event.supply_level = 0.1
            event.surge_multiplier = 1.0
            event.final_price = self._calculate_final_price(
                event.base_price, 1.0, event.distance_miles, event.duration_minutes
            )
            severity = "high"
            desc = "Demand at 0.95 but no surge applied"
            
        elif anomaly_type == "regional_bleedthrough":
            # Airport pricing (2x) applied in suburbs
            event.final_price *= 2.0
            severity = "medium"
            desc = f"Regional bleedthrough: airport pricing in {event.region}"
            
        elif anomaly_type == "rounding_error":
            # Price with absurd decimal places (floating point bug)
            event.final_price = round(event.final_price * 1.000001, 6)
            # Add some tiny fractional cents that shouldn't exist
            event.final_price += 0.003721
            severity = "low"
            desc = f"Rounding error: price=${event.final_price}"
        
        elif anomaly_type == "gradual_drift":
            # Price slowly creeps up — a config leak or rounding accumulation
            # Multiply by 1.3-1.8x (not dramatic enough to trigger point anomaly)
            drift_mult = self.rng.uniform(1.3, 1.8)
            event.final_price = round(event.final_price * drift_mult, 2)
            severity = "medium"
            desc = f"Gradual drift: price inflated by {drift_mult:.2f}x"
        
        else:
            severity = "low"
            desc = f"Unknown anomaly type: {anomaly_type}"
        
        event.is_anomaly = True
        event.anomaly_type = anomaly_type
        
        # Log the anomaly for ground truth
        self.anomaly_log.append(AnomalyEvent(
            timestamp=event.timestamp,
            category=event.category,
            anomaly_type=anomaly_type,
            severity=severity,
            expected_price=original_price,
            actual_price=event.final_price,
            description=desc
        ))
        
        return event
    
    def _should_inject_anomaly(self, timestamp: datetime) -> Optional[str]:
        """
        Decide whether to inject an anomaly at this timestamp.
        
        Uses a Poisson-like process with configurable rate, plus
        clustered anomaly bursts — when a bug hits production, it
        corrupts 5-20 consecutive transactions, not just one.
        """
        # Continue existing cluster
        if self._cluster_remaining > 0:
            self._cluster_remaining -= 1
            return self._cluster_type
        
        rate = self.config['data_generation'].get('anomaly_rate', 0.03)
        
        if self.rng.random() < rate:
            anomaly_type = self.rng.choice(self.ANOMALY_TYPES)
            # 30% chance this anomaly starts a cluster (burst of 5-20 events)
            if self.rng.random() < 0.30:
                self._cluster_remaining = int(self.rng.integers(5, 20))
                self._cluster_type = anomaly_type
            return anomaly_type
        return None
    
    def _add_external_events(
        self, timestamp: datetime
    ) -> float:
        """
        Simulate external events (concerts, weather, holidays) that
        cause legitimate demand spikes.
        
        Returns a demand multiplier > 1.0 during events.
        """
        # Simulate ~2 major events per week
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour
        
        # Concert/sports events (evening, semi-random days)
        if (day_of_year % 3 == 0) and (18 <= hour <= 23):
            return 1.5 + self.rng.uniform(0, 0.5)
        
        # Bad weather events (random, all day)
        if (day_of_year % 7 == 2) and self.rng.random() < 0.3:
            return 1.3 + self.rng.uniform(0, 0.4)
        
        # Holiday surge
        if timestamp.month == 12 and timestamp.day >= 20:
            return 1.4
        
        return 1.0
    
    def generate(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate the full synthetic dataset.
        
        Returns:
            pricing_df: DataFrame of all pricing events
            anomaly_df: DataFrame of ground truth anomaly labels
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        
        cfg = self.config['data_generation']
        
        if end_date is None:
            end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        if start_date is None:
            start_date = end_date - timedelta(days=cfg['history_days'])
        
        categories = cfg['categories']
        base_prices = cfg['base_prices']
        events_per_hour = cfg['events_per_hour']
        
        logger.info(
            f"Generating data from {start_date} to {end_date} "
            f"({(end_date - start_date).days} days, "
            f"~{events_per_hour} events/hour, {len(categories)} categories)"
        )
        
        all_events = []
        current = start_date
        
        while current < end_date:
            # Number of events this hour (Poisson distributed)
            n_events = max(1, self.rng.poisson(events_per_hour))
            
            # External event multiplier for this hour
            event_mult = self._add_external_events(current)
            
            for _ in range(n_events):
                # Random offset within the hour
                offset_minutes = self.rng.uniform(0, 60)
                ts = current + timedelta(minutes=offset_minutes)
                
                # Pick category and region
                category = self.rng.choice(categories)
                region = self.rng.choice(self.REGIONS)
                
                base_price = base_prices[category]
                
                # Generate demand/supply
                demand, supply = self._generate_demand_supply(ts, region)
                demand *= event_mult  # Apply external event boost
                demand = min(demand, 1.0)
                
                # Calculate surge
                surge = self._calculate_surge_multiplier(
                    demand, supply, ts.hour, category
                )
                
                # Generate trip attributes
                distance, duration = self._generate_trip_attributes(category, region)
                
                # Calculate price
                final_price = self._calculate_final_price(
                    base_price, surge, distance, duration
                )
                
                event = PricingEvent(
                    timestamp=ts,
                    category=category,
                    base_price=base_price,
                    surge_multiplier=round(surge, 4),
                    final_price=final_price,
                    demand_level=round(demand, 4),
                    supply_level=round(supply, 4),
                    distance_miles=distance,
                    duration_minutes=duration,
                    region=region
                )
                
                # Decide whether to inject anomaly
                anomaly_type = self._should_inject_anomaly(ts)
                if anomaly_type:
                    event = self._inject_anomaly(event, anomaly_type)
                
                all_events.append(event)
            
            current += timedelta(hours=1)
        
        # Build DataFrames
        pricing_df = pd.DataFrame([vars(e) for e in all_events])
        pricing_df = pricing_df.sort_values('timestamp').reset_index(drop=True)
        pricing_df['event_id'] = range(len(pricing_df))
        
        anomaly_df = pd.DataFrame([vars(a) for a in self.anomaly_log])
        
        logger.info(
            f"Generated {len(pricing_df)} pricing events, "
            f"{len(anomaly_df)} injected anomalies "
            f"({len(anomaly_df)/len(pricing_df)*100:.2f}%)"
        )
        
        return pricing_df, anomaly_df
    
    def generate_and_save(
        self, output_dir: str = "data"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate data and save to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        pricing_df, anomaly_df = self.generate()
        
        pricing_path = os.path.join(output_dir, "pricing_events.csv")
        anomaly_path = os.path.join(output_dir, "anomaly_ground_truth.csv")
        
        pricing_df.to_csv(pricing_path, index=False)
        anomaly_df.to_csv(anomaly_path, index=False)
        
        logger.info(f"Saved pricing data to {pricing_path}")
        logger.info(f"Saved anomaly ground truth to {anomaly_path}")
        
        return pricing_df, anomaly_df


if __name__ == "__main__":
    generator = DynamicPricingDataGenerator()
    pricing_df, anomaly_df = generator.generate_and_save()
    print(f"\nDataset Summary:")
    print(f"  Total events: {len(pricing_df):,}")
    print(f"  Date range: {pricing_df['timestamp'].min()} to {pricing_df['timestamp'].max()}")
    print(f"  Categories: {pricing_df['category'].nunique()}")
    print(f"  Anomalies: {len(anomaly_df):,} ({anomaly_df['anomaly_type'].value_counts().to_dict()})")
    print(f"\nPrice Statistics:")
    print(pricing_df.groupby('category')['final_price'].describe().round(2))
