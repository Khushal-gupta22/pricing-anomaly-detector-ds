"""
Real-World Dataset Adapter
==========================
Adapts public datasets into the format expected by our anomaly detection
pipeline. This validates that our detectors work on real-world distributions,
not just synthetic data.

Supported datasets:
- NYC Taxi & Limousine Commission (TLC) trip records
  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Usage:
    from src.real_data_adapter import NYCTaxiAdapter
    adapter = NYCTaxiAdapter()
    pricing_df = adapter.load_and_transform("data/yellow_tripdata_2024-01.parquet")
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from loguru import logger
import os


class NYCTaxiAdapter:
    """
    Transforms NYC TLC Yellow Taxi trip records into our pricing schema.
    
    NYC taxi data is publicly available at:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    
    The yellow taxi dataset contains:
    - tpep_pickup_datetime, tpep_dropoff_datetime
    - trip_distance (miles)
    - fare_amount, tip_amount, total_amount
    - PULocationID, DOLocationID (zones)
    - passenger_count, payment_type
    
    We map this to our schema:
    - timestamp -> tpep_pickup_datetime
    - category -> inferred from trip characteristics (short/medium/long)
    - base_price -> base fare ($3.50 flag drop + $0.50/0.2mi or $0.50/min)
    - surge_multiplier -> total_amount / expected_fare (>1 = de facto surge)
    - final_price -> total_amount
    - demand_level -> inferred from pickup density per zone per hour
    - supply_level -> inverse of demand (approximation)
    - distance_miles -> trip_distance
    - duration_minutes -> dropoff - pickup time
    - region -> mapped from PULocationID zones
    
    Anomaly injection:
    Real data already has natural anomalies (data entry errors, extreme fares,
    zero-distance trips, $0 fares). We also optionally inject synthetic anomalies
    for evaluation with ground truth.
    """
    
    # NYC taxi zones grouped into rough regions
    # Manhattan core (zones ~100-200), airports (1=Newark, 132=JFK, 138=LaGuardia)
    AIRPORT_ZONES = {1, 132, 138}
    MANHATTAN_CORE_ZONES = set(range(4, 13)) | set(range(42, 44)) | set(range(48, 50)) | \
                           set(range(68, 75)) | set(range(79, 88)) | set(range(90, 100)) | \
                           set(range(107, 114)) | set(range(125, 131)) | set(range(137, 138)) | \
                           set(range(140, 170)) | set(range(186, 195)) | set(range(202, 210)) | \
                           set(range(224, 234)) | set(range(236, 244)) | set(range(246, 250)) | \
                           set(range(261, 264))
    
    def __init__(self):
        pass
    
    def load_and_transform(
        self,
        file_path: str,
        max_rows: Optional[int] = 200000,
        inject_anomalies: bool = True,
        anomaly_rate: float = 0.03,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load NYC taxi data and transform to our pricing schema.
        
        Args:
            file_path: Path to parquet or CSV file
            max_rows: Cap rows for memory/speed (None = all)
            inject_anomalies: Whether to inject synthetic anomalies for evaluation
            anomaly_rate: Fraction of events to make anomalous
            seed: Random seed
            
        Returns:
            pricing_df: Transformed pricing events
            anomaly_df: Ground truth anomaly labels (if injected)
        """
        logger.info(f"Loading NYC taxi data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}\n"
                f"Download NYC taxi data from:\n"
                f"  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page\n"
                f"Example:\n"
                f"  wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet\n"
                f"  mv yellow_tripdata_2024-01.parquet data/"
            )
        
        # Load data
        if file_path.endswith('.parquet'):
            raw_df = pd.read_parquet(file_path)
        else:
            raw_df = pd.read_csv(file_path, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        
        logger.info(f"  Loaded {len(raw_df):,} raw records")
        
        # Subsample
        if max_rows and len(raw_df) > max_rows:
            raw_df = raw_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
            logger.info(f"  Subsampled to {len(raw_df):,} records")
        
        # Clean and transform
        pricing_df = self._transform(raw_df)
        
        # Natural anomaly flagging
        pricing_df = self._flag_natural_anomalies(pricing_df)
        
        # Optionally inject synthetic anomalies
        anomaly_df = pd.DataFrame()
        if inject_anomalies:
            pricing_df, anomaly_df = self._inject_anomalies(pricing_df, anomaly_rate, seed)
        
        logger.info(
            f"  Final dataset: {len(pricing_df):,} events, "
            f"{pricing_df['is_anomaly'].sum():,} anomalies "
            f"({pricing_df['is_anomaly'].mean()*100:.2f}%)"
        )
        
        return pricing_df, anomaly_df
    
    def _transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw NYC taxi data to our schema."""
        df = raw_df.copy()
        
        # Rename/extract columns
        df['timestamp'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['dropoff_time'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
        # Duration
        df['duration_minutes'] = (df['dropoff_time'] - df['timestamp']).dt.total_seconds() / 60
        
        # Distance
        df['distance_miles'] = df['trip_distance'].astype(float)
        
        # Price
        df['final_price'] = df['total_amount'].astype(float)
        
        # Base price (NYC taxi: $3.50 flag + $0.50/0.2mi = $2.50/mi)
        df['base_price'] = 3.50 + df['distance_miles'] * 2.50
        
        # Surge multiplier (effective)
        df['surge_multiplier'] = np.where(
            df['base_price'] > 0.5,
            df['final_price'] / df['base_price'],
            1.0
        )
        
        # Category based on trip distance
        df['category'] = pd.cut(
            df['distance_miles'],
            bins=[-np.inf, 2, 5, 15, np.inf],
            labels=['ride_short', 'ride_standard', 'ride_long', 'ride_premium']
        ).astype(str)
        
        # Region from pickup zone
        df['region'] = 'other'
        if 'PULocationID' in df.columns:
            pu_zone = df['PULocationID'].astype(int)
            df.loc[pu_zone.isin(self.AIRPORT_ZONES), 'region'] = 'airport'
            df.loc[pu_zone.isin(self.MANHATTAN_CORE_ZONES), 'region'] = 'downtown'
            df.loc[~pu_zone.isin(self.AIRPORT_ZONES | self.MANHATTAN_CORE_ZONES), 'region'] = 'suburbs'
        
        # Demand/supply approximation from hourly pickup density
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df.groupby('hour')['timestamp'].transform('count')
        max_hourly = hourly_counts.max()
        df['demand_level'] = np.clip(hourly_counts / max_hourly, 0.05, 1.0)
        df['supply_level'] = np.clip(1.0 - df['demand_level'] * 0.6 + np.random.default_rng(42).normal(0, 0.1, len(df)), 0.05, 1.0)
        
        # Filter out bad records
        mask = (
            (df['duration_minutes'] > 0.5) &
            (df['duration_minutes'] < 300) &
            (df['distance_miles'] > 0) &
            (df['distance_miles'] < 100) &
            (df['final_price'] > 0) &
            (df['final_price'] < 500) &
            df['timestamp'].notna()
        )
        df = df[mask].copy()
        
        # Set anomaly columns
        df['is_anomaly'] = False
        df['anomaly_type'] = None
        
        # Event ID
        df = df.reset_index(drop=True)
        df['event_id'] = range(len(df))
        
        # Select final columns
        cols = [
            'event_id', 'timestamp', 'category', 'base_price', 'surge_multiplier',
            'final_price', 'demand_level', 'supply_level', 'distance_miles',
            'duration_minutes', 'region', 'is_anomaly', 'anomaly_type'
        ]
        
        logger.info(f"  Transformed {len(df):,} valid records")
        return df[cols].copy()
    
    def _flag_natural_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag naturally occurring anomalies in real data.
        These are real data quality issues / extreme events.
        """
        # Extremely high prices (>$200 for a taxi ride)
        extreme_price = df['final_price'] > 200
        
        # Very short distance but high price (likely error)
        short_expensive = (df['distance_miles'] < 0.5) & (df['final_price'] > 50)
        
        # Extremely long trips (>50 miles)
        very_long = df['distance_miles'] > 50
        
        # Surge > 5x (abnormal effective surge)
        extreme_surge = df['surge_multiplier'] > 5
        
        natural = extreme_price | short_expensive | very_long | extreme_surge
        
        df.loc[natural, 'is_anomaly'] = True
        df.loc[extreme_price, 'anomaly_type'] = 'extreme_price'
        df.loc[short_expensive, 'anomaly_type'] = 'short_distance_high_price'
        df.loc[very_long, 'anomaly_type'] = 'extreme_distance'
        df.loc[extreme_surge, 'anomaly_type'] = 'extreme_surge'
        
        n_natural = natural.sum()
        logger.info(f"  Flagged {n_natural} natural anomalies ({n_natural/len(df)*100:.2f}%)")
        
        return df
    
    def _inject_anomalies(
        self, df: pd.DataFrame, rate: float, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Inject synthetic anomalies into real data for evaluation."""
        rng = np.random.default_rng(seed)
        
        normal_mask = ~df['is_anomaly']
        normal_indices = df[normal_mask].index.tolist()
        
        n_inject = int(len(normal_indices) * rate)
        inject_indices = rng.choice(normal_indices, size=n_inject, replace=False)
        
        anomaly_types = [
            'flash_crash', 'runaway_surge', 'negative_price',
            'surge_stuck_high', 'gradual_drift'
        ]
        
        anomaly_records = []
        
        for idx in inject_indices:
            atype = rng.choice(anomaly_types)
            original_price = df.loc[idx, 'final_price']
            
            if atype == 'flash_crash':
                df.loc[idx, 'final_price'] = 0.01
                df.loc[idx, 'surge_multiplier'] = 0.001
            elif atype == 'runaway_surge':
                mult = rng.uniform(10, 25)
                df.loc[idx, 'surge_multiplier'] = mult
                df.loc[idx, 'final_price'] = original_price * mult
            elif atype == 'negative_price':
                df.loc[idx, 'final_price'] = -abs(rng.normal(5, 2))
                df.loc[idx, 'surge_multiplier'] = -0.5
            elif atype == 'surge_stuck_high':
                df.loc[idx, 'surge_multiplier'] = 4.5
                df.loc[idx, 'final_price'] = df.loc[idx, 'base_price'] * 4.5
            elif atype == 'gradual_drift':
                drift = rng.uniform(1.5, 2.5)
                df.loc[idx, 'final_price'] = original_price * drift
            
            df.loc[idx, 'is_anomaly'] = True
            df.loc[idx, 'anomaly_type'] = atype
            
            anomaly_records.append({
                'timestamp': df.loc[idx, 'timestamp'],
                'category': df.loc[idx, 'category'],
                'anomaly_type': atype,
                'severity': 'critical' if atype in ['flash_crash', 'negative_price', 'runaway_surge'] else 'high',
                'expected_price': original_price,
                'actual_price': df.loc[idx, 'final_price'],
                'description': f'Injected {atype} (original=${original_price:.2f})',
            })
        
        anomaly_df = pd.DataFrame(anomaly_records)
        logger.info(f"  Injected {n_inject} synthetic anomalies into real data")
        
        return df, anomaly_df


def download_sample_nyc_data(output_path: str = "data/nyc_taxi_sample.parquet"):
    """
    Download a small sample of NYC taxi data for testing.
    
    Uses the TLC public dataset (January 2024 yellow taxi trips).
    This is ~45MB as parquet.
    """
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    
    logger.info(f"Downloading NYC taxi data from {url}")
    logger.info("This is ~45MB and may take a minute...")
    
    try:
        import urllib.request
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Manual download:")
        logger.info(f"  1. Visit {url}")
        logger.info(f"  2. Save to {output_path}")
        raise


if __name__ == "__main__":
    # Demo: download and transform
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Path to NYC taxi parquet')
    parser.add_argument('--download', action='store_true', help='Download sample data first')
    parser.add_argument('--max-rows', type=int, default=100000)
    args = parser.parse_args()
    
    if args.download:
        file_path = download_sample_nyc_data()
    elif args.file:
        file_path = args.file
    else:
        file_path = "data/yellow_tripdata_2024-01.parquet"
    
    adapter = NYCTaxiAdapter()
    pricing_df, anomaly_df = adapter.load_and_transform(file_path, max_rows=args.max_rows)
    
    print(f"\nDataset Summary:")
    print(f"  Total events: {len(pricing_df):,}")
    print(f"  Categories: {pricing_df['category'].value_counts().to_dict()}")
    print(f"  Regions: {pricing_df['region'].value_counts().to_dict()}")
    print(f"  Anomalies: {pricing_df['is_anomaly'].sum():,}")
    print(f"\nPrice Statistics:")
    print(pricing_df.groupby('category')['final_price'].describe().round(2))
