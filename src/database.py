"""
Database Layer
==============
SQLite-based storage for pricing events, detected anomalies, and model state.
Uses SQLAlchemy for ORM with raw SQL fallbacks for performance-critical queries.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from loguru import logger


class PricingDatabase:
    """
    Manages persistent storage for the anomaly detection system.
    
    Schema:
    - pricing_events: Raw pricing data from the algorithm
    - detected_anomalies: Anomalies flagged by our detection pipeline
    - anomaly_ground_truth: Known injected anomalies (for evaluation)
    - detection_runs: Metadata about each pipeline execution
    - model_performance: Precision/recall tracking over time
    """
    
    SCHEMA_SQL = """
    -- Raw pricing events from the dynamic pricing algorithm
    CREATE TABLE IF NOT EXISTS pricing_events (
        event_id INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        category TEXT NOT NULL,
        base_price REAL NOT NULL,
        surge_multiplier REAL NOT NULL,
        final_price REAL NOT NULL,
        demand_level REAL,
        supply_level REAL,
        distance_miles REAL,
        duration_minutes REAL,
        region TEXT,
        is_anomaly INTEGER DEFAULT 0,
        anomaly_type TEXT
    );
    
    -- Anomalies detected by our pipeline
    CREATE TABLE IF NOT EXISTS detected_anomalies (
        anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id INTEGER,
        timestamp TEXT NOT NULL,
        category TEXT NOT NULL,
        detection_method TEXT NOT NULL,
        anomaly_type TEXT,
        anomaly_score REAL NOT NULL,
        severity TEXT NOT NULL,
        final_price REAL,
        expected_price_low REAL,
        expected_price_high REAL,
        description TEXT,
        is_acknowledged INTEGER DEFAULT 0,
        detected_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (event_id) REFERENCES pricing_events(event_id)
    );
    
    -- Ground truth anomalies (for evaluation)
    CREATE TABLE IF NOT EXISTS anomaly_ground_truth (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        category TEXT NOT NULL,
        anomaly_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        expected_price REAL,
        actual_price REAL,
        description TEXT
    );
    
    -- Pipeline execution metadata
    CREATE TABLE IF NOT EXISTS detection_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        events_processed INTEGER,
        anomalies_detected INTEGER,
        methods_used TEXT,
        status TEXT DEFAULT 'running',
        error_message TEXT
    );
    
    -- Model performance tracking
    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        method TEXT NOT NULL,
        precision_score REAL,
        recall_score REAL,
        f1_score REAL,
        evaluated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (run_id) REFERENCES detection_runs(run_id)
    );
    
    -- Indexes for query performance
    CREATE INDEX IF NOT EXISTS idx_pricing_timestamp ON pricing_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_pricing_category ON pricing_events(category);
    CREATE INDEX IF NOT EXISTS idx_pricing_anomaly ON pricing_events(is_anomaly);
    CREATE INDEX IF NOT EXISTS idx_detected_timestamp ON detected_anomalies(timestamp);
    CREATE INDEX IF NOT EXISTS idx_detected_severity ON detected_anomalies(severity);
    """
    
    def __init__(self, db_path: str = "data/pricing_anomalies.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA_SQL)
        logger.info(f"Database initialized at {self.db_path}")
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def load_pricing_data(self, df: pd.DataFrame) -> int:
        """
        Bulk load pricing events into the database.
        Returns number of rows inserted.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing data
            conn.execute("DELETE FROM pricing_events")
            
            df_db = df.copy()
            df_db['timestamp'] = df_db['timestamp'].astype(str)
            df_db['is_anomaly'] = df_db['is_anomaly'].astype(int)
            
            cols = [
                'event_id', 'timestamp', 'category', 'base_price',
                'surge_multiplier', 'final_price', 'demand_level',
                'supply_level', 'distance_miles', 'duration_minutes',
                'region', 'is_anomaly', 'anomaly_type'
            ]
            df_db[cols].to_sql(
                'pricing_events', conn, if_exists='replace', index=False
            )
        
        logger.info(f"Loaded {len(df)} pricing events into database")
        return len(df)
    
    def load_ground_truth(self, df: pd.DataFrame) -> int:
        """Load ground truth anomalies."""
        with sqlite3.connect(self.db_path) as conn:
            df_db = df.copy()
            df_db['timestamp'] = df_db['timestamp'].astype(str)
            df_db.to_sql(
                'anomaly_ground_truth', conn, if_exists='replace', index=False
            )
        logger.info(f"Loaded {len(df)} ground truth anomalies")
        return len(df)
    
    def save_detected_anomalies(self, anomalies: pd.DataFrame) -> int:
        """Save detected anomalies from the pipeline."""
        if anomalies.empty:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            anomalies_db = anomalies.copy()
            if 'timestamp' in anomalies_db.columns:
                anomalies_db['timestamp'] = anomalies_db['timestamp'].astype(str)
            anomalies_db.to_sql(
                'detected_anomalies', conn, if_exists='append', index=False
            )
        logger.info(f"Saved {len(anomalies)} detected anomalies")
        return len(anomalies)
    
    def get_pricing_events(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Query pricing events with optional filters."""
        query = "SELECT * FROM pricing_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_detected_anomalies(
        self,
        severity: Optional[str] = None,
        start_time: Optional[str] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """Query detected anomalies."""
        query = "SELECT * FROM detected_anomalies WHERE 1=1"
        params = []
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        query += f" ORDER BY anomaly_score DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """Get summary statistics of detected anomalies."""
        query = """
        SELECT 
            category,
            severity,
            detection_method,
            COUNT(*) as count,
            AVG(anomaly_score) as avg_score,
            MIN(timestamp) as first_seen,
            MAX(timestamp) as last_seen
        FROM detected_anomalies
        GROUP BY category, severity, detection_method
        ORDER BY count DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_hourly_stats(self, category: Optional[str] = None) -> pd.DataFrame:
        """Get hourly pricing statistics for trend analysis."""
        query = """
        SELECT 
            strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
            category,
            COUNT(*) as event_count,
            AVG(final_price) as avg_price,
            MIN(final_price) as min_price,
            MAX(final_price) as max_price,
            AVG(surge_multiplier) as avg_surge,
            AVG(demand_level) as avg_demand,
            AVG(supply_level) as avg_supply,
            SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
        FROM pricing_events
        WHERE 1=1
        """
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " GROUP BY hour, category ORDER BY hour"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def start_detection_run(self, methods: List[str]) -> int:
        """Record the start of a detection pipeline run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO detection_runs (started_at, methods_used, status) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), ",".join(methods), "running")
            )
            return cursor.lastrowid
    
    def complete_detection_run(
        self, run_id: int, events_processed: int, anomalies_detected: int,
        status: str = "completed", error: Optional[str] = None
    ):
        """Record completion of a detection run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE detection_runs 
                   SET completed_at=?, events_processed=?, anomalies_detected=?, 
                       status=?, error_message=?
                   WHERE run_id=?""",
                (datetime.now().isoformat(), events_processed,
                 anomalies_detected, status, error, run_id)
            )
    
    def save_model_performance(
        self, run_id: int, method: str,
        precision: float, recall: float, f1: float
    ):
        """Save evaluation metrics for a detection method."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO model_performance 
                   (run_id, method, precision_score, recall_score, f1_score)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, method, precision, recall, f1)
            )
    
    def get_performance_history(self) -> pd.DataFrame:
        """Get historical model performance metrics."""
        query = """
        SELECT mp.*, dr.started_at as run_date
        FROM model_performance mp
        JOIN detection_runs dr ON mp.run_id = dr.run_id
        ORDER BY dr.started_at DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def clear_detected_anomalies(self):
        """Clear all detected anomalies (for re-running pipeline)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM detected_anomalies")
        logger.info("Cleared detected anomalies table")
