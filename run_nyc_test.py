"""
Test anomaly detectors on real NYC Taxi data.
==============================================
Loads the NYC TLC Yellow Taxi parquet file, transforms it into our
pricing schema, runs ALL detectors (Ensemble + Business Rules + LOF +
DBSCAN), evaluates against ground truth (natural + injected anomalies),
and prints a full comparison report.

Usage:
    python run_nyc_test.py
    python run_nyc_test.py --max-rows 50000
    python run_nyc_test.py --file data/yellow_tripdata_2024-01.parquet
"""
import sys
import os
import subprocess
import time

# ── Auto-activate venv if working, else use system Python ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")

if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    try:
        subprocess.run([VENV_PYTHON, "-c", "import sys"], timeout=10,
                       check=True, capture_output=True)
        print("[nyc_test] Re-launching with venv Python...", flush=True)
        result = subprocess.run(
            [VENV_PYTHON, "-u", __file__] + sys.argv[1:],
            cwd=PROJECT_ROOT,
        )
        sys.exit(result.returncode)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass

# ── We are inside the venv now ──
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
from src.real_data_adapter import NYCTaxiAdapter
from src.anomaly_detector import (
    EnsembleAnomalyDetector, IsolationForestDetector,
    LocalOutlierFactorDetector, OneClassSVMDetector, DBSCANDetector
)
from src.business_rules import BusinessRuleValidator


def evaluate(y_true, y_pred, method_name):
    """Compute precision, recall, F1."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'method': method_name,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'flagged': int(tp + fp),
    }


def main():
    parser = argparse.ArgumentParser(description="Test detectors on NYC Taxi data")
    parser.add_argument('--file', type=str, default='data/nyc_taxi_sample.parquet',
                        help='Path to NYC taxi parquet file')
    parser.add_argument('--max-rows', type=int, default=100000,
                        help='Max rows to load (default: 100,000)')
    parser.add_argument('--anomaly-rate', type=float, default=0.03,
                        help='Rate of synthetic anomalies to inject (default: 3%%)')
    parser.add_argument('--no-inject', action='store_true',
                        help='Skip synthetic anomaly injection (only detect natural anomalies)')
    args = parser.parse_args()

    # ══════════════════════════════════════════════
    # STEP 1: Load and transform NYC Taxi data
    # ══════════════════════════════════════════════
    print("=" * 65, flush=True)
    print("  TESTING DETECTORS ON REAL NYC TAXI DATA", flush=True)
    print("=" * 65, flush=True)

    adapter = NYCTaxiAdapter()
    pricing_df, anomaly_df = adapter.load_and_transform(
        file_path=args.file,
        max_rows=args.max_rows,
        inject_anomalies=not args.no_inject,
        anomaly_rate=args.anomaly_rate,
    )

    print(f"\n[DATA] Dataset Summary:", flush=True)
    print(f"   Total events:     {len(pricing_df):,}", flush=True)
    print(f"   Total anomalies:  {pricing_df['is_anomaly'].sum():,} "
          f"({pricing_df['is_anomaly'].mean()*100:.2f}%)", flush=True)
    print(f"   Categories:       {sorted(pricing_df['category'].unique().tolist())}", flush=True)
    print(f"   Regions:          {sorted(pricing_df['region'].unique().tolist())}", flush=True)
    print(f"   Date range:       {pricing_df['timestamp'].min()} to "
          f"{pricing_df['timestamp'].max()}", flush=True)

    # Show per-category price stats
    print(f"\n   Price statistics (by category):", flush=True)
    stats = pricing_df.groupby('category')['final_price'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(stats.round(2).to_string(index=True), flush=True)

    # Show anomaly type breakdown
    anom_types = pricing_df[pricing_df['is_anomaly']]['anomaly_type'].value_counts()
    print(f"\n   Anomaly types:", flush=True)
    for atype, count in anom_types.items():
        print(f"     {atype}: {count}", flush=True)

    ground_truth = pricing_df['is_anomaly'].astype(bool)

    # ══════════════════════════════════════════════
    # STEP 2: Train/test split (80/20 temporal)
    # ══════════════════════════════════════════════
    split_idx = int(len(pricing_df) * 0.8)
    train_df = pricing_df.iloc[:split_idx].copy()
    test_df = pricing_df.iloc[split_idx:].copy()
    gt_test = test_df['is_anomaly'].astype(bool)

    print(f"\n   Train: {len(train_df):,} events | Test: {len(test_df):,} events", flush=True)

    results = []

    # ══════════════════════════════════════════════
    # STEP 3: Run each detector
    # ══════════════════════════════════════════════
    print(f"\n{'=' * 65}", flush=True)
    print("  RUNNING DETECTORS", flush=True)
    print(f"{'=' * 65}\n", flush=True)

    # --- Ensemble (Isolation Forest + Statistical + Contextual) ---
    print("[1/5] Ensemble (IF + Statistical + Contextual)...", flush=True)
    t0 = time.time()
    ensemble = EnsembleAnomalyDetector()
    ensemble.fit(train_df)
    ens_preds = ensemble.predict(test_df)
    elapsed = time.time() - t0
    r = evaluate(gt_test, ens_preds['is_anomaly'], 'Ensemble')
    r['time'] = f"{elapsed:.1f}s"
    results.append(r)
    print(f"   >> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
          f"({r['flagged']} flagged, {elapsed:.1f}s)", flush=True)

    # --- Business Rules ---
    print("[2/5] Business Rules...", flush=True)
    t0 = time.time()
    rules = BusinessRuleValidator()
    rule_preds = rules.validate(test_df)
    elapsed = time.time() - t0
    r = evaluate(gt_test, rule_preds['rule_anomaly'], 'Business Rules')
    r['time'] = f"{elapsed:.1f}s"
    results.append(r)
    print(f"   >> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
          f"({r['flagged']} flagged, {elapsed:.1f}s)", flush=True)

    # --- Combined (Ensemble + Rules) ---
    combined_pred = ens_preds['is_anomaly'] | rule_preds['rule_anomaly']
    r = evaluate(gt_test, combined_pred, 'Combined (Ens+Rules)')
    r['time'] = '-'
    results.append(r)
    print(f"   >> Combined: P={r['precision']:.3f}  R={r['recall']:.3f}  "
          f"F1={r['f1']:.3f}  ({r['flagged']} flagged)", flush=True)

    # --- Local Outlier Factor ---
    # LOF/DBSCAN are transductive (fit_predict on same data), so we run
    # them on the full dataset but evaluate ONLY on test indices for a
    # fair comparison with the inductive methods above.
    print("[3/5] Local Outlier Factor...", flush=True)
    t0 = time.time()
    lof = LocalOutlierFactorDetector(n_neighbors=20, contamination=0.10)
    lof_preds = lof.fit_predict(pricing_df)  # LOF is transductive
    elapsed = time.time() - t0
    lof_test = lof_preds.iloc[split_idx:]  # evaluate on test set only
    r = evaluate(gt_test, lof_test['lof_anomaly'], 'LOF')
    r['time'] = f"{elapsed:.1f}s"
    results.append(r)
    print(f"   >> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
          f"({r['flagged']} flagged, {elapsed:.1f}s)", flush=True)

    # --- DBSCAN ---
    print("[4/5] DBSCAN (auto-tuned eps)...", flush=True)
    t0 = time.time()
    dbscan = DBSCANDetector(min_samples=10, auto_tune_eps=True)
    dbscan_preds = dbscan.fit_predict(pricing_df)  # DBSCAN is transductive
    elapsed = time.time() - t0
    dbscan_test = dbscan_preds.iloc[split_idx:]  # evaluate on test set only
    n_clusters = dbscan_preds['dbscan_cluster'].nunique() - (1 if -1 in dbscan_preds['dbscan_cluster'].values else 0)
    r = evaluate(gt_test, dbscan_test['dbscan_anomaly'], 'DBSCAN')
    r['time'] = f"{elapsed:.1f}s"
    r['clusters'] = n_clusters
    results.append(r)
    print(f"   >> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
          f"({r['flagged']} flagged, {n_clusters} clusters, {elapsed:.1f}s)", flush=True)

    # --- One-Class SVM ---
    print("[5/5] One-Class SVM...", flush=True)
    t0 = time.time()
    svm = OneClassSVMDetector(nu=0.10, max_train_size=20000)
    svm.fit(train_df)
    svm_preds = svm.predict(test_df)
    elapsed = time.time() - t0
    r = evaluate(gt_test, svm_preds['svm_anomaly'], 'One-Class SVM')
    r['time'] = f"{elapsed:.1f}s"
    results.append(r)
    print(f"   >> P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
          f"({r['flagged']} flagged, {elapsed:.1f}s)", flush=True)

    # ══════════════════════════════════════════════
    # STEP 4: Final comparison table
    # ══════════════════════════════════════════════
    print(f"\n{'=' * 65}", flush=True)
    print("  RESULTS COMPARISON", flush=True)
    print(f"{'=' * 65}\n", flush=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False).reset_index(drop=True)

    # Pretty print
    print(f"{'Method':<25} {'Precision':>9} {'Recall':>8} {'F1':>8} {'Flagged':>8} {'Time':>8}", flush=True)
    print("-" * 65, flush=True)
    for _, row in results_df.iterrows():
        print(f"{row['method']:<25} {row['precision']:>9.3f} {row['recall']:>8.3f} "
              f"{row['f1']:>8.3f} {row['flagged']:>8,} {row['time']:>8}", flush=True)

    # ══════════════════════════════════════════════
    # STEP 5: Per-anomaly-type recall (best method)
    # ══════════════════════════════════════════════
    print(f"\n{'=' * 65}", flush=True)
    print("  PER-ANOMALY-TYPE DETECTION (Ensemble + Rules on test set)", flush=True)
    print(f"{'=' * 65}\n", flush=True)

    anom_test = test_df[test_df['is_anomaly'] == True]
    if not anom_test.empty:
        print(f"{'Anomaly Type':<30} {'Count':>6} {'Ens Recall':>11} {'Rule Recall':>12} {'Combined':>10}", flush=True)
        print("-" * 65, flush=True)
        for atype in anom_test['anomaly_type'].dropna().unique():
            mask = test_df['anomaly_type'] == atype
            total = mask.sum()
            if total == 0:
                continue
            ens_caught = ens_preds.loc[mask, 'is_anomaly'].sum()
            rule_caught = rule_preds.loc[mask, 'rule_anomaly'].sum()
            comb_caught = combined_pred[mask].sum()
            print(f"{str(atype):<30} {total:>6} {ens_caught/total:>11.1%} "
                  f"{rule_caught/total:>12.1%} {comb_caught/total:>10.1%}", flush=True)

    print(f"\n{'=' * 65}", flush=True)
    print("  DONE", flush=True)
    print(f"{'=' * 65}", flush=True)

    # Save results
    results_df.to_csv(os.path.join(PROJECT_ROOT, "data", "nyc_test_results.csv"), index=False)
    print(f"\nResults saved to data/nyc_test_results.csv", flush=True)


if __name__ == "__main__":
    main()
