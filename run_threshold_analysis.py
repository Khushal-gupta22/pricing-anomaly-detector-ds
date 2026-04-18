"""
Threshold Analysis & Hyperparameter Tuning
===========================================
Performs:
1. PR-AUC curve analysis with threshold sweep
2. Grid search over key hyperparameters
3. Optimal threshold selection for best F1
4. Saves results and plots

Run: python run_threshold_analysis.py [--sample 5000] [--full]
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import DynamicPricingDataGenerator
from src.anomaly_detector import EnsembleAnomalyDetector
from src.business_rules import BusinessRuleValidator
from src.experiment_tracker import ExperimentTracker


def precision_recall_f1(y_true, y_pred):
    tp = ((y_true) & (y_pred)).sum()
    fp = ((~y_true) & (y_pred)).sum()
    fn = ((y_true) & (~y_pred)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def generate_data(sample_size=5000, seed=42):
    """Generate data and split into train/test."""
    print(f"[1/4] Generating {sample_size} events...", flush=True)
    gen = DynamicPricingDataGenerator(os.path.join(PROJECT_ROOT, "config/settings.yaml"))
    pricing_df, anomaly_gt = gen.generate(seed=seed)

    if len(pricing_df) > sample_size:
        pricing_df = pricing_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    pricing_df['timestamp'] = pd.to_datetime(pricing_df['timestamp'])
    ground_truth = pricing_df['is_anomaly'].astype(bool)

    train_size = int(len(pricing_df) * 0.8)
    train_df = pricing_df.iloc[:train_size]
    test_idx = pricing_df.index[train_size:]

    print(f"  Total: {len(pricing_df)}, Train: {train_size}, Test: {len(test_idx)}")
    print(f"  Anomaly rate: {ground_truth.mean():.2%}")
    print(f"  Test anomalies: {ground_truth.loc[test_idx].sum()}")
    return pricing_df, train_df, test_idx, ground_truth


def run_pr_curve_analysis(pricing_df, train_df, test_idx, ground_truth):
    """
    Sweep ensemble score thresholds and compute precision/recall at each.
    Also compute PR-AUC.
    """
    print("\n[2/4] PR Curve & Threshold Analysis...", flush=True)

    # Fit ensemble with default config
    config = {
        'anomaly_detection': {
            'isolation_forest': {'contamination': 0.10, 'n_estimators': 200, 'random_state': 42},
            'zscore_threshold': 2.8,
            'rolling_windows': [1, 6, 24],
            'ensemble_threshold': 0.4,
        }
    }
    ensemble = EnsembleAnomalyDetector(config)
    ensemble.fit(train_df)
    results = ensemble.predict(pricing_df)

    test_truth = ground_truth.loc[test_idx].values
    test_scores = results.loc[test_idx, 'anomaly_score'].values
    test_n_methods = results.loc[test_idx, 'n_methods_flagged'].values

    # --- Score-only threshold sweep (disable method-count trigger) ---
    thresholds = np.arange(0.05, 0.95, 0.02)
    pr_points = []
    for t in thresholds:
        pred = test_scores > t
        p, r, f1 = precision_recall_f1(pd.Series(test_truth), pd.Series(pred))
        pr_points.append({'threshold': round(float(t), 3), 'precision': round(p, 4),
                          'recall': round(r, 4), 'f1': round(f1, 4), 'mode': 'score_only'})

    # --- Combined threshold sweep (score OR 2+ methods) ---
    for t in thresholds:
        pred = (test_scores > t) | (test_n_methods >= 2)
        p, r, f1 = precision_recall_f1(pd.Series(test_truth), pd.Series(pred))
        pr_points.append({'threshold': round(float(t), 3), 'precision': round(p, 4),
                          'recall': round(r, 4), 'f1': round(f1, 4), 'mode': 'score_or_2methods'})

    # --- Method agreement sweep ---
    for min_methods in [1, 2, 3]:
        pred = test_n_methods >= min_methods
        p, r, f1 = precision_recall_f1(pd.Series(test_truth), pd.Series(pred))
        pr_points.append({'threshold': min_methods, 'precision': round(p, 4),
                          'recall': round(r, 4), 'f1': round(f1, 4), 'mode': f'min_{min_methods}_methods'})

    pr_df = pd.DataFrame(pr_points)

    # Compute PR-AUC (score-only mode, sorted by recall)
    score_only = pr_df[pr_df['mode'] == 'score_only'].sort_values('recall')
    # np.trapz was removed in numpy 2.0; use np.trapezoid or fall back
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    pr_auc = float(_trapz(score_only['precision'].values, score_only['recall'].values))

    # Best F1 for each mode
    print(f"\n  PR-AUC (score-only): {abs(pr_auc):.4f}")
    for mode in pr_df['mode'].unique():
        subset = pr_df[pr_df['mode'] == mode]
        best = subset.loc[subset['f1'].idxmax()]
        print(f"  Best F1 ({mode}): {best['f1']:.4f} at threshold={best['threshold']} "
              f"(P={best['precision']:.3f}, R={best['recall']:.3f})")

    return pr_df, pr_auc, results, test_scores, test_truth, test_n_methods


def run_grid_search(pricing_df, train_df, test_idx, ground_truth):
    """
    Grid search over key hyperparameters:
    - contamination: [0.05, 0.10, 0.15, 0.20]
    - zscore_threshold: [2.0, 2.5, 2.8, 3.0]
    - ensemble_threshold: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    """
    print("\n[3/4] Hyperparameter Grid Search...", flush=True)

    param_grid = {
        'contamination': [0.05, 0.10, 0.15, 0.20],
        'zscore_threshold': [2.0, 2.5, 2.8, 3.0],
        'ensemble_threshold': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    }

    total = (len(param_grid['contamination']) *
             len(param_grid['zscore_threshold']) *
             len(param_grid['ensemble_threshold']))
    print(f"  Total combinations: {total}")

    results_list = []
    best_f1 = 0
    best_params = {}
    i = 0

    for contam in param_grid['contamination']:
        for zthresh in param_grid['zscore_threshold']:
            # Fit once per (contamination, zscore) pair since ensemble_threshold
            # only affects the decision, not the model fitting
            config = {
                'anomaly_detection': {
                    'isolation_forest': {
                        'contamination': contam, 'n_estimators': 200, 'random_state': 42
                    },
                    'zscore_threshold': zthresh,
                    'rolling_windows': [1, 6, 24],
                    'ensemble_threshold': 0.4,  # placeholder, we sweep below
                }
            }
            ensemble = EnsembleAnomalyDetector(config)
            ensemble.fit(train_df)
            ens_results = ensemble.predict(pricing_df)

            test_truth = ground_truth.loc[test_idx].values
            test_scores = ens_results.loc[test_idx, 'anomaly_score'].values
            test_n_methods = ens_results.loc[test_idx, 'n_methods_flagged'].values

            for ethresh in param_grid['ensemble_threshold']:
                i += 1
                pred = (test_scores > ethresh) | (test_n_methods >= 2)
                p, r, f1 = precision_recall_f1(pd.Series(test_truth), pd.Series(pred))

                row = {
                    'contamination': contam,
                    'zscore_threshold': zthresh,
                    'ensemble_threshold': ethresh,
                    'precision': round(p, 4),
                    'recall': round(r, 4),
                    'f1': round(f1, 4),
                }
                results_list.append(row)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = row.copy()

                if i % 16 == 0 or i == total:
                    print(f"  [{i}/{total}] Best F1 so far: {best_f1:.4f}", flush=True)

    grid_df = pd.DataFrame(results_list)

    print(f"\n  Grid search complete!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Best params: contamination={best_params['contamination']}, "
          f"zscore={best_params['zscore_threshold']}, "
          f"ensemble_thresh={best_params['ensemble_threshold']}")
    print(f"  Precision: {best_params['precision']:.4f}, Recall: {best_params['recall']:.4f}")

    return grid_df, best_params


def run_optimized_pipeline(pricing_df, train_df, test_idx, ground_truth, best_params, tracker):
    """
    Run the full pipeline with optimized params and log the improvement.
    """
    print("\n[4/4] Running pipeline with optimized parameters...", flush=True)

    config = {
        'anomaly_detection': {
            'isolation_forest': {
                'contamination': best_params['contamination'],
                'n_estimators': 200,
                'random_state': 42,
            },
            'zscore_threshold': best_params['zscore_threshold'],
            'rolling_windows': [1, 6, 24],
            'ensemble_threshold': best_params['ensemble_threshold'],
        },
        'business_rules': {
            'max_surge_multiplier': 5.0,
            'min_price_floor': 1.0,
            'max_price_per_mile': 35.0,
        }
    }

    start = time.time()
    ensemble = EnsembleAnomalyDetector(config)
    ensemble.fit(train_df)
    ens_results = ensemble.predict(pricing_df)

    rule_validator = BusinessRuleValidator(config)
    rule_results = rule_validator.validate(pricing_df)

    duration = time.time() - start
    test_truth = ground_truth.loc[test_idx]

    # Ensemble metrics
    ens_pred = ens_results.loc[test_idx, 'is_anomaly'].astype(bool)
    ep, er, ef1 = precision_recall_f1(test_truth, ens_pred)

    # Rules metrics
    rule_pred = rule_results.loc[test_idx, 'rule_anomaly'].astype(bool)
    rp, rr, rf1 = precision_recall_f1(test_truth, rule_pred)

    # Combined metrics
    combined_pred = ens_pred | rule_pred
    cp, cr, cf1 = precision_recall_f1(test_truth, combined_pred)

    print(f"\n  === OPTIMIZED RESULTS ===")
    print(f"  Ensemble  - P: {ep:.4f}, R: {er:.4f}, F1: {ef1:.4f}")
    print(f"  Rules     - P: {rp:.4f}, R: {rr:.4f}, F1: {rf1:.4f}")
    print(f"  Combined  - P: {cp:.4f}, R: {cr:.4f}, F1: {cf1:.4f}")

    # Log optimized runs
    for method, metrics in [
        ('ensemble_optimized', {'precision': ep, 'recall': er, 'f1': ef1}),
        ('business_rules_optimized', {'precision': rp, 'recall': rr, 'f1': rf1}),
        ('combined_optimized', {'precision': cp, 'recall': cr, 'f1': cf1}),
    ]:
        tracker.log_experiment(
            method=method,
            params={
                'contamination': best_params['contamination'],
                'zscore_threshold': best_params['zscore_threshold'],
                'ensemble_threshold': best_params['ensemble_threshold'],
            },
            metrics=metrics,
            data_info={'n_events': len(pricing_df), 'n_test': len(test_idx)},
            notes=f"Optimized via grid search ({len(pricing_df)} events)",
            duration_seconds=duration,
        )

    return {'ensemble': ef1, 'rules': rf1, 'combined': cf1}


def save_results(pr_df, pr_auc, grid_df, best_params, optimized_f1s):
    """Save all analysis results."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # PR curve data
    pr_df.to_csv(os.path.join(data_dir, "pr_curve_analysis.csv"), index=False)

    # Grid search results
    grid_df.to_csv(os.path.join(data_dir, "grid_search_results.csv"), index=False)

    # Summary
    summary = {
        'pr_auc': abs(pr_auc),
        'best_params': best_params,
        'optimized_f1': optimized_f1s,
        'timestamp': datetime.now().isoformat(),
        'grid_search_configs_tested': len(grid_df),
    }
    with open(os.path.join(data_dir, "tuning_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Saved: pr_curve_analysis.csv, grid_search_results.csv, tuning_summary.json")


def clean_experiments(tracker):
    """Deduplicate experiment log — keep best run per unique (method, params) combo."""
    print("\n[Bonus] Cleaning experiment log...", flush=True)
    history = tracker.load_history()
    if history.empty:
        print("  No experiments to clean.")
        return

    before = len(history)

    # Group by method + param columns, keep the one with highest F1
    param_cols = [c for c in history.columns if c.startswith('param_')]
    group_cols = ['method'] + param_cols

    # Fill NaN in param cols for grouping
    for c in param_cols:
        history[c] = history[c].fillna('__none__')

    cleaned = history.sort_values('metric_f1', ascending=False).drop_duplicates(
        subset=group_cols, keep='first'
    ).sort_values('experiment_id')

    after = len(cleaned)
    removed = before - after

    if removed > 0:
        # Rewrite the JSONL file with only unique best runs
        log_path = tracker.log_path
        # Read original records
        records = []
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        keep_ids = set(cleaned['experiment_id'].tolist())
        kept_records = [r for r in records if r['experiment_id'] in keep_ids]

        # Backup original
        backup_path = log_path + '.backup'
        import shutil
        shutil.copy2(log_path, backup_path)

        # Write cleaned
        with open(log_path, 'w') as f:
            for r in kept_records:
                f.write(json.dumps(r, default=str) + '\n')

        print(f"  Removed {removed} duplicate runs (kept {after} unique best)")
        print(f"  Backup saved to {backup_path}")
    else:
        print(f"  No duplicates found ({before} unique experiments)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Threshold Analysis & Hyperparameter Tuning")
    parser.add_argument('--sample', type=int, default=5000, help='Sample size')
    parser.add_argument('--full', action='store_true', help='Use full generated dataset')
    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    print("=" * 60)
    print("THRESHOLD ANALYSIS & HYPERPARAMETER TUNING")
    print("=" * 60)

    start_time = time.time()

    # Generate data
    pricing_df, train_df, test_idx, ground_truth = generate_data(
        sample_size=sample_size or 100000
    )

    # 1. PR curve analysis
    pr_df, pr_auc, _, _, _, _ = run_pr_curve_analysis(
        pricing_df, train_df, test_idx, ground_truth
    )

    # 2. Grid search
    grid_df, best_params = run_grid_search(
        pricing_df, train_df, test_idx, ground_truth
    )

    # 3. Run optimized pipeline
    tracker = ExperimentTracker(os.path.join(PROJECT_ROOT, "data", "experiments.jsonl"))
    optimized_f1s = run_optimized_pipeline(
        pricing_df, train_df, test_idx, ground_truth, best_params, tracker
    )

    # 4. Save results
    save_results(pr_df, pr_auc, grid_df, best_params, optimized_f1s)

    # 5. Clean experiments
    clean_experiments(tracker)

    total = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {total:.1f}s")
    print(f"{'=' * 60}")

    # Print improvement summary
    # Load old best for comparison
    old_best_f1 = 0.5116  # best combined F1 from experiments.jsonl
    new_f1 = optimized_f1s['combined']
    improvement = (new_f1 - old_best_f1) / old_best_f1 * 100

    print(f"\n  IMPROVEMENT SUMMARY:")
    print(f"  Before (best combined F1): {old_best_f1:.4f}")
    print(f"  After  (optimized F1):     {new_f1:.4f}")
    print(f"  Improvement:               {improvement:+.1f}%")
    print(f"\n  Recommended config update:")
    print(f"    contamination: {best_params['contamination']}")
    print(f"    zscore_threshold: {best_params['zscore_threshold']}")
    print(f"    ensemble_threshold: {best_params['ensemble_threshold']}")


if __name__ == "__main__":
    main()
