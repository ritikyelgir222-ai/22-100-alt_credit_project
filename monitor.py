"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THE RETRAIN TRIGGER HERE IS THE APPROVAL RATE, NOT JUST AUC: this
project's entire business framing is "extend credit access" -- if the
approval rate at the fixed 3% risk ceiling quietly shrinks over time
(even with stable AUC), that's a direct, real erosion of the project's
stated goal, and is the number a credit-policy stakeholder actually
cares about month to month, not an abstract ranking-quality metric.

WHY THE RIGHT CADENCE HERE IS CLOSER TO DAY 7's LOAN-MATURATION LAG
THAN DAY 2's REAL-TIME FRAUD MONITORING: like Day 7, "did this applicant
actually become seriously delinquent" takes up to two years to fully
resolve (the dataset's own target definition), so genuinely new ground
truth to validate against accumulates slowly -- frequent retraining
wouldn't have much new signal to learn from.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import roc_auc_score

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import approval_rate_at_risk_ceiling, RISK_CEILING

APPROVAL_RATE_DROP_THRESHOLD = 0.05  # retrain if approval rate drops more than 5 points at the same risk ceiling


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/credit_model.joblib")
    config = joblib.load("outputs/model_config.joblib")

    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    feature_cols = config["feature_cols"]
    X = engineered[feature_cols]
    y = engineered["SeriousDlqin2yrs"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("=== Feature Drift (PSI): key risk features, train vs. test ===")
    for feat in ["total_past_due_incidents", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")

    scores = model.predict_proba(X_test)[:, 1]
    current_auc = roc_auc_score(y_test, scores)
    current_approval_rate, current_default_rate = approval_rate_at_risk_ceiling(y_test.values, scores)

    print(f"\n=== Performance on test batch ===")
    print(f"AUC: {current_auc:.4f}")
    print(f"Approval rate at {RISK_CEILING:.0%} risk ceiling: {current_approval_rate:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)
    baseline_approval_rate = baseline["test_approval_rate_at_risk_ceiling"]
    drop = baseline_approval_rate - current_approval_rate

    if drop > APPROVAL_RATE_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: approval rate dropped by {drop:.3f} (threshold: {APPROVAL_RATE_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (approval rate change: {-drop:+.3f}, threshold: {APPROVAL_RATE_DROP_THRESHOLD})")

    print("\n=== Recommended monitoring cadence ===")
    print("Similar cadence CATEGORY to Day 7's loan-default monitoring, for a similar")
    print("reason: 'seriously delinquent within 2 years' (this dataset's own target")
    print("definition) takes a long time to fully resolve, so genuinely new ground")
    print("truth accumulates slowly -- a monitoring system should track approval-rate")
    print("and score-distribution drift in the meantime, but full retraining on fresh")
    print("outcomes can't happen faster than outcomes actually mature.")


if __name__ == "__main__":
    run_monitoring_check()
