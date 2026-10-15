"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
WHY THE BUSINESS METRIC HERE IS "APPROVAL RATE AT A RISK CEILING," NOT
A COST MATRIX (Day 7) OR A RECALL FLOOR (Day 9): this project's entire
framing is "extend credit access" -- the business goal isn't just to
avoid bad loans, it's to approve as MANY creditworthy people as
possible, including applicants a traditional score might reject. The
right business question is: "if we commit to keeping the approved
population's delinquency rate at or below a target ceiling, what
fraction of applicants can we approve?" A higher approval rate at the
same risk ceiling is a direct, quantifiable measure of expanded access.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data

RISK_CEILING = 0.03  # target: approved population's delinquency rate <= 3% (well below the 6.68% baseline)


def approval_rate_at_risk_ceiling(y_true, y_score, ceiling=RISK_CEILING):
    """
    Sorts applicants by predicted risk (ascending) and finds the largest
    APPROVED group (lowest-risk applicants) whose actual delinquency
    rate stays at or below the ceiling -- the direct "how many people
    can we respons‌ibly extend credit to" business answer.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(y_score)  # ascending: lowest risk first
    sorted_y = y_true[order]
    cumulative_default_rate = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)

    eligible = np.where(cumulative_default_rate <= ceiling)[0]
    if len(eligible) == 0:
        return 0.0, 0.0
    n_approved = eligible[-1] + 1
    approval_rate = n_approved / len(y_true)
    actual_default_rate_among_approved = cumulative_default_rate[eligible[-1]]
    return approval_rate, actual_default_rate_among_approved


def train_and_evaluate():
    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    feature_cols = get_feature_columns()

    X = engineered[feature_cols]
    y = engineered["SeriousDlqin2yrs"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_val_scores = logreg.predict_proba(X_val_scaled)[:, 1]

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_val_scores = rf.predict_proba(X_val)[:, 1]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_val_scores = xgb.predict_proba(X_val)[:, 1]

    experiment_log = []
    for name, scores in [("logistic_regression (baseline)", logreg_val_scores),
                           ("random_forest", rf_val_scores),
                           ("xgboost", xgb_val_scores)]:
        auc = roc_auc_score(y_val, scores)
        approval_rate, actual_default = approval_rate_at_risk_ceiling(y_val.values, scores)
        experiment_log.append({
            "model": name, "val_auc": round(auc, 4),
            "val_approval_rate_at_3pct_ceiling": round(approval_rate, 4),
            "val_actual_default_rate_among_approved": round(actual_default, 4),
        })

    log_df = pd.DataFrame(experiment_log)
    print("=== Experiment Log (validation set) ===")
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)
    print(f"\nSELECTION CRITERION: at a fixed {RISK_CEILING:.0%} risk ceiling, pick whichever")
    print("model APPROVES THE MOST applicants -- i.e. extends credit access furthest")
    print("while keeping the approved population's actual default rate at or below the ceiling.")

    best_model_name = log_df.loc[log_df["val_approval_rate_at_3pct_ceiling"].idxmax(), "model"]
    print(f"\nModel selected: {best_model_name}")

    final_model = {"logistic_regression (baseline)": logreg, "random_forest": rf, "xgboost": xgb}[best_model_name]
    if best_model_name == "logistic_regression (baseline)":
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    test_scores = final_model.predict_proba(X_test_input)[:, 1]
    test_auc = roc_auc_score(y_test, test_scores)
    test_approval_rate, test_actual_default = approval_rate_at_risk_ceiling(y_test.values, test_scores)

    # WHY WE ALSO REPORT THE "APPROVE EVERYONE" BASELINE: this is the
    # direct point of comparison a stakeholder needs -- how much further
    # does the model extend credit access than a blanket approval policy
    # would, AT THE SAME EFFECTIVE RISK LEVEL (the baseline's natural
    # default rate is simply the population's overall rate)?
    naive_approval_rate = 1.0
    naive_default_rate = float(y_test.mean())

    business_summary = {
        "selected_model": best_model_name,
        "risk_ceiling": RISK_CEILING,
        "test_auc": round(test_auc, 4),
        "test_approval_rate_at_risk_ceiling": round(test_approval_rate, 4),
        "test_actual_default_rate_among_approved": round(test_actual_default, 4),
        "naive_approve_everyone_default_rate": round(naive_default_rate, 4),
        "n_test_applicants": len(y_test),
        "n_approved": int(round(test_approval_rate * len(y_test))),
        "note": (
            "approval_rate_at_risk_ceiling answers the central 'extend credit "
            "access' business question: of all applicants, what share can be "
            "responsibly approved while keeping the APPROVED group's actual "
            "default rate at or below the target ceiling? This is a direct, "
            "quantifiable measure of expanded access, not just a technical score."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    print(json.dumps(business_summary, indent=2))
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n=== Feature Importance (top 10) ===")
        print(importances.head(10).round(4).to_string())
        importances.to_csv("outputs/feature_importance.csv", header=["importance"])
    else:
        coefs = pd.Series(final_model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
        print("\n=== Logistic Regression Coefficients (top 10 by magnitude) ===")
        print(coefs.head(10).round(4).to_string())
        coefs.to_csv("outputs/feature_importance.csv", header=["coefficient"])

    joblib.dump(final_model, "outputs/credit_model.joblib")
    joblib.dump(scaler, "outputs/scaler.joblib")
    joblib.dump({
        "feature_cols": feature_cols,
        "uses_scaling": best_model_name == "logistic_regression (baseline)",
        "risk_ceiling": RISK_CEILING,
    }, "outputs/model_config.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": best_model_name,
            "data_source": "Give Me Some Credit (Kaggle competition, 2011)",
            "n_features": len(feature_cols),
            "training_rows": len(X_train),
            "risk_ceiling": RISK_CEILING,
            "test_approval_rate_at_risk_ceiling": round(test_approval_rate, 4),
            "intended_use": "Rank credit applicants by risk to approve the largest possible share while keeping the approved population's default rate at or below a target ceiling.",
            "known_limitations": (
                "Trained on 2011-era US consumer credit data, not genuinely "
                "alternative/non-bureau data (see data_loader.py's scope note "
                "on why the more elaborate Home Credit dataset wasn't used). "
                "The approval-rate business metric assumes the risk ceiling "
                "itself is the right one for a real lender's risk appetite, "
                "which would need to be set by an actual credit-policy team, "
                "not assumed from this exercise alone."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/credit_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
