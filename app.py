"""
Phase 10: MLOps & Deployment
--------------------------------
Run with:  uvicorn app:app --reload
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Alternative Credit Scoring API", version="1.0")

MODEL = joblib.load("outputs/credit_model.joblib")
SCALER = joblib.load("outputs/scaler.joblib")
CONFIG = joblib.load("outputs/model_config.joblib")

# WHY THIS THRESHOLD IS 0.679, NOT SOMETHING CLOSE TO THE 3% RISK
# CEILING ITSELF: computed directly from the trained model's own test-
# set scores (the exact boundary score between the approved and
# declined applicants in train_model.py's approval-rate calculation) --
# NOT a round, assumed number. It looks high relative to a 3% target
# default rate because XGBoost's scale_pos_weight correction (needed to
# handle this dataset's 6.68% class imbalance during training) shifts
# the model's predicted probabilities well away from being literally
# calibrated as "this applicant's true default probability" -- the
# RANKING of applicants by score is what the business logic depends on,
# not the raw score's face value as a probability.
APPROVAL_SCORE_THRESHOLD = 0.6788


class ApplicantProfile(BaseModel):
    revolving_utilization: float  # fraction of available revolving credit in use
    age: int
    debt_ratio: float
    monthly_income: float
    open_credit_lines: int
    real_estate_loans: int
    num_dependents: int
    past_due_30_59_days: int
    past_due_60_89_days: int
    past_due_90_plus_days: int


class CreditDecision(BaseModel):
    risk_score: float
    decision: str  # "Approve" or "Decline"
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=CreditDecision)
def score_applicant(profile: ApplicantProfile):
    total_past_due = profile.past_due_30_59_days + profile.past_due_60_89_days + profile.past_due_90_plus_days
    row = {
        "RevolvingUtilizationOfUnsecuredLines": min(profile.revolving_utilization, 5),
        "age": profile.age,
        "DebtRatio": min(profile.debt_ratio, 1.13),
        "MonthlyIncome_log": np.log1p(profile.monthly_income),
        "NumberOfOpenCreditLinesAndLoans": profile.open_credit_lines,
        "NumberRealEstateLoansOrLines": profile.real_estate_loans,
        "NumberOfDependents": profile.num_dependents,
        "total_past_due_incidents": total_past_due,
        "has_sentinel_flag": 0,  # a live applicant's raw input never carries the historical sentinel coding artifact
        "income_was_missing": 0,
    }

    X = pd.DataFrame([row])[CONFIG["feature_cols"]]
    X_input = SCALER.transform(X) if CONFIG["uses_scaling"] else X

    risk_score = float(MODEL.predict_proba(X_input)[0, 1])
    decision = "Decline" if risk_score >= APPROVAL_SCORE_THRESHOLD else "Approve"

    if hasattr(MODEL, "feature_importances_"):
        importances = pd.Series(MODEL.feature_importances_, index=CONFIG["feature_cols"])
    else:
        importances = pd.Series(abs(MODEL.coef_[0]), index=CONFIG["feature_cols"])
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return CreditDecision(
        risk_score=round(risk_score, 4),
        decision=decision,
        top_factors=top_factors,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Alternative Credit Scoring</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Alternative Credit Scoring — Test Form</h2>
        <p>Full docs at <a href="/docs">/docs</a>.</p>
        <form id="creditForm">
            <label>Revolving credit utilization (0-1+): <input name="revolving_utilization" type="number" step="0.01" value="0.3"></label><br><br>
            <label>Age: <input name="age" type="number" value="35"></label><br><br>
            <label>Debt-to-income ratio: <input name="debt_ratio" type="number" step="0.01" value="0.4"></label><br><br>
            <label>Monthly income ($): <input name="monthly_income" type="number" value="4000"></label><br><br>
            <label>Open credit lines: <input name="open_credit_lines" type="number" value="6"></label><br><br>
            <label>Real estate loans: <input name="real_estate_loans" type="number" value="1"></label><br><br>
            <label>Dependents: <input name="num_dependents" type="number" value="1"></label><br><br>
            <label>Times 30-59 days past due: <input name="past_due_30_59_days" type="number" value="0"></label><br><br>
            <label>Times 60-89 days past due: <input name="past_due_60_89_days" type="number" value="0"></label><br><br>
            <label>Times 90+ days late: <input name="past_due_90_plus_days" type="number" value="0"></label><br><br>
            <button type="submit">Score this applicant</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("creditForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                revolving_utilization: parseFloat(form.get("revolving_utilization")),
                age: parseInt(form.get("age")),
                debt_ratio: parseFloat(form.get("debt_ratio")),
                monthly_income: parseFloat(form.get("monthly_income")),
                open_credit_lines: parseInt(form.get("open_credit_lines")),
                real_estate_loans: parseInt(form.get("real_estate_loans")),
                num_dependents: parseInt(form.get("num_dependents")),
                past_due_30_59_days: parseInt(form.get("past_due_30_59_days")),
                past_due_60_89_days: parseInt(form.get("past_due_60_89_days")),
                past_due_90_plus_days: parseInt(form.get("past_due_90_plus_days"))
            };
            const res = await fetch("/score", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Risk score: " + data.risk_score +
                " | Decision: " + data.decision +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """
