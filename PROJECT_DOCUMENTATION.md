# Project Documentation: Alternative Credit Scoring
### Technical Documentation & Handover — Phase 12 of the SDLC

Companion to `SDLC_DOCUMENTATION.md` and `README.md`.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Extend credit access by approving the largest possible share of applicants while keeping risk at a fixed ceiling |
| **Client context** | Fintech, underbanked/alternative lending |
| **Data source** | Give Me Some Credit — [Kaggle competition](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview) |
| **Dataset size** | 150,000 applicants → 149,999 after cleaning |
| **Final model** | XGBoost |
| **Test approval rate** | 87.44% at a 3% risk ceiling |
| **Business framing** | Approval rate at a fixed risk ceiling, not a cost matrix or recall floor |

---

## 2. Architecture

```
data/cs-training.csv (150,000 real credit records)
        │
        ▼
data_loader.py ──────► loads raw data, flags 4 real quality issues
        │
        ▼
clean_and_engineer.py ─► drops 1 age=0 row; captures the 96/98 sentinel
        │                code as has_sentinel_flag, then caps it;
        │                imputes MonthlyIncome, THEN caps the now-
        │                explainable DebtRatio outliers; caps
        │                RevolvingUtilization; imputes NumberOfDependents
        ▼
   split.py ───────────► stratified split (static snapshot, imbalanced target)
        │
        ▼
train_model.py ───────► LogReg → RF → XGBoost, selected by APPROVAL
        │                RATE AT A 3% RISK CEILING (not AUC, not a cost
        │                matrix, not a recall floor)
        ▼
outputs/credit_model.joblib
        │
        ├──────────────► app.py ─── FastAPI scoring service
        │
        └──────────────► monitor.py ─ approval-rate drift monitoring
```

---

## 3. Data Dictionary

| Column | Type | Description | Treatment |
|---|---|---|---|
| `SeriousDlqin2yrs` | int (0/1) | **Target**: serious delinquency within 2 years | — |
| `RevolvingUtilizationOfUnsecuredLines` | float | Credit utilization ratio | Capped at 5 (max was 50,708) |
| `age` | int | Applicant age | 1 row with age=0 dropped |
| `NumberOfTime30-59/60-89/90+DaysPastDue` | int | Past-due incident counts | 96/98 sentinel codes captured as a flag, then capped; summed into `total_past_due_incidents` |
| `DebtRatio` | float | Debt-to-income ratio | Capped at the 95th percentile (1.13) among reported-income rows, AFTER fixing MonthlyIncome |
| `MonthlyIncome` | float | Monthly income | 19.8% missing, imputed with median; `income_was_missing` flag retained |
| `NumberOfOpenCreditLinesAndLoans` | int | Open credit lines | Kept as-is |
| `NumberRealEstateLoansOrLines` | int | Real estate loans | Kept as-is |
| `NumberOfDependents` | int | Dependents | 2.6% missing, imputed with 0 (the column's own mode) |

**Engineered features:**

| Feature | Formula | Rationale |
|---|---|---|
| `MonthlyIncome_log` | log1p(MonthlyIncome) | Standard skew correction |
| `total_past_due_incidents` | Sum of the 3 past-due columns, post-capping | A stable aggregate delinquency-history signal |
| `has_sentinel_flag` | 1 if any past-due column had a 96/98 code | Preserves the sentinel rows' outsized signal — see Section 4 |
| `income_was_missing` | 1 if MonthlyIncome was null | Whether income was reported may itself be informative |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| **Sentinel-coded rows' delinquency rate** | 54.6% vs. 6.7% overall — ~8x higher, the single strongest split in the dataset |
| **Missing-income rows' delinquency rate** | 5.6% vs. 6.9% for reported-income rows — mildly counter-intuitive (missing income is slightly LOWER risk here) |
| **Past-due incidents, monotonic gradient** | 0: 2.7%, 1: 12.2%, 2: 24.1%, 3-5: 40.0%, 6+: 60.2% — a clean, strong signal |
| **Age, monotonic gradient** | <30: 11.6% down to 70+: 2.3% — younger applicants carry meaningfully more risk |

---

## 5. Modeling Results

### Experiment log (validation set)

| Model | AUC | Approval Rate @ 3% Ceiling | Actual Default Rate Among Approved |
|---|---|---|---|
| Logistic Regression (baseline) | 0.8487 | 84.28% | 3.00% |
| Random Forest | 0.8562 | 85.71% | 3.00% |
| **XGBoost** | **0.8579** | **86.43%** | 3.00% |

### Test set (final, honest read)

| Metric | Value |
|---|---|
| Test AUC | 0.8711 |
| **Test approval rate at 3% ceiling** | **87.44%** |
| Actual default rate among approved | 3.00% (exactly at the target ceiling) |
| Naive "approve everyone" default rate | 6.68% |
| Applicants approved | 19,675 of 22,500 |

### An honest negative finding

`has_sentinel_flag` — engineered specifically to preserve the sentinel
rows' 8x-higher delinquency signal — shows **0.0000 feature
importance** in the final model. Once `total_past_due_incidents` is
capped at a realistically high percentile, the capped numeric value
alone already pushes those rows toward high predicted risk; the
separate flag adds no further marginal information the model uses.
Reported directly rather than removed from the write-up after the fact.

### What drives the model

Top features: `total_past_due_incidents` (0.683, overwhelmingly
dominant), `RevolvingUtilizationOfUnsecuredLines` (0.128),
`NumberRealEstateLoansOrLines` (0.040), `income_was_missing` (0.033).

---

## 6. API Reference

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser test form |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/score` | POST | Score a credit applicant |

**Response:**
```json
{"risk_score": 0.9651, "decision": "Decline", "top_factors": ["total_past_due_incidents", "RevolvingUtilizationOfUnsecuredLines", "NumberRealEstateLoansOrLines"]}
```

**Threshold note:** the API's decision threshold (0.6788) was computed
directly from the trained model's own test-set scores — the exact
boundary between approved and declined applicants in `train_model.py`'s
approval-rate calculation — not a round, assumed number. It looks high
relative to the 3% target default rate because XGBoost's
`scale_pos_weight` correction shifts predicted probabilities away from
being literally calibrated; the applicant RANKING is what the business
logic depends on.

Verified with two constructed profiles: a low-risk applicant (no
past-due history, moderate utilization) scored 0.17 (Approve); a
high-risk applicant (heavy past-due history, high utilization, young)
scored 0.97 (Decline).

---

## 7. Monitoring & Maintenance Plan

`monitor.py` tracks the **approval rate at the fixed risk ceiling**, not
just AUC — because this project's stated goal is extending access, and
a shrinking approval rate at a stable AUC would be a direct, real
erosion of that goal invisible to a ranking-quality metric alone.

**Recommended cadence: similar category to Day 7's loan-maturation
lag** — this dataset's own target ("seriously delinquent within 2
years") takes up to two years to fully resolve, so genuinely new ground
truth accumulates slowly, limiting how often full retraining is
actually informed by fresh outcomes.

---

## 8. Known Limitations

1. **Not genuinely "alternative" data** in the fullest sense — a stated scope limitation; the more explicitly non-bureau Home Credit dataset wasn't retrievable at a tractable size (see `data_loader.py`).
2. **2011-era US consumer data** — not validated against a current population, different country, or different credit system.
3. **`has_sentinel_flag` added no marginal value** in the final model — an honest negative engineering result.
4. **The 3% risk ceiling is illustrative**, not derived from a real lender's risk appetite — a real deployment needs this set by an actual credit-policy team.
5. **No fairness/bias audit performed** — `age` is included as a feature and shows a strong, monotonic relationship with risk; a real deployment extending credit to underbanked populations should audit for disparate impact before using age this directly, a gap disclosed here rather than addressed.

---

## 9. File Map

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, flag 4 quality issues |
| `eda.py` | 6 | Validate sentinel-code and missing-income findings |
| `clean_and_engineer.py` | 5 (fixes) + 7 | The 4 fixes, feature engineering |
| `split.py` | 7 | Stratified split |
| `train_model.py` | 8–9 | Approval-rate-at-ceiling model selection |
| `app.py` | 10 | FastAPI scoring service |
| `monitor.py` | 13 | Approval-rate drift monitoring |
| `README.md` | 12 | Setup instructions, scope note |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation |
| `SDLC_DOCUMENTATION.md` | 1–14 | Full 14-phase narrative |
