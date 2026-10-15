# Alternative Credit Scoring — Day 22 (Fintech / Underbanked, ML)

Full project code for Day 22 of the 100-day series. Uses the real
**Give Me Some Credit** dataset (2011 Kaggle competition): 150,000 real,
anonymized consumer credit records.

## A note on scope, stated upfront

The more explicitly "alternative data" dataset for this exact business
problem is Home Credit Group's "Home Credit Default Risk" (a company
that literally serves underbanked populations using non-bureau signals
like occupation type and document completeness). That dataset's main
table alone is ~170MB — too large to reliably retrieve as a complete
file with this project's tooling. This project uses **Give Me Some
Credit** instead: still real behavioral credit data, aligned with the
same underlying business problem (extend credit using repayment/
utilization behavior, not a single opaque score), but smaller and more
tractable. Disclosed here honestly, not glossed over.

## Four real data quality issues, one with a genuine causal finding

1. **1 row with `age == 0`** — an unambiguous error, dropped.
2. **269 rows with a 96/98 "sentinel" code** in all three past-due columns simultaneously — not genuine counts, almost certainly a legacy system's placeholder. **These rows have a 54.6% delinquency rate vs. 6.7% overall** — far too predictive to simply drop. Captured as its own flag, then the sentinel values themselves replaced with a realistic capped count.
3. **The real story: `MonthlyIncome` missing in 19.8% of rows causes `DebtRatio` to explode** (up to 61,106+) for those rows, since the ratio's denominator breaks. Verified: 26,771 of 28,877 extreme-`DebtRatio` rows have missing income. Fixed in the correct order — impute income first, *then* cap the now-explainable `DebtRatio` outliers.
4. **`RevolvingUtilizationOfUnsecuredLines` up to 50,708** (should be a ratio near 0–1) — almost certainly a unit error, capped rather than dropped.

## An honest "didn't pan out" finding

The `has_sentinel_flag` engineered specifically to preserve the sentinel
rows' strong signal ended up with **zero feature importance** in the
final model — the capped `total_past_due_incidents` value alone already
captures that risk once the sentinel values are set to a realistic high
percentile. Reported directly, the same honesty standard as Day 7's
`monthly_burden` feature not panning out as hypothesized.

## The business metric: approval rate at a risk ceiling

This project's framing is "extend credit access," not just "avoid bad
loans" — a different business question than Day 7's cost-matrix
minimization or Day 9's recall floor. The metric: **of all applicants,
what share can be responsibly approved while keeping the approved
group's actual default rate at or below a 3% ceiling?**

| Model | Val Approval Rate @ 3% Ceiling |
|---|---|
| Logistic Regression (baseline) | 84.3% |
| Random Forest | 85.7% |
| **XGBoost** | **86.4%** |

**Test result: 87.4% of applicants approved**, with the approved
group's actual default rate landing exactly at the 3% ceiling — vs.
6.68% under a blanket "approve everyone" policy.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py          # Phase 5 — loads real data, flags 4 quality issues
python eda.py                   # Phase 6 — validates the sentinel-code and age findings
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — the 4 fixes, feature engineering
python train_model.py          # Phase 8-9 — approval-rate-at-ceiling model selection
python monitor.py              # Phase 13 — approval-rate drift monitoring
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** for a test form.

## File map

| File | SDLC Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Loads real data, flags 4 quality issues |
| `eda.py` | 6 | Validates the sentinel-code and missing-income findings |
| `clean_and_engineer.py` | 5 (fixes) + 7 | The 4 fixes (including the sentinel-flag/capping resolution), feature engineering |
| `split.py` | 7 | Stratified split |
| `train_model.py` | 8–9 | Approval-rate-at-risk-ceiling business metric and model selection |
| `app.py` | 10 | FastAPI scoring service |
| `monitor.py` | 13 | Approval-rate drift monitoring |

## Known limitations (stated honestly)

- **Not genuinely "alternative" data** in the fullest sense — a stated scope limitation (see above); this is behavioral credit-bureau-style data, not non-bureau signals.
- **2011-era US consumer data** — not validated against a current population or a different country's credit system.
- **`has_sentinel_flag` turned out to add no marginal value** in the final model — an honest negative result from feature engineering, not hidden after the fact.
- **The 3% risk ceiling is illustrative**, not derived from a real lender's actual risk appetite — a real deployment would need this set by an actual credit-policy team.
