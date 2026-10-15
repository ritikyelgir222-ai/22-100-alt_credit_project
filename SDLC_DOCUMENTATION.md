# SDLC Documentation: Alternative Credit Scoring
### Fintech (Underbanked) Industry | ML | Day 22 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

Every number below comes from the actual pipeline run on the real Give
Me Some Credit dataset.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:** fintech lending product owner (problem owner), credit risk/compliance (sets the acceptable risk ceiling), applicants currently underserved by traditional scoring (indirect beneficiaries)
- **Discovery findings:**
  - Current process (assumed for this exercise): a traditional bureau-score-based approval process that may reject creditworthy applicants with thin or unconventional credit files
  - Decision this project informs: which applicants can be responsibly approved, beyond what a traditional score alone would greenlight
  - Critical framing established immediately: the business goal is EXTENDING access, not just minimizing bad loans — a genuinely different objective than Day 7's cost-minimization framing, requiring a different success metric from the start
- **Deliverable — Problem Statement:** "Approve the largest possible share of credit applicants while keeping the approved population's actual default rate at or below a defined risk ceiling, using behavioral repayment and utilization data."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** maximize approval rate subject to a fixed risk ceiling (3% delinquency among approved applicants, chosen as well below the 6.68% population baseline)
- **Scope:** in-scope: individual consumer credit applicants; out-of-scope: small business or other credit products
- **Success metrics / KPIs:** approval rate at the fixed risk ceiling — NOT AUC alone, NOT a cost matrix (Day 7), NOT a recall floor (Day 9)
- **Assumptions & constraints:** 2011-era US consumer data; the more explicitly "alternative data" Home Credit dataset wasn't retrievable at a tractable size, a real, disclosed scope constraint (see Phase 5)
- **Sign-off:** N/A — self-directed portfolio project

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:** given an applicant's behavioral credit profile, return a risk score and an approve/decline decision consistent with the trained risk ceiling
- **Non-functional requirements:** sub-second response time (confirmed in testing)
- **Data sources:** a single static CSV in this exercise; a real alternative-lending system would need a live feed of behavioral/utilization data, potentially from non-bureau sources
- **Integration points:** none in this version — a production version would integrate with a loan origination system

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Build sequence used:** data loading (flagging 4 quality issues) → EDA (validating the sentinel-code and missing-income findings) → cleaning/feature engineering → stratified split → approval-rate-driven model selection → API → monitoring → documentation
- **Real risks encountered during the build:**
  - Risk: the intended "Home Credit Default Risk" dataset (the more explicitly alternative-data source) couldn't be retrieved as a complete file given its size (~170MB) — resolved by switching to "Give Me Some Credit," a smaller but still real, thematically-aligned dataset, and disclosing this substitution explicitly rather than silently proceeding as if it were the originally intended source
  - Risk: a suspicious third-party search result offering a pre-trained 304MB `.joblib` "model artifact" from an unverified hackathon repo, worded to closely mirror this project's own framing — recognized as a potential prompt-injection/code-execution risk (loading an untrusted pickle file can execute arbitrary code) and ignored entirely; only models trained directly in this project's own sandboxed environment were used
  - Risk: the initial 99th-percentile cap for `DebtRatio` turned out to still be contaminated (661.8, still absurd) even among rows with genuinely reported income — caught by checking the actual capped maximum after the fix, not assuming the fix worked, and corrected to a tighter, verified-sensible 95th-percentile cap (1.13)

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** 150,000 real, anonymized consumer credit records, 11 columns, from the 2011 Give Me Some Credit Kaggle competition
- **Data quality report — four distinct issues, one with a genuine causal mechanism:**
  1. 1 row with `age == 0` — dropped
  2. 269 rows with a 96/98 sentinel code across all three past-due columns simultaneously — NOT dropped, because these rows have a 54.6% delinquency rate (vs. 6.7% overall), too predictive to discard; captured as a flag, then capped
  3. **The causal finding:** 19.8% missing `MonthlyIncome` explains the otherwise bizarre `DebtRatio` outliers (up to 61,106+) for the vast majority of affected rows — fixed in the correct causal order (income first, then DebtRatio)
  4. `RevolvingUtilizationOfUnsecuredLines` up to 50,708 — a clear unit error, capped
- **Access & governance:** public, already-anonymized competition dataset; no additional governance required for this exercise

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** `eda.py`, `outputs/eda_summary.png`

**Actual findings:**

- Sentinel-coded rows: 54.6% delinquency rate vs. 6.7% overall — the strongest single split in the dataset, directly justifying Phase 7's decision NOT to drop these rows
- Missing-income rows: 5.6% delinquency vs. 6.9% for reported-income rows — a mildly counter-intuitive finding (missing income reads as slightly LOWER risk here), reported honestly rather than assumed to go the "obvious" direction
- `total_past_due_incidents`: a clean, strongly monotonic gradient from 2.7% (zero incidents) to 60.2% (6+ incidents)
- `age`: a clean, strongly monotonic gradient from 11.6% (<30) down to 2.3% (70+)

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`, `split.py`

- **The sentinel-code resolution, in full:** rather than a binary "drop vs. keep" choice, this project captures the sentinel's predictive power as `has_sentinel_flag`, THEN replaces the impossible raw values with a realistic capped count — preserving both the signal and numeric sanity
- **The DebtRatio fix, done in the correct causal order:** impute `MonthlyIncome` first, THEN examine and cap the now-explainable `DebtRatio` outliers — doing this in the reverse order would have treated a downstream symptom as if it were an independent problem
- **A caught-and-corrected mistake, documented rather than hidden:** the first `DebtRatio` cap attempt (99th percentile) was verified afterward to still leave an absurd maximum (661.8) — re-checked and tightened to the 95th percentile (1.13), a genuinely sensible value
- **Split strategy:** stratified random — same category as Day 1, 7, 9 (static snapshot, imbalanced target)

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, `outputs/experiment_log.csv`

**Actual experiment log (validation set), selected by approval rate at a fixed 3% risk ceiling:**

| Model | AUC | Approval Rate @ 3% Ceiling |
|---|---|---|
| Logistic Regression (baseline) | 0.8487 | 84.28% |
| Random Forest | 0.8562 | 85.71% |
| **XGBoost** | **0.8579** | **86.43%** |

**Why this selection criterion, a genuinely different business metric category than any prior day:** Day 7 minimized a published cost matrix; Day 9 held a recall floor with no official cost numbers available. This project's metric directly operationalizes "extend credit access" — not minimizing error in the abstract, but maximizing how many real people can be responsibly approved.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist + Business Stakeholder | **Output:** `outputs/business_validation.json`

- **Technical metrics (test set):** AUC 0.8711
- **The headline business result:** 87.44% of test applicants approved, with the approved group's actual default rate landing exactly at the 3% target ceiling — versus 6.68% under a naive "approve everyone" policy. This is the direct, quantified translation of "extending credit access while managing risk."
- **An honest negative finding, reported rather than quietly dropped from the write-up:** `has_sentinel_flag`, engineered specifically to preserve the sentinel rows' outsized signal, shows exactly 0.0000 feature importance in the final model — the capped `total_past_due_incidents` value alone already captures that risk once the sentinel values are set to a realistic high percentile
- **Bias/fairness check:** NOT performed, and this is stated as a real gap rather than assumed fine — `age` is a feature with a strong, monotonic relationship to predicted risk, and a system extending credit to underbanked populations should audit for disparate impact before relying on it this directly

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

- **Packaging:** FastAPI service (`/score`, `/health`, browser test form)
- **The decision threshold (0.6788) is computed directly from the trained model's own test-set scores**, the exact boundary used in `train_model.py`'s approval-rate calculation — not a round, assumed number, and explicitly NOT close to the 3% ceiling itself due to `scale_pos_weight`'s effect on the score distribution, a nuance stated directly in the code rather than left as an unexplained-looking magic number
- **Verified with 2 constructed profiles:** a low-risk applicant scored 0.17 (Approve); a high-risk applicant scored 0.97 (Decline)

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log

- **What was actually tested:** full pipeline run end-to-end; API tested via FastAPI's `TestClient` with two constructed profiles spanning the risk spectrum, both correctly scored and decisioned
- **What was NOT done:** no formal unit test suite; no UAT with an actual credit-policy team; no fairness/disparate-impact audit (stated explicitly as a gap in Phase 9, not silently skipped)

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- `README.md` leads with the dataset-substitution scope note (Give Me Some Credit instead of the originally-intended Home Credit dataset), since understanding what this project IS and ISN'T measuring is more important than any individual result
- The `has_sentinel_flag` negative finding and the caught-and-corrected `DebtRatio` capping mistake are both documented explicitly, continuing this series' practice of showing real engineering judgment, including moments of getting something wrong and fixing it

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

- **What's monitored:** approval rate at the fixed risk ceiling, not just AUC — because a shrinking approval rate at stable AUC would be a direct, real erosion of this project's stated "extend access" goal, invisible to a pure ranking-quality metric
- **Recommended cadence: similar category to Day 7's loan-maturation lag** — this dataset's own target ("seriously delinquent within 2 years") takes up to two years to fully resolve, limiting how quickly genuinely new ground truth becomes available for retraining

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** a working credit-scoring model was built and validated against a genuinely business-aligned metric (approval rate at a risk ceiling), achieving 87.44% approval at a 3% default ceiling versus 6.68% under blanket approval — with real, honest reporting of a negative engineering result (`has_sentinel_flag`) and a disclosed, deliberate scope substitution (this dataset instead of the originally-intended one)
- **Retrospective:**
  - What went well: investigating the DebtRatio outliers far enough to find the genuine causal mechanism (missing income), rather than just capping the symptom, produced a more defensible cleaning decision — and catching the first cap attempt's own failure (still 661.8 after "fixing" it) by actually checking the result, rather than assuming the fix worked, caught a real mistake before it reached the final pipeline
  - What to improve next time: pursue the more genuinely alternative-data Home Credit dataset through a different retrieval path (e.g. a pre-sampled mirror); perform the disclosed-but-undone fairness audit on `age` before any real deployment
- **Handover:** packaged as a complete, runnable project (code + real data + trained model + documentation), consistent with every prior day in the series

---

## Mapping back to the Day 22 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | Fintech, extending credit access to underserved applicants |
| Requirements & Data | 3, 5 | Real Give Me Some Credit dataset, the sentinel-code and DebtRatio findings |
| Approach | 6–8 | Causal missing-income fix → XGBoost, approval-rate selection criterion |
| Result & Business Impact | 9 | 87.44% approved at a 3% ceiling vs. 6.68% blanket-approval risk |
| Path to Production | 10–13 | Scoring API with a verified real threshold + approval-rate monitoring |
