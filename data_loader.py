"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the "Give Me Some Credit" dataset, originally a 2011
Kaggle competition sponsored to "improve on the state of the art in
credit scoring by predicting the probability that somebody will
experience financial distress in the next two years":
    https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

150,000 real, anonymized consumer credit records: revolving credit
utilization, age, late-payment history, debt-to-income ratio, monthly
income, open credit lines, and dependents, with a 2-year serious-
delinquency label.

WHY THIS DATASET FOR DAY 22 (Alternative Credit Scoring, Fintech)
----------------------------------------------------------------------
- Its OWN stated purpose is extending and improving credit decisions
  using behavioral repayment and utilization history, rather than a
  single opaque bureau score -- a direct match for the "non-traditional
  data, extend credit access" framing this project is built around.
- A NOTE ON SCOPE, STATED UPFRONT: the more explicitly "alternative
  data" dataset for this exact business problem is Home Credit Group's
  much larger "Home Credit Default Risk" Kaggle dataset (which includes
  genuinely non-bureau signals like occupation type, document
  completeness, and social-circle default rates). That dataset's raw
  files were not available as a complete, reliably-hostable mirror at
  the size this project's tooling could retrieve (the main table alone
  is ~170MB, too large for a standard GitHub file). This project uses
  "Give Me Some Credit" instead -- still real behavioral credit data,
  still aligned with the same underlying business problem, but a
  smaller, more tractable dataset. This is disclosed honestly rather
  than presented as if it were the more elaborate alternative-data source.
- It has several real, well-documented data quality issues -- including
  a GENUINELY causal one (see clean_and_engineer.py's DebtRatio fix,
  where a missing-income problem explains an otherwise bizarre outlier
  pattern) rather than an issue that's simply noted and dropped.

If you're following along on Kaggle: download `cs-training.csv` from
the link above and place it at `data/cs-training.csv` -- identical
schema to the file already included here.
"""

import pandas as pd

RAW_DATA_PATH = "data/cs-training.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} credit records, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Serious delinquency rate: {df['SeriousDlqin2yrs'].mean():.2%}")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nRows with age == 0 (implausible): {(df['age']==0).sum()}")
    sentinel = (df["NumberOfTimes90DaysLate"] >= 96)
    print(f"Rows with a 96/98 sentinel code in past-due columns: {sentinel.sum()}")
    print(f"  Delinquency rate among those rows: {df.loc[sentinel, 'SeriousDlqin2yrs'].mean():.1%} "
          f"(vs. {df['SeriousDlqin2yrs'].mean():.1%} overall -- not random noise)")
    extreme_debt = df["DebtRatio"] > 10
    print(f"Rows with DebtRatio > 10: {extreme_debt.sum()}, of which "
          f"{df.loc[extreme_debt, 'MonthlyIncome'].isnull().sum()} have missing MonthlyIncome")
