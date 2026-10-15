"""
Phases 5 (data quality) & 7 (feature engineering)
------------------------------------------------------
FOUR REAL DATA QUALITY ISSUES, each handled differently -- including one
(the sentinel delinquency code) that is NOT simply dropped, because it
turns out to carry the single strongest signal in the entire dataset.
"""

import numpy as np
import pandas as pd

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: age == 0 (1 row). WHY DROPPED: a borrower cannot genuinely
    # be 0 years old -- an unambiguous data entry error, and a single
    # row has negligible impact on the training population either way.
    # -----------------------------------------------------------------
    df = df[df["age"] > 0].copy()

    # -----------------------------------------------------------------
    # Fix 2: the 96/98 "sentinel" codes in all three past-due columns
    # (269 rows). WHY THIS IS NOT A SIMPLE DROP-OR-CAP DECISION: these
    # values (96 or 98, appearing IDENTICALLY across all three
    # past-due columns in the same row) are not genuine counts -- no
    # real borrower has been "90+ days late 98 times." This is almost
    # certainly a coding artifact from the original data system (a
    # placeholder for "special status" or "data unavailable" accounts).
    #
    # THE KEY FINDING THAT CHANGES THE RIGHT TREATMENT: these 269 rows
    # have a 54.6% delinquency rate, roughly 8x the 6.7% baseline --
    # this is NOT random noise to discard, it's one of the strongest
    # signals in the whole dataset. Simply dropping these rows (the
    # "textbook" outlier response) would throw away real predictive
    # information. Simply leaving the raw 96/98 values in as numeric
    # counts would let a handful of rows disproportionately distort
    # any model sensitive to feature scale (logistic regression
    # especially).
    #
    # THE RESOLUTION: capture the sentinel's predictive power
    # explicitly as its own binary flag (`has_sentinel_flag`), THEN
    # replace the sentinel values themselves with a realistic capped
    # count (the 99th percentile of genuine, non-sentinel values in
    # that column) so the numeric feature no longer contains an
    # impossible value, while the flag preserves exactly the signal
    # that made these rows special in the first place.
    # -----------------------------------------------------------------
    is_sentinel = (df[PAST_DUE_COLUMNS] >= 96).any(axis=1)
    df["has_sentinel_flag"] = is_sentinel.astype(int)

    for col in PAST_DUE_COLUMNS:
        genuine_values = df.loc[~is_sentinel, col]
        cap_value = genuine_values.quantile(0.99)
        df.loc[df[col] >= 96, col] = cap_value

    # -----------------------------------------------------------------
    # Fix 3: MonthlyIncome missing (29,731 rows, ~19.8%) AND the
    # downstream DebtRatio explosion it causes. WHY THESE TWO ARE ONE
    # FIX, NOT TWO: verified that 26,771 of the 28,877 rows with
    # DebtRatio > 10 have a MISSING MonthlyIncome -- meaning the
    # bizarre DebtRatio values (up to 61,106 even among reported-income
    # rows, let alone higher among missing-income ones) aren't really
    # about unusual debt, they're a DOWNSTREAM ARTIFACT of a broken
    # denominator in the original debt-to-income calculation. This is a
    # genuinely CAUSAL data quality finding, not just "here are two
    # separate outlier columns" -- fixing MonthlyIncome's missingness
    # and THEN re-examining DebtRatio is the right order of operations.
    #
    # WHY WE ALSO KEEP AN income_was_missing FLAG: whether income was
    # reported at all may itself be informative (e.g. different
    # employment situations), so this is preserved as a feature rather
    # than silently erased by the imputation.
    # -----------------------------------------------------------------
    df["income_was_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # -----------------------------------------------------------------
    # Capping DebtRatio: WHY THE 95th PERCENTILE, NOT THE 99th.
    # Checked both: among rows with genuinely reported income, the 95th
    # percentile is 1.13 (a highly leveraged but entirely plausible
    # debt-to-income ratio), while the 99th percentile is STILL 661.8 --
    # meaning even the "has income" population contains a residual group
    # of clearly erroneous values the missing-income fix doesn't touch.
    # This means there are actually TWO separate sources of DebtRatio
    # corruption in this dataset (missing income, and a smaller set of
    # other data errors even among reported-income rows) -- using the
    # tighter, genuinely-informative 95th percentile as the cap is what
    # actually fixes both, where the looser 99th percentile would not.
    # -----------------------------------------------------------------
    reasonable_cap = df.loc[df["income_was_missing"] == 0, "DebtRatio"].quantile(0.95)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=reasonable_cap)

    # -----------------------------------------------------------------
    # Fix 4: RevolvingUtilizationOfUnsecuredLines extreme values (max
    # 50,708 -- a ratio, which should essentially never exceed a small
    # multiple of 1). WHY CAPPED, NOT DROPPED: 241 rows exceed 10, almost
    # certainly a unit error in the source system (e.g. a raw dollar
    # balance entered where a ratio was expected) -- capping at a
    # generous but realistic ceiling preserves these rows' OTHER
    # information rather than discarding them entirely.
    # -----------------------------------------------------------------
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=5)

    # -----------------------------------------------------------------
    # Fix 5: NumberOfDependents missing (3,924 rows, ~2.6%). WHY
    # IMPUTED WITH 0, NOT THE MEDIAN: the column's own mode is 0 (no
    # dependents) by a wide margin -- using 0 reflects the single most
    # likely true value for a missing entry, a more defensible choice
    # here than a dataset-wide median/mean for a count variable this
    # concentrated at zero.
    # -----------------------------------------------------------------
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # WHY log1p ON MonthlyIncome: same heavy right-skew discipline as
    # every prior monetary feature in this series.
    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])

    # -----------------------------------------------------------------
    # Engineered feature: total_past_due_incidents
    # BUSINESS LOGIC: the three separate past-due-severity columns
    # (30-59, 60-89, 90+ days) are individually noisy and correlated --
    # summing them into one "how many total delinquency incidents of any
    # severity" measure gives a more stable aggregate signal, computed
    # AFTER the sentinel-code fix above so it isn't distorted by the
    # impossible raw 96/98 values.
    # -----------------------------------------------------------------
    df["total_past_due_incidents"] = df[PAST_DUE_COLUMNS].sum(axis=1)

    return df


def get_feature_columns() -> list:
    return [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome_log", "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines", "NumberOfDependents",
        "total_past_due_incidents", "has_sentinel_flag", "income_was_missing",
    ]


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns()

    print(f"Raw rows: {len(raw)}  ->  Cleaned rows: {len(engineered)}")
    print(f"Remaining DebtRatio max after cleaning: {engineered['DebtRatio'].max():.2f}")
    print(f"Remaining RevolvingUtilization max after cleaning: {engineered['RevolvingUtilizationOfUnsecuredLines'].max():.2f}")
    print(f"\nFeature columns: {feature_cols}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
