"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY STRATIFIED RANDOM (same category as Day 1, 7, 9): this is a static
snapshot with no time axis, and a meaningfully imbalanced target
(6.68% delinquency rate) -- stratification preserves that rate across
all three splits, the same reasoning used every other time this series
has faced a static, moderately-imbalanced classification target.
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
