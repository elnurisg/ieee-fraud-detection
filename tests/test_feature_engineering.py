# tests/test_feature_engineering.py

import pandas as pd
from src.feature_engineering import transform_transaction_dt, group_email_domains

def test_transform_transaction_dt():
    # Create a dummy DataFrame with a TransactionDT column (seconds from ref date)
    df = pd.DataFrame({
        "TransactionDT": [86400, 172800]  # 1 day and 2 days in seconds
    })
    # Use a known reference date for reproducibility
    df = transform_transaction_dt(df, ref_date="2020-01-01")
    
    # Check that new time-related columns exist
    for col in ["hour", "day_of_week", "day_of_month", "month"]:
        assert col in df.columns, f"Column '{col}' should be in the DataFrame."
    
    # Ensure TransactionDT is now a datetime dtype
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionDT"]), "TransactionDT should be datetime dtype."

def test_group_email_domains():
    # Create a dummy DataFrame with email domains
    df = pd.DataFrame({
        "P_emaildomain": ["gmail.com", "hotmail.com", "otherdomain.com", None],
        "R_emaildomain": ["yahoo.com", "gmail.com", None, "hotmail.com"]
    })
    df = group_email_domains(df)
    
    # Check that the grouped columns exist
    assert "P_emaildomain_group" in df.columns, "P_emaildomain_group column is missing."
    assert "R_emaildomain_group" in df.columns, "R_emaildomain_group column is missing."
    
    # Optionally, verify that known main providers remain unchanged while others are set to "other"
    expected_p = ["gmail.com", "hotmail.com", "other", "unknown"]
    expected_r = ["yahoo.com", "gmail.com", "unknown", "hotmail.com"]
    assert list(df["P_emaildomain_group"]) == expected_p, "P_emaildomain_group values are not as expected."
    assert list(df["R_emaildomain_group"]) == expected_r, "R_emaildomain_group values are not as expected."
