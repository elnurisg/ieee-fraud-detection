import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import LabelEncoder

def transform_transaction_dt(df: pd.DataFrame, ref_date="2017-11-30"):
    """
    Converts TransactionDT (seconds from a reference) to a datetime and extracts time features.
    """
    startdate = dt.datetime.strptime(ref_date, "%Y-%m-%d")
    df["TransactionDT"] = df["TransactionDT"].apply(lambda x: startdate + dt.timedelta(seconds=x))
    df["hour"] = df["TransactionDT"].dt.hour
    df["day_of_week"] = df["TransactionDT"].dt.dayofweek
    df["day_of_month"] = df["TransactionDT"].dt.day
    df["month"] = df["TransactionDT"].dt.month
    return df

def group_email_domains(df: pd.DataFrame):
    """
    Groups email domains into 'main providers' vs. 'other'.
    """
    main_providers = {"gmail.com", "yahoo.com", "hotmail.com"}
    def map_email_provider(domain):
        if pd.isnull(domain):
            return "unknown"
        domain = domain.lower()
        return domain if domain in main_providers else "other"
    
    df["P_emaildomain_group"] = df["P_emaildomain"].apply(map_email_provider)
    df["R_emaildomain_group"] = df["R_emaildomain"].apply(map_email_provider)

    df = df.drop(columns=["P_emaildomain", "R_emaildomain"])
    return df

def process_addresses(df: pd.DataFrame):
    """
    Processes address fields (addr1, addr2) as categorical/numerical features.
    """
    df["addr1"] = df["addr1"].fillna(-999).astype("int")
    df["addr2"] = df["addr2"].fillna(-999).astype("int")
    return df

def process_distances(df: pd.DataFrame):
    """
    Handles distance features: fills missing values and creates missing indicators.
    """
    df["dist1_missing"] = df["dist1"].isnull().astype(int)
    df["dist2_missing"] = df["dist2"].isnull().astype(int)
    
    df["dist1"] = df["dist1"].fillna(-999)
    df["dist2"] = df["dist2"].fillna(-999)
    return df

def process_m_flags(df: pd.DataFrame):
    """
    Converts binary flags M1–M9 to numerical form (except M4 which is multi-class) 
    and creates an aggregated feature.
    """
    m_cols = [col for col in df.columns if col.startswith("M")]
    if "M4" in m_cols:
        m_cols.remove("M4")  # M4 is multi-class; handle separately
    for col in m_cols:
        df[col] = df[col].map({"T": 1, "F": 0})
    df["M_true_count"] = df[m_cols].sum(axis=1)
    return df

def convert_object_to_category(df: pd.DataFrame):
    """
    Converts object columns to 'category' dtype.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

def label_encode_categoricals(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def engineer_features(df: pd.DataFrame):
    """
    Master function that calls all feature engineering functions.
    """
    df = transform_transaction_dt(df)
    df = group_email_domains(df)
    df = process_addresses(df)
    df = process_distances(df)
    df = process_m_flags(df)
    df = label_encode_categoricals(df)
    # Additional transformations can be added here.
    return df
