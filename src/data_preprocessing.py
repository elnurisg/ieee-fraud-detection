import pandas as pd
import os

DATA_DIR = os.path.abspath("../data")

def load_raw_data(data_dir = DATA_DIR):
    """
    Loads the raw CSV files from the specified directory.
    Returns:
        train_transaction, train_identity, test_transaction, test_identity (as DataFrames)
    """
    train_transaction = pd.read_csv(os.path.join(data_dir, "raw/train_transaction.csv"))
    train_identity = pd.read_csv(os.path.join(data_dir, "raw/train_identity.csv"))
    test_transaction = pd.read_csv(os.path.join(data_dir, "raw/test_transaction.csv"))
    test_identity = pd.read_csv(os.path.join(data_dir, "raw/test_identity.csv"))
    
    return train_transaction, train_identity, test_transaction, test_identity

def merge_data(train_transaction: pd.DataFrame, train_identity: pd.DataFrame):
    """
    Merges train_transaction and train_identity on 'TransactionID'.
    """
    df_train = train_transaction.merge(train_identity, on="TransactionID", how="left")
    return df_train

def basic_cleaning(df: pd.DataFrame):
    """
    Performs basic cleaning, e.g. filling missing values.
    """
    # Fill numeric columns with -999 and categorical with "Unknown"
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-999)
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return df
