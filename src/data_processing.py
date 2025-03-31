import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from src.feature_engineering import engineer_features

def get_project_root():
    """
    Returns the project root directory.
    If __file__ is available (e.g., running from a module), it uses that;
    otherwise (e.g., in a notebook), it falls back to os.getcwd().
    Adjust the number of ".." levels as needed.
    """
    try:
        # When running as a module, __file__ is defined.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Suppose helpers.py is in: <project_root>/src/utils/
        # Then going up two levels should get you to the project root.
        return os.path.abspath(os.path.join(base_dir, ".."))
    except NameError:
        # __file__ is not defined in a notebook, so use current working directory.
        return os.getcwd()

PROJECT_ROOT = get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "saved_models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

drop_cols = ["TransactionID", "TransactionDT"]

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

def fill_missing_values(df: pd.DataFrame):
    """
    Filling missing values.
    """
    # Fill numeric columns with -999 and categorical with "Unknown"
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-999)
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return df

def prepare_data(categorical_handling = 'object_to_category'):
    """
    Loads, merges, cleans, and applies feature engineering to the training data.
    Returns the processed DataFrame.
    """
    train_trans, train_id, test_trans, test_id = load_raw_data()
    
    df_train = process_data(train_trans, train_id, categorical_handling)
    X_test = process_data(test_trans, test_id, categorical_handling)

    X_train, X_val, y_train, y_val = split_tr_data(df_train)

    # Correct the column names in X_test to match those in X_train
    fix = {o:n for o, n in zip(X_test.columns, X_train.columns)}
    X_test.rename(columns=fix, inplace=True)

    # Save the expected columns from training
    expected_columns = list(X_train.columns)
    with open(os.path.join(DATA_DIR, 'expected_columns.json'), 'w') as f:
        json.dump(expected_columns, f)


    return X_train, X_val, y_train, y_val, X_test

def prepare_data_for_production(df_transaction, df_identity, categorical_handling = 'object_to_category'):
    """
    Processes the input data for production use.
    """
    df = process_data(df_transaction, df_identity, categorical_handling)

    with open(os.path.join(DATA_DIR, "expected_columns.json"), 'r') as f:
        expected_columns = json.load(f)
  
    df = df.reindex(columns=expected_columns, fill_value=-999)

    return df

def process_data(df_transaction, df_identity, categorical_handling = 'object_to_category'):
    """
    Processes the data by merging, cleaning, and applying feature engineering.
    Returns the processed DataFrame.
    """
    df = merge_data(df_transaction, df_identity)
    df = engineer_features(df, categorical_handling)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = fill_missing_values(df)
    return df

def split_tr_data(df: pd.DataFrame, target_col: str = "isFraud", test_size: float = 0.2, random_state: int = 42):
    """
    Splits the DataFrame into training and validation sets.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val
