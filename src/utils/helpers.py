import pandas as pd
import os
from src.data_preprocessing import *
from src.feature_engineering import engineer_features
from sklearn.model_selection import train_test_split

drop_cols = ["TransactionID", "TransactionDT"]

def prepare_data():
    """
    Loads, merges, cleans, and applies feature engineering to the training data.
    Returns the processed DataFrame.
    """
    train_trans, train_id, test_trans, test_id = load_raw_data()
    df_train = merge_data(train_trans, train_id)
    X_test = merge_data(test_trans, test_id)
    df_train = engineer_features(df_train)
    X_test = engineer_features(X_test)
    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    df_train = basic_cleaning(df_train)
    X_test = basic_cleaning(X_test)

    X_train, X_val, y_train, y_val = split_data(df_train)

    # Correct the column names in X_test to match those in X_train
    fix = {o:n for o, n in zip(X_test.columns, X_train.columns)}
    X_test.rename(columns=fix, inplace=True)

    return X_train, X_val, y_train, y_val, X_test

def save_processed_data(X_train, X_val, y_train, y_val, X_test):
    """
    Saves the processed data to CSV files.
    """
    X_train.to_csv(os.path.join(DATA_DIR, "processed/X_train.csv"))
    X_val.to_csv(os.path.join(DATA_DIR, "processed/X_val.csv"))
    y_train.to_csv(os.path.join(DATA_DIR, "processed/y_train.csv"))
    y_val.to_csv(os.path.join(DATA_DIR, "processed/y_val.csv"))
    X_test.to_csv(os.path.join(DATA_DIR, "processed/X_test.csv"))
    # Save the processed data
    print("Processed data saved to CSV files.")
    return X_train, X_val, y_train, y_val, X_test

def read_processed_data():
    """
    Reads the processed data from CSV files.
    """
    X_train = pd.read_csv(os.path.join(DATA_DIR, "processed/X_train.csv"))
    X_val = pd.read_csv(os.path.join(DATA_DIR, "processed/X_val.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "processed/y_train.csv"))
    y_val = pd.read_csv(os.path.join(DATA_DIR, "processed/y_val.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "processed/X_test.csv"))
    return X_train, X_val, y_train, y_val, X_test

def split_data(df: pd.DataFrame, target_col: str = "isFraud", test_size: float = 0.2, random_state: int = 42):
    """
    Splits the DataFrame into training and validation sets.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val
