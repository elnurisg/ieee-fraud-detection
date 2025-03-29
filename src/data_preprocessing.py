import pandas as pd
import os

DATA_DIR = os.path.abspath("../data")

def load_merged_raw_data(data_dir = DATA_DIR):
    """
    Loads the train_transaction and train_identity CSVs from the given directory,
    merges them on 'TransactionID' (left join), and returns the combined dataframe.
    """
    train_transaction_path = os.path.join(data_dir, "raw/train_transaction.csv")
    train_identity_path = os.path.join(data_dir, "raw/train_identity.csv")
    
    # Read CSVs
    train_transaction = pd.read_csv(train_transaction_path)
    train_identity = pd.read_csv(train_identity_path)
    
    # Merge
    df = train_transaction.merge(train_identity, on="TransactionID", how="left")
    return df

def load_data(file_name, data_dir = DATA_DIR):

    data_path = os.path.join(data_dir, file_name)
    
    # Read CSVs
    df = pd.read_csv(data_path)
    return df