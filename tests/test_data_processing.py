import os
import json
import pandas as pd
import numpy as np
import pytest

# Import functions and globals from your data_processing module
from src.data_processing import (
    get_project_root,
    load_raw_data,
    merge_data,
    fill_missing_values,
    prepare_data,
    prepare_data_for_production,
    process_data,
    split_tr_data,
    DATA_DIR,
)

# A dummy engineer_features function to override the real one during tests.
def dummy_engineer_features(df, categorical_handling='object_to_category'):
    # For testing, simply return the DataFrame unchanged.
    return df

# Fixture: Create a temporary data directory with minimal dummy CSV files.
@pytest.fixture
def temp_data_dir(tmp_path):
    # Create the directory structure: data/raw and data/processed
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    # Create dummy train_transaction.csv
    train_trans = pd.DataFrame({
        "TransactionID": [1, 2],
        "isFraud": [0, 1],
        "TransactionDT": [100, 200],
        "TransactionAmt": [10.0, 20.0],
        "ProductCD": ["H", "W"]
    })
    train_trans.to_csv(raw_dir / "train_transaction.csv", index=False)

    # Create dummy train_identity.csv
    train_id = pd.DataFrame({
        "TransactionID": [1, 2],
        "DeviceType": ["mobile", "desktop"]
    })
    train_id.to_csv(raw_dir / "train_identity.csv", index=False)

    # Create dummy test_transaction.csv
    test_trans = pd.DataFrame({
        "TransactionID": [3],
        "TransactionDT": [300],
        "TransactionAmt": [30.0],
        "ProductCD": ["H"]
    })
    test_trans.to_csv(raw_dir / "test_transaction.csv", index=False)

    # Create dummy test_identity.csv
    test_id = pd.DataFrame({
        "TransactionID": [3],
        "DeviceType": ["mobile"]
    })
    test_id.to_csv(raw_dir / "test_identity.csv", index=False)

    # Create a dummy sample_submission.csv in raw folder
    sample_submission = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 0, 0]
    })
    sample_submission.to_csv(raw_dir / "sample_submission.csv", index=False)

    return data_dir

# Fixture: Override DATA_DIR in the module with our temporary directory.
@pytest.fixture(autouse=True)
def override_data_dir(temp_data_dir, monkeypatch):
    monkeypatch.setattr("src.data_processing.DATA_DIR", str(temp_data_dir))

# Test get_project_root
def test_get_project_root():
    root = get_project_root()
    assert os.path.isdir(root), "Project root should be a valid directory."

# Test load_raw_data: Check that the CSV files are loaded.
def test_load_raw_data(temp_data_dir):
    train_trans, train_id, test_trans, test_id = load_raw_data(str(temp_data_dir))
    assert not train_trans.empty
    assert "TransactionID" in train_trans.columns
    assert "DeviceType" in train_id.columns
    assert not test_trans.empty
    assert not test_id.empty

# Test merge_data: Ensure merging works on dummy DataFrames.
def test_merge_data():
    train_trans = pd.DataFrame({
        "TransactionID": [1, 2],
        "isFraud": [0, 1]
    })
    train_id = pd.DataFrame({
        "TransactionID": [1, 2],
        "DeviceType": ["mobile", "desktop"]
    })
    df_merged = merge_data(train_trans, train_id)
    assert "DeviceType" in df_merged.columns
    assert len(df_merged) == 2

# Test fill_missing_values: Check that numeric and object columns are filled correctly.
def test_fill_missing_values():
    df = pd.DataFrame({
        "num": [1, np.nan, 3],
        "cat": ["A", None, "B"]
    })
    df_filled = fill_missing_values(df)
    # Numeric NaN replaced with -999, object NaN replaced with "Unknown"
    assert df_filled.loc[1, "num"] == -999
    assert df_filled.loc[1, "cat"] == "Unknown"

# Test split_tr_data: Check that splitting returns the expected sizes.
def test_split_tr_data():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
        "isFraud": [0, 1, 0, 1, 0]
    })
    X_train, X_val, y_train, y_val = split_tr_data(df, target_col="isFraud", test_size=0.4, random_state=42)
    # With 5 rows and test_size 0.4, we expect 3 training and 2 validation rows.
    assert len(X_train) == 3
    assert len(X_val) == 2
    # Ensure "isFraud" is dropped from features
    assert "isFraud" not in X_train.columns

# Test process_data: Override engineer_features with dummy, then check processing.
def test_process_data(monkeypatch):
    # Override engineer_features with dummy that returns df unchanged
    monkeypatch.setattr("src.data_processing.engineer_features", dummy_engineer_features)
    # Create dummy transaction and identity DataFrames
    df_transaction = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [100, 200],
        "ProductCD": ["H", "W"],
        "isFraud": [0, 1]
    })
    df_identity = pd.DataFrame({
        "TransactionID": [1, 2],
        "DeviceType": ["mobile", "desktop"]
    })
    df_processed = process_data(df_transaction, df_identity, categorical_handling='object_to_category')
    # Check that drop_cols are removed and missing values are filled
    for col in ["TransactionID", "TransactionDT"]:
        assert col not in df_processed.columns
    # Check that numeric missing values are filled with -999
    # (There shouldn't be any missing in our dummy data, but we can simulate one)
    df_transaction.loc[0, "ProductCD"] = None  # Force a missing categorical value
    df_processed = process_data(df_transaction, df_identity, categorical_handling='object_to_category')
    assert df_processed.loc[0, "ProductCD"] == "Unknown"

# Test prepare_data: Override load_raw_data and engineer_features.
def test_prepare_data(monkeypatch, temp_data_dir):
    # Override engineer_features with dummy so process_data returns df unchanged after merge and fill_missing
    monkeypatch.setattr("src.data_processing.engineer_features", dummy_engineer_features)
    # Create dummy DataFrames for train and test CSVs and write them into temp_data_dir/raw
    raw_dir = os.path.join(temp_data_dir, "raw")
    train_trans = pd.DataFrame({
        "TransactionID": [1, 2],
        "isFraud": [0, 1],
        "TransactionDT": [100, 200],
        "TransactionAmt": [10.0, 20.0],
        "ProductCD": ["H", "W"]
    })
    train_id = pd.DataFrame({
        "TransactionID": [1, 2],
        "DeviceType": ["mobile", "desktop"]
    })
    test_trans = pd.DataFrame({
        "TransactionID": [3],
        "TransactionDT": [300],
        "TransactionAmt": [30.0],
        "ProductCD": ["H"]
    })
    test_id = pd.DataFrame({
        "TransactionID": [3],
        "DeviceType": ["mobile"]
    })
    train_trans.to_csv(os.path.join(raw_dir, "train_transaction.csv"), index=False)
    train_id.to_csv(os.path.join(raw_dir, "train_identity.csv"), index=False)
    test_trans.to_csv(os.path.join(raw_dir, "test_transaction.csv"), index=False)
    test_id.to_csv(os.path.join(raw_dir, "test_identity.csv"), index=False)
    
    # Call prepare_data, which should create an expected_columns.json file.
    X_train, X_val, y_train, y_val, X_test = prepare_data(categorical_handling='object_to_category')
    
    # Check that expected_columns.json now exists in DATA_DIR
    expected_columns_path = os.path.join(temp_data_dir, "expected_columns.json")
    assert os.path.isfile(expected_columns_path)
    
    # Load expected_columns and check that it matches columns in X_train
    with open(expected_columns_path, "r") as f:
        expected_columns = json.load(f)
    assert expected_columns == list(X_train.columns)
    
    # Check that the split produces non-empty outputs
    assert not X_train.empty and not X_val.empty

# Test prepare_data_for_production: Use dummy expected_columns.json and check reindexing.
def test_prepare_data_for_production(monkeypatch, temp_data_dir):
    # Override engineer_features with dummy
    monkeypatch.setattr("src.data_processing.engineer_features", dummy_engineer_features)
    # Create dummy transaction and identity DataFrames
    df_transaction = pd.DataFrame({
        "TransactionID": [1],
        "TransactionDT": [100],
        "ProductCD": ["H"],
        "isFraud": [0]
    })
    df_identity = pd.DataFrame({
        "TransactionID": [1],
        "DeviceType": ["mobile"]
    })
    # Create dummy expected_columns.json in temp_data_dir
    expected_cols = ["col1", "col2", "col3"]
    expected_columns_path = os.path.join(temp_data_dir, "expected_columns.json")
    with open(expected_columns_path, "w") as f:
        json.dump(expected_cols, f)
    
    df_prod = prepare_data_for_production(df_transaction, df_identity, categorical_handling='object_to_category')
    # Check that df_prod has exactly the expected columns (filled with -999 if missing)
    assert list(df_prod.columns) == expected_cols
    # Check that any missing column value is filled with -999
    for col in expected_cols:
        # Since our dummy df doesn't have these columns, they should be filled with -999
        assert all(df_prod[col] == -999)
