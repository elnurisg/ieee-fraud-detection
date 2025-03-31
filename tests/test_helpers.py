import os
import tempfile
import pandas as pd
import numpy as np
import joblib
import json
import pytest

from src.utils.helpers import (
    save_processed_data,
    read_processed_data,
    print_evaluation_metrics,
    save_model,
    load_model,
    prepare_my_submission,
)

# Fixture to create a temporary DATA_DIR structure
@pytest.fixture
def temp_data_dir(tmp_path):
    # Create a temporary 'data' directory with 'raw' and 'processed' subdirectories.
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    # Create a dummy sample_submission.csv in the raw directory
    sample_submission = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 0, 0]
    })
    sample_submission.to_csv(raw_dir / "sample_submission.csv", index=False)
    return data_dir

# Fixture to override DATA_DIR and MODEL_DIR in your helpers module
@pytest.fixture(autouse=True)
def override_dirs(temp_data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.helpers.DATA_DIR", str(temp_data_dir))
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    monkeypatch.setattr("src.utils.helpers.MODEL_DIR", str(model_dir))

def test_save_and_read_processed_data():
    # Create dummy DataFrames for processed data
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    X_val = pd.DataFrame({"a": [4, 5]})
    y_train = pd.Series([0, 1, 0], name="isFraud")
    y_val = pd.Series([1, 0], name="isFraud")
    X_test = pd.DataFrame({"a": [6, 7]})
    
    # Save the processed data
    save_processed_data(X_train, X_val, y_train, y_val, X_test)
    
    # Read the data back; now y_train and y_val should be Series
    X_train_r, X_val_r, y_train_r, y_val_r, X_test_r = read_processed_data()
    
    # Drop "Unnamed: 0" if present in X_train_r, X_val_r, X_test_r
    for df in [X_train_r, X_val_r, X_test_r]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    
    pd.testing.assert_frame_equal(X_train, X_train_r)
    pd.testing.assert_frame_equal(X_val, X_val_r)
    pd.testing.assert_frame_equal(X_test, X_test_r)
    pd.testing.assert_series_equal(y_train, y_train_r, check_names=False)
    pd.testing.assert_series_equal(y_val, y_val_r, check_names=False)

def test_print_evaluation_metrics(capsys):
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.2]
    print_evaluation_metrics(y_true, y_pred)
    captured = capsys.readouterr().out
    for label in ["AUC:", "Accuracy:", "Precision:", "Recall:", "F1 Score:"]:
        assert label in captured

def test_save_and_load_model(tmp_path):
    dummy_model = {"dummy": "model"}
    model_name = "test_model"
    save_model(dummy_model, model_name)
    loaded_model = load_model(model_name)
    assert loaded_model == dummy_model

def test_prepare_my_submission(temp_data_dir):
    y_pred = [0.3, 0.7, 0.2]
    model_name = "test_submission"
    submission = prepare_my_submission(y_pred, model_name, use_threshold=False)
    assert "TransactionID" in submission.columns
    assert "isFraud" in submission.columns
    assert len(submission) == 3
