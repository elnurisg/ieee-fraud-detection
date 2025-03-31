import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_processing import *

MODEL_DIR = os.path.abspath("../src/models/saved_models")

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

def print_evaluation_metrics(y_true, y_pred):
    """
    Prints evaluation metrics: AUC, accuracy, precision, recall, and F1 score.
    """
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred > 0.5)
    precision = precision_score(y_true, y_pred > 0.5)
    recall = recall_score(y_true, y_pred > 0.5)
    f1 = f1_score(y_true, y_pred > 0.5)

    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def save_model(model, model_name):
    """
    Saves the trained model to a file.
    """
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.joblib"))
    print(f"Model {model_name} saved to {MODEL_DIR}.")