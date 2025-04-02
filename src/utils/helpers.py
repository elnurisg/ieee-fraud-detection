import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        return os.path.abspath(os.path.join(base_dir, "../.."))
    except NameError:
        # __file__ is not defined in a notebook, so use current working directory.
        return os.getcwd()

PROJECT_ROOT = get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "saved_models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Get the project root directory dynamically
PROJECT_ROOT = get_project_root()

# Build an absolute path to your saved models folder
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "saved_models")

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
    y_train = pd.read_csv(os.path.join(DATA_DIR, "processed/y_train.csv"), index_col=0).squeeze()
    y_val = pd.read_csv(os.path.join(DATA_DIR, "processed/y_val.csv"), index_col=0).squeeze()
    X_test = pd.read_csv(os.path.join(DATA_DIR, "processed/X_test.csv"))
    return X_train, X_val, y_train, y_val, X_test

def print_evaluation_metrics(y_true, y_pred):
    """
    Prints evaluation metrics: AUC, accuracy, precision, recall, and F1 score.
    """
    y_pred = np.array(y_pred)
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

def load_model(model_name):
    """
    Loads a trained model from a file.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model {model_name} loaded from {MODEL_DIR}.")
        return model
    else:
        print(f"Model {model_name} not found in {MODEL_DIR}.")
        return None

def prepare_my_submission(y_pred, model_name, use_threshold=False):
    sample_submission = pd.read_csv(os.path.join(DATA_DIR, "raw/sample_submission.csv"))

    # If it's probabilities, you can threshold it (e.g., at 0.5):
    if use_threshold:
        y_pred = (y_pred > 0.5).astype(int)

    # Assign predictions
    sample_submission["isFraud"] = y_pred

    # Save to CSV
    sample_submission.to_csv(DATA_DIR + f"/submission_{model_name}.csv", index=False)
    return sample_submission


def cross_validate_model(X, y, train_func, predict_func, k=5, threshold=0.5, **train_kwargs):
    """
    Performs K-fold cross-validation for a given model and returns evaluation metrics.
    
    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.
        train_func (function): Function to train the model. It should accept 
            (X_train, X_val, y_train, y_val, **train_kwargs) and return a trained model.
        predict_func (function): Function to generate predictions from the model.
            It should accept (model, X) and return predicted probabilities.
        k (int): Number of folds (default is 5).
        threshold (float): Threshold to convert probabilities into class labels (default is 0.5).
        train_kwargs: Additional keyword arguments to pass to the training function.
    
    Returns:
        models (list): List of trained models (one per fold).
        fold_metrics (list): List of dictionaries containing evaluation metrics for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    models = []
    fold_metrics = []
    fold_no = 1
    
    for train_idx, val_idx in kf.split(X):
        print(f"Fold {fold_no}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train the model using the provided training function.
        model = train_func(X_train, X_val, y_train, y_val, **train_kwargs)
        
        # Generate predicted probabilities on the validation set.
        y_pred_prob = predict_func(model, X_val)
        
        # Convert predicted probabilities to class labels based on the threshold.
        y_pred_class = (y_pred_prob >= threshold).astype(int)
        
        # Calculate evaluation metrics.
        auc     = roc_auc_score(y_val, y_pred_prob)
        acc     = accuracy_score(y_val, y_pred_class)
        prec    = precision_score(y_val, y_pred_class, zero_division=0)
        recall  = recall_score(y_val, y_pred_class, zero_division=0)
        f1      = f1_score(y_val, y_pred_class, zero_division=0)
        
        print(f"Fold {fold_no} Metrics:")
        print(f"  AUC:      {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision:{prec:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}\n")
        
        # Store metrics for the current fold.
        fold_metrics.append({
            'auc': auc,
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1
        })
        models.append(model)
        fold_no += 1
    
    # Compute average metrics over all folds.
    avg_metrics = {
        'auc': np.mean([m['auc'] for m in fold_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
    }
    
    print("Average Metrics over {} folds:".format(k))
    print(f"  AUC:      {avg_metrics['auc']:.4f}")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  Precision:{avg_metrics['precision']:.4f}")
    print(f"  Recall:   {avg_metrics['recall']:.4f}")
    print(f"  F1 Score: {avg_metrics['f1']:.4f}\n")
    
    return models, fold_metrics, avg_metrics