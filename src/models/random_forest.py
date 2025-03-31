# src/models/random_forest.py

from sklearn.ensemble import RandomForestClassifier
from src.utils.helpers import print_evaluation_metrics, save_model
from src.models.config import rf_params

def train_random_forest(X_train, X_val, y_train, y_val, params=rf_params):
    """
    Trains a Random Forest model and evaluates it on the validation set.
    """
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_val = model.predict_proba(X_val)[:, 1]

    print_evaluation_metrics(y_val, y_pred_val)
    save_model(model, "random_forest_model")

    return model

def predict_random_forest(model, X):
    """
    Predict using the trained Random Forest model.
    """
    return model.predict_proba(X)[:, 1]
