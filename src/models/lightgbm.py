# src/models/lightgbm.py

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from src.utils.helpers import print_evaluation_metrics, save_model
from src.models.config import lightgbm_params

def train_lightgbm(X_train, X_val, y_train, y_val, params=lightgbm_params, num_boost_round=500):
    """
    Trains a LightGBM model on the given training and validation data.
    Returns the trained model and the validation AUC.
    """

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train the model with early stopping
    model = lgb.train(params,
                      train_data,
                      num_boost_round=num_boost_round,
                      valid_sets=[train_data, val_data],
                      callbacks=[
                        early_stopping(stopping_rounds=50),
                        log_evaluation(period=50)
                        ]
                      )
    
    # Predict on the validation set and calculate AUC
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    print_evaluation_metrics(y_val, y_pred)
    save_model(model, "lightgbm_model")
    return model

def predict_lightgbm(model, X):
    """
    Generates predictions using the trained LightGBM model.
    """
    return model.predict(X, num_iteration=model.best_iteration)
