# src/models/lightgbm.py

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

def train_lightgbm(X_train, X_val, y_train, y_val, params=None, num_boost_round=500):
    """
    Trains a LightGBM model on the given training and validation data.
    Returns the trained model and the validation AUC.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'verbose': -1
        }

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
    val_auc = roc_auc_score(y_val, y_pred)
    
    return model, val_auc

def predict_lightgbm(model, X):
    """
    Generates predictions using the trained LightGBM model.
    """
    return model.predict(X, num_iteration=model.best_iteration)
