# src/models/xgboost.py
import xgboost as xgb
from src.utils.helpers import print_evaluation_metrics, save_model
from src.models.config import xgboost_params

def train_xgboost_model(X_train, X_val, y_train, y_val, params=xgboost_params, num_boost_round=500, early_stopping_rounds=50):
    """
    Trains an XGBoost model with the given training data and parameters.
    Returns the trained model and the validation AUC.
    """    
    
    # Create DMatrix with enable_categorical=True
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, 
                      evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=50)
    
    if hasattr(model, "best_ntree_limit"):
        y_pred_val = model.predict(dval, ntree_limit=model.best_ntree_limit)
    else:
        y_pred_val = model.predict(dval)  

    # Calculate evaluation metrics
    print_evaluation_metrics(y_val, y_pred_val)
    save_model(model, "xgboost_model")
    return model

def predict_xgboost(model, X):
    """
    Generates predictions using the trained XGBoost model.
    """
    dX = xgb.DMatrix(X, enable_categorical=True)

    if hasattr(model, "best_ntree_limit"):
        y_pred = model.predict(dX, ntree_limit=model.best_ntree_limit)
    else:
        y_pred = model.predict(dX)  

    return y_pred
