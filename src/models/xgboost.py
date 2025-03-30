# src/models/xgboost.py
# src/models/xgboost.py
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def train_xgboost_model(X_train, X_val, y_train, y_val, params=None, num_boost_round=500, early_stopping_rounds=50):
    """
    Trains an XGBoost model with the given training data and parameters.
    Returns the trained model and the validation AUC.
    """
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
    
    
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
          
    val_auc = roc_auc_score(y_val, y_pred_val)
    
    return model, val_auc

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
