xgboost_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

lightgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'verbose': -1
}

rf_params = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "class_weight": "balanced"
}
