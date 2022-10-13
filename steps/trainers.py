import mlflow
import xgboost as xgb
import pandas as pd
from zenml.client import Client
from zenml.steps import step, Output

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def xgb_trainer_mlflow(dtrain: xgb.DMatrix, dvalid: xgb.DMatrix) -> xgb.core.Booster:
    """Train xgboose

    Args:
        dtrain (xgb.DMatrix): train dataset
        dvalid (xgb.DMatrix): validation dataset (for model training)
        fe_golden_val_df (pd.DataFrame): validation dataset (for model deployment)

    Returns:
        _type_: _description_
    """
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    xgb_params = {
        'min_child_weight': 1, 
        'learning_rate': 0.05, 
        'colsample_bytree': 0.7, 
        'max_depth': 10,
        'subsample': 0.7,
        'n_estimators': 5000,
        'n_jobs': -1, 
        'booster' : 'gbtree', 
        'silent': 1,
        'eval_metric': 'rmse'}

    mlflow.xgboost.autolog()  

    model = xgb.train(xgb_params, dtrain, 700, 
                        watchlist, early_stopping_rounds=100, \
                        maximize=False, verbose_eval=50)
    return model