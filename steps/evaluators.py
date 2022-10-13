import logging

import pandas as pd
from zenml.steps import step, Output
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

@step
def evaluator(model: xgb.core.Booster, dtrain: xgb.DMatrix, dvalid: xgb.DMatrix, dtest: xgb.DMatrix, 
                y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) ->\
     Output(train_r2=float, validation_r2=float, test_r2=float,\
            train_rmse=float,validaton_rmse=float, test_rmse=float):
    """
    Evaluate model
    """
    y_train_pred = model.predict(dtrain)
    y_valid_pred = model.predict(dvalid)
    y_test_pred = model.predict(dtest)
    train_r2 = r2_score(y_train_pred, y_train)
    validation_r2 = r2_score(y_val, y_valid_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = float(np.sqrt(mean_squared_error(y_train_pred, y_train)))
    validaton_rmse = float(np.sqrt(mean_squared_error(y_valid_pred, y_val)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test_pred, y_test)))
    logging.info(f'Train r2 score: {train_r2}')
    logging.info(f'Validation r2 score: {validation_r2}')
    logging.info(f'Test r2 score: {test_r2}')
    logging.info(f'Train RMSE: {train_rmse:.4f}')
    logging.info(f'Validation RMSE: {validaton_rmse:.4f}')
    logging.info(f'Test RMSE: {test_rmse:.4f}')
    return train_r2, validation_r2, test_r2, train_rmse, validaton_rmse, test_rmse
