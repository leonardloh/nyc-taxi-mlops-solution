import pandas as pd
from zenml.steps import step, Output
import xgboost as xgb
from model.data_cleaning import (
    get_train_val_goldval_data,
    get_feature_engineered_test_df)

@step
def preprocess_training_data(train_df: pd.DataFrame, val_df: pd.DataFrame)->\
    Output(dtrain=xgb.DMatrix, dvalid=xgb.DMatrix, dtest=xgb.DMatrix, y_train=pd.Series, y_val=pd.Series, y_test=pd.Series):
    dtrain, dvalid, dtest, y_train, y_val, y_test = get_train_val_goldval_data(train_df=train_df, val_df=val_df)
    return dtrain, dvalid, dtest, y_train, y_val, y_test

@step
def preprocess_inference_data(production_df: pd.DataFrame)->pd.DataFrame:
    preprocessed_production_df =  get_feature_engineered_test_df(test_df = production_df)
    return preprocessed_production_df
    