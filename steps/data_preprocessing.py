import pandas as pd
from model.data_cleaning import DataCleaning
from zenml.steps import step, Output
import xgboost as xgb

@step
def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame)->\
    Output(dtrain=xgb.DMatrix, dvalid=xgb.DMatrix, dtest=xgb.DMatrix, y_train=pd.Series, y_val=pd.Series, y_test=pd.Series):
    dc = DataCleaning(train_df=train_df, val_df=val_df)
    dtrain, dvalid, dtest, y_train, y_val, y_test = dc.get_train_val_goldval_data()
    return dtrain, dvalid, dtest, y_train, y_val, y_test