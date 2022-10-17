import pandas as pd
from zenml.steps import step, Output

@step
def get_train_test_df() -> Output(train_df = pd.DataFrame, val_df = pd.DataFrame):
    train_df = pd.read_parquet("data/train.parquet")
    val_df = pd.read_parquet("data/val.parquet")
    return train_df, val_df

@step
def get_train_df_evidently() -> pd.DataFrame:
    """Get train data without label

    Returns:
        pd.DataFrame: train data without label
    """
    train_df = pd.read_parquet("data/train.parquet").drop('fare_amount', axis=1)
    return train_df

@step
def get_production_df()->pd.DataFrame:
    production_df = pd.read_parquet("data/production.parquet")
    return production_df