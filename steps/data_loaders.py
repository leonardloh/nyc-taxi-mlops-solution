import pandas as pd
from zenml.steps import step, Output

@step
def get_train_test_df() -> Output(train_df = pd.DataFrame, val_df = pd.DataFrame):
    train_df = pd.read_parquet("data/train.parquet")
    val_df = pd.read_parquet("data/val.parquet")
    return train_df, val_df