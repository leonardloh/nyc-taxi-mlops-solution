import pandas as pd
from model.optimize import optimize_floats_ints
from model.geo_distance import *
from sklearn.model_selection import train_test_split
import xgboost as xgb

SEED = 123

def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Drop na from dropoff latitude and longtitude

    Args:
        df (pd.DataFrame): input df

    Returns:
        pd.DataFrame: cleaned dataframe without na in dropoff latitude and longtitude
    """
    return df.dropna(subset=['dropoff_latitude', 'dropoff_longitude'], axis=0)

def filter_fare_amount(df: pd.DataFrame) -> pd.DataFrame:
    """filter df based on fare amount

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    df = df.drop(df[df['fare_amount'] < 2.5].index, axis=0)
    df = df.drop(df[df['fare_amount'] > 500].index, axis=0)
    return df

def filter_unused_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df[df['pickup_longitude'] == 0].index, axis=0)
    df = df.drop(df[df['pickup_latitude'] == 0].index, axis=0)
    df = df.drop(df[df['dropoff_longitude'] == 0].index, axis=0)
    df = df.drop(df[df['dropoff_latitude'] == 0].index, axis=0)
    df = df.drop(df[df['passenger_count'] == 208].index, axis=0)
    df = df.drop(df[df['passenger_count'] > 5].index, axis=0)
    df = df.drop(df[df['passenger_count'] == 0].index, axis=0)
    return df

def format_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df['key'] = pd.to_datetime(df['key'])
    df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])
    return df

def add_datetime_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['Year'] = df['pickup_datetime'].dt.year
    df['Month'] = df['pickup_datetime'].dt.month
    df['Date'] = df['pickup_datetime'].dt.day
    df['Day of Week'] = df['pickup_datetime'].dt.dayofweek
    df['Hour'] = df['pickup_datetime'].dt.hour
    df = df.drop('pickup_datetime', axis = 1)
    df = df.drop('key', axis = 1)
    return df

def filter_pickup_long_lat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()

    df = df.drop(df.index[(df.pickup_longitude < -75) | 
            (df.pickup_longitude > -72) | 
            (df.pickup_latitude < 40) | 
            (df.pickup_latitude > 42)])
    df = df.drop(df.index[(df.dropoff_longitude < -75) | 
            (df.dropoff_longitude > -72) | 
            (df.dropoff_latitude < 40) | 
            (df.dropoff_latitude > 42)])
    return df

def optimize(df: pd.DataFrame) -> pd.DataFrame:
    return optimize_floats_ints(df)

def calc_dists(df: pd.DataFrame) -> pd.DataFrame:
    df['geodesic'] = df.apply(lambda x: geodesic_dist(x), axis = 1 )
    df['circle'] = df.apply(lambda x: circle_dist(x), axis = 1 )
    df['jfk'] = df.apply(lambda x: jfk_dist(x), axis = 1 )
    df['lga'] = df.apply(lambda x: lga_dist(x), axis = 1 )
    df['ewr'] = df.apply(lambda x: ewr_dist(x), axis = 1 )
    df['tsq'] = df.apply(lambda x: tsq_dist(x), axis = 1 )
    df['cpk'] = df.apply(lambda x: cpk_dist(x), axis = 1 )
    df['lib'] = df.apply(lambda x: lib_dist(x), axis = 1 )
    df['gct'] = df.apply(lambda x: gct_dist(x), axis = 1 )
    df['met'] = df.apply(lambda x: met_dist(x), axis = 1 )
    df['wtc'] = df.apply(lambda x: wtc_dist(x), axis = 1 )
    return df

def get_feature_engineered_train_df(train_df: pd.DataFrame) ->\
    pd.DataFrame:
    """
    Perform feature engineering on train data set
    """
    preprocessed_train_df = drop_na(train_df)
    preprocessed_train_df = filter_fare_amount(preprocessed_train_df)
    preprocessed_train_df = filter_unused_data(preprocessed_train_df)
    preprocessed_train_df = format_datetime(preprocessed_train_df)
    preprocessed_train_df = add_datetime_feature(preprocessed_train_df)
    preprocessed_train_df = filter_pickup_long_lat(preprocessed_train_df)
    preprocessed_train_df = optimize(preprocessed_train_df)
    preprocessed_train_df = calc_dists(preprocessed_train_df)
    preprocessed_train_df = optimize(preprocessed_train_df)
    return preprocessed_train_df

def get_feature_engineered_test_df(test_df: pd.DataFrame)->pd.DataFrame:
        """Perform feature engineering on any data set

        Args:
            test_df (pd.DataFrame): input any dataframe

        Returns:
            pd.DataFrame: preprocessed dataframe
        """
        preprocessed_test_df = format_datetime(test_df)
        preprocessed_test_df = add_datetime_feature(preprocessed_test_df)
        preprocessed_test_df = optimize(preprocessed_test_df)
        preprocessed_test_df = calc_dists(preprocessed_test_df)
        preprocessed_test_df = optimize(preprocessed_test_df)
        return preprocessed_test_df

def get_train_val_goldval_data(train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Get training set, validation set and golden test set.
        """
        fe_train_df, fe_golden_val_df = get_feature_engineered_train_df(train_df=train_df), get_feature_engineered_test_df(test_df=val_df)
        X, y = fe_train_df.drop("fare_amount", axis=1), fe_train_df["fare_amount"]
        X_test, y_test = fe_golden_val_df.drop("fare_amount", axis=1), fe_golden_val_df["fare_amount"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        return dtrain, dvalid, dtest, y_train, y_val, y_test