from zenml.steps import step


@step
def deployment_trigger(train_r2: float, validation_r2: float, test_r2: float,\
            train_rmse: float,validaton_rmse: float, test_rmse: float) -> bool:
    """Only deploy if the test r2 > 60%."""
    return test_r2 > 0.6