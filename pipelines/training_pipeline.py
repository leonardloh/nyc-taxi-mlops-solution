from zenml.pipelines import pipeline


@pipeline(enable_cache=False, name="nyc_training_pipeline")
def training_pipeline(
    training_data_loader,
    training_data_preprocessor,
    trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    """Train, evaluate, and deploy a model."""
    train_df, val_df = training_data_loader()
    dtrain, dvalid, dtest, y_train, y_val, y_test =  training_data_preprocessor(train_df, val_df)
    model = trainer(dtrain=dtrain, dvalid=dvalid)
    train_r2, validation_r2, test_r2, train_rmse, validaton_rmse, test_rmse = evaluator(model=model, dtrain=dtrain, dvalid=dvalid, 
                                                                                        dtest=dtest, y_train=y_train,y_val=y_val, y_test=y_test)
    deployment_decision = deployment_trigger(train_r2, validation_r2, test_r2, train_rmse, validaton_rmse, test_rmse)
    model_deployer(deployment_decision, model)