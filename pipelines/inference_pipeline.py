from zenml.pipelines import pipeline

@pipeline(enable_cache=False, name="nyc_inference_pipeline")
def inference_pipeline(
    inference_data_loader,
    inference_data_preprocessor,
    prediction_service_loader,
    predictor,
    training_data_loader,
    drift_detector,
):
    """Inference pipeline with skew and drift detection."""
    inference_df = inference_data_loader()
    preprocessed_production_df = inference_data_preprocessor(inference_df)
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, preprocessed_production_df)
    training_df = training_data_loader()
    drift_detector(training_df, inference_df)