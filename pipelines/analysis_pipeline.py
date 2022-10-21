from zenml.pipelines import pipeline

@pipeline(enable_cache=False, name="nyc_analysis_pipeline")
def analysis_pipeline(
    inference_data_loader,
    training_data_loader,
    drift_detector,
):
    inference_df = inference_data_loader()
    training_df = training_data_loader()
    drift_detector(training_df, inference_df)