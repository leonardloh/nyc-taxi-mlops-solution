from pipelines.inference_pipeline import inference_pipeline
from steps.data_loaders import get_production_df, get_train_df_evidently
from steps.prediction_steps import prediction_service_loader, predictor
from steps.drift_detection import drift_detector
from steps.data_preprocessing import preprocess_inference_data
from zenml.integrations.evidently.visualizers.evidently_visualizer import EvidentlyVisualizer
from zenml.post_execution import get_pipeline, get_pipelines

# initialize and run the inference pipeline
def main():
    inference_pipeline_instance = inference_pipeline(
        inference_data_loader=get_production_df(),
        inference_data_preprocessor=preprocess_inference_data(),
        prediction_service_loader=prediction_service_loader(),
        predictor=predictor(),
        training_data_loader=get_train_df_evidently(),
        drift_detector=drift_detector,
    )
    inference_pipeline_instance.run()

    # inf_run = inference_pipeline_instance.get_runs()[-1]
    # drift_detection_step = inf_run.get_step(step="drift_detector")
    # EvidentlyVisualizer().visualize(drift_detection_step)

    # pipeline = get_pipeline(pipeline="nyc_training_pipeline")
    # evidently_outputs = pipeline.runs[-1].get_step(step="drift_detector")
    # EvidentlyVisualizer().visualize(evidently_outputs)

if __name__ == "__main__":
    main()