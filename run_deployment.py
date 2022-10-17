from pipelines.inference_pipeline import inference_pipeline
from steps.data_loaders import get_production_df, get_train_df_evidently
from steps.prediction_steps import prediction_service_loader, predictor
from steps.drift_detection import drift_detector
from steps.data_preprocessing import preprocess_inference_data
from zenml.integrations.evidently.visualizers.evidently_visualizer import EvidentlyVisualizer

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
    inference_pipeline_instance.run(unlisted=True)

    inf_run = inference_pipeline_instance.get_runs()[-1]
    drift_detection_step = inf_run.get_step(step="drift_detector")
    EvidentlyVisualizer().visualize(drift_detection_step)

if __name__ == "__main__":
    main()