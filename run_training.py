from pipelines.training_pipeline import training_pipeline
from steps import (
    data_loaders,
    data_preprocessing,
    trainers,
    evaluators,
    deployment_triggers,
    model_deployers
)
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import logging

def main():
    training_pipeline_instance = training_pipeline(
        training_data_loader = data_loaders.get_train_test_df(),
        training_data_preprocessor = data_preprocessing.preprocess_training_data(),
        trainer = trainers.xgb_trainer_mlflow(),
        evaluator = evaluators.evaluator(),
        deployment_trigger = deployment_triggers.deployment_trigger(),
        model_deployer = model_deployers.model_deployer,
    )
    training_pipeline_instance.run()

    logging.info(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `nyc_training_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

if __name__ == "__main__":
    main()   