from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

if __name__ == "__main__":
    mlflow_model_deployer_component = (
        MLFlowModelDeployer.get_active_model_deployer()
    )
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="nyc_inference_pipeline",
        pipeline_step_name="mlflow_model_deployer_step"
    )
    existing_service = existing_services[0]