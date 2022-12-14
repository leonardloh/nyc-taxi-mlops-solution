from zenml.integrations.evidently.steps import (
    EvidentlyProfileParameters,
    evidently_profile_step,
)

evidently_profile_params = EvidentlyProfileParameters(
    profile_sections=["datadrift"],
    verbose_level=1
)
drift_detector = evidently_profile_step(
    step_name="drift_detector", 
    params=evidently_profile_params
)