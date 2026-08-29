# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "sujat-khan"
    repo_name = "MLFlow-DVC-Dagshub-Project"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging (highest version number)
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise ValueError(f"No model versions found in stage 'Staging' for model '{model_name}'")
    latest_version_staging = max(staging_versions, key=lambda v: int(v.version)).version

    # Promote the new model to production and archive current production models
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()