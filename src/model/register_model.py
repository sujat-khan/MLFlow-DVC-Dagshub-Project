# register model

import json
import mlflow
import logging
import os
import dagshub

# Set up DagsHub credentials for MLflow tracking

import dagshub

# mlflow.set_tracking_uri('https://dagshub.com/sujat-khan/MLFlow-DVC-Dagshub-Project.mlflow')
# dagshub.init(repo_owner='sujat-khan', repo_name='MLFlow-DVC-Dagshub-Project', mlflow=True)

# Set up DagsHub credentials for MLflow tracking, to be used when Dagshub token is used and save in github repo env
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


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        client = mlflow.tracking.MlflowClient()
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        model_uri = f"runs:/{run_id}/{model_path}"
        
        version = None
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            version = model_version.version
        except Exception as reg_err:
            logger.warning('mlflow.register_model fallback triggered: %s', reg_err)
            run = client.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            source = f"{artifact_uri}/{model_path}"
            
            try:
                client.create_registered_model(model_name)
            except Exception:
                pass
            
            # Check if version for this run already exists
            existing_versions = client.search_model_versions(f"name='{model_name}'")
            for v in existing_versions:
                if v.run_id == run_id:
                    version = v.version
                    break
            
            if not version:
                created = client.create_model_version(
                    name=model_name,
                    source=source,
                    run_id=run_id
                )
                version = created.version
        
        # Transition the model to "Staging" stage and archive older staging versions
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True
        )
        
        logger.debug(f'Model {model_name} version {version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()