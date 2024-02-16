import json
from typing import Dict, List

import mlflow
from mlflow.entities.run import Run
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient


def load_production_model(tracking_uri: str, model_name: str) -> PyFuncModel:
    """
    Loads the model deployed in the 'Production' stage from the specified tracking URI.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.

    Returns:
    - PyFuncModel: The PyFuncModel representing the loaded production model.
    """
    return load_model_by_stage(tracking_uri, model_name, "Production")


def load_staging_model(tracking_uri: str, model_name: str) -> PyFuncModel:
    """
    Loads the model deployed in the 'Staging' stage from the specified tracking URI.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.

    Returns:
    - PyFuncModel: The PyFuncModel representing the loaded staging model.
    """
    return load_model_by_stage(tracking_uri, model_name, "Staging")


def get_latest_model_versions(tracking_uri: str, model_name: str) -> List[Dict]:
    """
    Retrieves the latest model versions and their stages for a specified model.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to retrieve latest versions for.

    Returns:
    - List[Dict]: A list of dictionaries containing version and stage information
                  for the latest versions of the specified model.
                  Example: [{"version": "1", "stage": "Production"}, ...]
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    latest_versions = client.get_latest_versions(name=model_name)
    return [{"version": version.version, "stage": version.current_stage} for version in latest_versions]


def transition_model_to_staging(tracking_uri: str, model_name: str, model_version: str) -> None:
    """
    Transitions a specific model version to the 'Staging' stage.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to transition.
    - model_version (str): The version of the model to transition to the 'Staging' stage.

    Note:
    - This function transitions the specified model version to the 'Staging' stage.
      It does not archive existing versions by default.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage="Staging", archive_existing_versions=False
    )


def transition_model_to_production(tracking_uri: str, model_name: str, model_version: str) -> None:
    """
    Transitions a specific model version to the 'Production' stage.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to transition.
    - model_version (str): The version of the model to transition to the 'Production' stage.

    Note:
    - This function transitions the specified model version to the 'Production' stage.
      It archives existing versions by default.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage="Production", archive_existing_versions=True
    )


def update_model_description(tracking_uri: str, model_name: str, model_version: str, description: str) -> None:
    """
    Updates the description of a specific model version.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to update.
    - model_version (str): The version of the model to update.
    - description (str): The new description for the model version.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.update_model_version(name=model_name, version=model_version, description=description)


def tag_model(tracking_uri: str, model_name: str, model_version: str, tags: Dict) -> None:
    """
    Tags a specific model version with provided key-value pairs.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to tag.
    - model_version (str): The version of the model to tag.
    - tags (Dict): A dictionary containing key-value pairs to tag the model version.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    for tag_k, tag_v in tags.items():
        # Tag using model version
        client.set_model_version_tag(name=model_name, version=f"{model_version}", key=tag_k, value=tag_v)


def load_model_by_stage(tracking_uri: str, model_name: str, model_stage: str) -> PyFuncModel:
    """
    Loads a model based on its name and deployment stage.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.
    - model_stage (str): The deployment stage of the model ('Production', 'Staging', etc.).

    Returns:
    - PyFuncModel: The loaded model in the specified stage.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_stage}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model


def get_model_version_by_stage(tracking_uri, model_name, model_stage):
    latest_model_versions = get_latest_model_versions(tracking_uri, model_name)
    for model in latest_model_versions:
        if model["stage"] == model_stage:
            return model["version"]
    return None


def load_model_by_version(tracking_uri: str, model_name: str, model_version: str) -> PyFuncModel:
    """
    Loads a model based on its name and version number.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.
    - model_version (str): The version number of the model.

    Returns:
    - PyFuncModel: The loaded model of the specified version.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model


def register_model_from_run(tracking_uri: str, run: Run, model_name: str) -> None:
    """
    Registers a model generated from an MLflow Run.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - run (Run): MLflow Run object containing information about the run.
    - model_name (str): The desired name for the registered model.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = build_model_uri_from_run(run)
    mlflow.register_model(model_uri=model_uri, name=model_name)


def build_model_uri_from_run(run: Run) -> str:
    """
    Builds the model URI from the MLflow Run object.

    Args:
    - run (Run): MLflow Run object containing information about the run.

    Returns:
    - str: The model URI constructed from the run information.
    """
    artifact_path = json.loads(run.data.tags["mlflow.log-model.history"])[0]["artifact_path"]
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    return model_uri
