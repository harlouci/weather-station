"""
This module includes the functions to train and evalure scikit-learn models
for weather prediction ML applications.
"""

import abc

# import plotly.express as px # TODO
from typing import List

import joblib
from sklearn.pipeline import Pipeline
from weather.data.prep_datasets import Dataset
from weather.helpers.utils import camel_to_snake, clean_temporary_dir, create_temporary_dir_if_not_exists
from weather.mlflow.tracking import Experiment
from weather.models.skl_train_models import score_evaluation_dict

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature


class SKLModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to encapsulate Scikit-learn models for training and inference.
    """

    def load_context(self, context):
        """
        Loads the serialized Scikit-learn transformer and classifier artifacts.

        This method is invoked when an MLflow model is loaded using pyfunc.load_model(),
        constructing the Python model instance.

        Args:
        - context: MLflow context containing the stored model artifacts.
        """
        import joblib

        self.loaded_data_transformer = joblib.load(context.artifacts["feature_eng_path"])
        self.loaded_classifier = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        """
        Generates predictions using the loaded scikit-learn transformer and classifier.

        This method retrieves the Scikit-learn transformer and classifier artifacts.

        Args:
        - context: MLflow context containing the stored model artifacts.
        - model_input: Input data to be processed by the model.

        Returns:
        - Tuple: Loaded transformer and classifier artifacts.
        """
        return self.loaded_data_transformer, self.loaded_classifier


def train_and_evaluate_with_tracking(
    data: Dataset,
    predictors_feature_engineering_transformer: Pipeline,
    classifers_list: List[abc.ABCMeta],
    score,
    experiment: Experiment,
) -> None:
    """
    Trains each classifier from the list on the training data and evaluates on all splits,
    while tracking metrics and artifacts using MLflow.

    Args:
    - data (Dataset): Dataset containing training, validation, and test data.
    - predictors_feature_engineering_transformer (Pipeline): scikit-learn feature engineering pipeline.
    - classifiers_list (List[abc.ABCMeta]): List of Scikit-learn classifier classes.
    - experiment (Experiment): Experiment settings for MLflow tracking.
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    # Create a temporary directory for storing artifacts
    tmp_dir = create_temporary_dir_if_not_exists()

    # Save the data transformer pipeline
    def tmp_fpath(fpath):
        return tmp_dir / fpath

    # Transform the data for training, validation, and test sets
    train_inputs = predictors_feature_engineering_transformer.fit_transform(data.train_x)
    joblib.dump(predictors_feature_engineering_transformer, tmp_fpath("predictors_feature_eng_pipeline.joblib"))
    # Loop through each classifier in the list
    for classifier in classifers_list:
        # Set up run-specific details
        classifier_shortname = camel_to_snake(classifier.__name__)
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"run_{classifier_shortname}"):
            # mlflow.doctor()
            mlflow.set_tag("sklearn_model", classifier_shortname)
            # Instantiate and train the classifier
            classifier_obj = classifier()
            classifier_obj.fit(train_inputs, data.train_y)
            joblib.dump(classifier_obj, tmp_fpath("model.joblib"))
            # Evaluate accuracy on different datasets
            score_dict = score_evaluation_dict(
                score,
                predictors_feature_engineering_transformer,
                classifier_obj,
                data,
            )
            # Track accuracy metrics
            mlflow.log_metric("train_" + score_dict["score_name"], score_dict["train"])
            mlflow.log_metric("val_" + score_dict["score_name"], score_dict["val"])
            mlflow.log_metric("test_" + score_dict["score_name"], score_dict["test"])
            # Generate an example input and a model signature
            sample = data.train_x.sample(3)
            signature = infer_signature(
                data.train_x.head(),  # TODO: slightly modify pipelines qui remove NaNs before feature engineering
                classifier_obj.predict(train_inputs),
            )
            # Log the trained model as an MLflow artifact
            artifacts = {
                "feature_eng_path": tmp_fpath("predictors_feature_eng_pipeline.joblib"),
                "model_path": tmp_fpath("model.joblib"),
            }
            mlflow_pyfunc_model_path = "classifier"
            mlflow.pyfunc.log_model(
                artifact_path=mlflow_pyfunc_model_path,
                python_model=SKLModelWrapper(),
                artifacts=artifacts,
                input_example=sample,
                signature=signature,
                extra_pip_requirements=["weather"],
            )

    # Clean up temporary directory
    clean_temporary_dir()
