import abc
from typing import List
import os
import joblib

import mlflow

from src.weather.data.prep_datasets import Dataset
from src.weather.features.dataframe_transformer import SimpleCustomPipeline
from src.weather.mlflow.tracking import Experiment
from src.weather.helpers.utils import create_temporary_dir_if_not_exists
from src.weather.helpers.utils import clean_temporary_dir
from src.weather.helpers.utils import camel_to_snake


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
        import src
        self.loaded_data_transformer = joblib.load(context.artifacts["feature_eng_path"])
        self.loaded_classifier = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        """
        Generates predictions using the loaded Scikit-learn transformer and classifier.

        This method retrieves the Scikit-learn transformer and classifier artifacts.

        Args:
        - context: MLflow context containing the stored model artifacts.
        - model_input: Input data to be processed by the model.

        Returns:
        - Tuple: Loaded transformer and classifier artifacts.
        """
        return self.loaded_data_transformer, self.loaded_classifier


def train_and_evaluate_with_tracking(
        data: Dataset, data_transformer: SimpleCustomPipeline,
        classifers_list: List[abc.ABCMeta],
        experiment:Experiment,
        ds_info:dict) -> None:
    """
    Trains each classifier from the list on the training data and evaluates on all splits,
    while tracking metrics and artifacts using MLflow.

    Args:
    - data (Dataset): Dataset containing training, validation, and test data.
    - data_transformer (Pipeline): Scikit-learn feature engineering pipeline.
    - classifiers_list (List[abc.ABCMeta]): List of Scikit-learn classifier classes.
    - experiment (Experiment): Experiment settings for MLflow tracking.
    - ds_info (dict): dataset metadata information (from dvc)
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    # Create a temporary directory for storing artifacts
    tmp_dir = create_temporary_dir_if_not_exists()
    # Save the data transformer pipeline
    tmp_fpath = lambda fpath: os.path.join(tmp_dir, fpath)
    joblib.dump(data_transformer, tmp_fpath('feature_eng.joblib'))
    # Transform the data for training, validation, and test sets
    train_inputs = data_transformer.transform(data.train_x)
    valid_inputs = data_transformer.transform(data.val_x)
    test_inputs = data_transformer.transform(data.test_x)
    # Loop through each classifier in the list
    for classifier in classifers_list:
        # Set up run-specific details
        classifier_shortname = camel_to_snake(classifier.__name__)
        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"run_{classifier_shortname}"):
            mlflow.set_tag("sklearn_model", classifier_shortname)
            # Instantiate and train the classifier
            classifier_obj = classifier()
            classifier_obj.fit(train_inputs, data.train_y)
            joblib.dump(classifier_obj, tmp_fpath('model.joblib'))
            # Evaluate accuracy on different datasets
            accuracy_dict = accuracy_evaluation(data_transformer, classifier_obj, data)
            # Track accuracy metrics
            mlflow.log_metric("train_accuracy", accuracy_dict['train'])
            mlflow.log_metric("valid_accuracy", accuracy_dict['val'])
            mlflow.log_metric("test_accuracy", accuracy_dict['test'])
            # Generate an example input and a model signature
            sample = data.train_x.sample(3)
            signature = infer_signature(data.train_x,
                                        classifier_obj.predict(train_inputs))
            # Log the trained model as an MLflow artifact
            artifacts = {"feature_eng_path": tmp_fpath('feature_eng.joblib'),
                         "model_path": tmp_fpath('model.joblib')}
            mlflow_pyfunc_model_path = 'classifier'
            mlflow.pyfunc.log_model(
                artifact_path=mlflow_pyfunc_model_path,
                python_model=SKLModelWrapper(),
                artifacts=artifacts,
                input_example=sample,
                signature=signature,
                extra_pip_requirements=["bank-marketing"],
            )
            # Log the ds_info as a YAML file under the run's root artifact directory
            mlflow.log_dict(ds_info, "data.yml")
            # Track ROC curve plots for validation and test sets
            display = RocCurveDisplay.from_predictions(data.val_y.values,
                                        classifier_obj.predict_proba(valid_inputs)[:,1])
            mlflow.log_figure(display.figure_, 'plots/ValidRocCurveDisplay.png')
            display = RocCurveDisplay.from_predictions(data.test_y.values,
                                        classifier_obj.predict_proba(test_inputs)[:,1])
            mlflow.log_figure(display.figure_, 'plots/TestRocCurveDisplay.png')
    # Clean up temporary directory
    clean_temporary_dir()