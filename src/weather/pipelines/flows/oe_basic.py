import logging
import os
import tempfile
from typing import Any
from pathlib import Path

import joblib
import mlflow
import prefect.context
import prefect.runtime.flow_run
import sklearn.pipeline
from weather.data.prep_datasets import Dataset
from weather.models.skl_tracked_train_models import SKLModelWrapper
from weather.pipelines.common import (
    build_transformer,
    data_preparation,
    data_validation,
    deploy,
    fit_transformer,
    log_metrics,
    make_mlflow_artifact_uri,
    score,
    stop_mlflow_run,
)
from weather.pipelines.definitions import (
    MLFLOW_TRACKING_URI,
    feature_names,
    metric,
    oldnames_newnames_dict,
    target_choice,
)
from weather.pipelines.flows.data_extraction import data_extraction
from weather.transformers.skl_transformer_makers import (
    make_dataset_ingestion_transformer,
    make_predictors_feature_engineering_transformer,
    make_remove_horizonless_rows_transformer,
    make_target_creation_transformer,
)
from mlflow.models import infer_signature
from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from sklearn.ensemble import RandomForestClassifier


@task
def train(dataset: Dataset, 
          model: Any, 
          predictors_feature_engineering_transformer: sklearn.pipeline.Pipeline
):
    x = predictors_feature_engineering_transformer.fit_transform(dataset.train_x)
    model.fit(x, dataset.train_y)
    return model

# NOTE(Participant): This task should only be used in this flow: in other flows we have
# other mechanisms to save the model to MLFlow already
@task
def save_model_mlflow(
    dataset,
    predictors_feature_engineering_transformer,
    classifier_obj,
):
    with tempfile.TemporaryDirectory() as temp_d:
        tmp_fpath = lambda fpath: os.path.join(temp_d, fpath)
        joblib.dump(predictors_feature_engineering_transformer, tmp_fpath("predictors_feature_eng_pipeline.joblib"))
        joblib.dump(classifier_obj, tmp_fpath("model.joblib"))
        mlflow.pyfunc.log_model(
            artifact_path="classifier",
            python_model=SKLModelWrapper(),
            artifacts={"predictors_feature_eng_path": tmp_fpath("predictors_feature_eng_pipeline.joblib"), 
                       "model_path": tmp_fpath("model.joblib")},
            input_example=dataset.train_x.sample(3),
            signature=infer_signature(
                dataset.train_x.head(), classifier_obj.predict(predictors_feature_engineering_transformer.transform(dataset.train_x))
            ),
            extra_pip_requirements=["weather"],
        )


@flow(name="orchestrated-experiment-basic", on_completion=[stop_mlflow_run])
def complete_flow(
    classifier_params: dict = None,
    #clustering_params: dict = None,
    weather_db_file: str | Path = None, 
    #socio_eco_data_file: str | None = None,
    mlflow_experiment_name: str = "Default",
):
    #
    # Run setup
    #
    # MLFlow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name=mlflow_experiment_name)
    run = mlflow.start_run(run_name=prefect.runtime.flow_run.get_name())

    # Create a configuration object we can pass around
    pipeline_experiment_bag = {}
    pipeline_experiment_bag["mlflow_experiment_name"] = experiment.name
    pipeline_experiment_bag["mlflow_experiment_id"] = experiment.experiment_id

    create_link_artifact(make_mlflow_artifact_uri(
        pipeline_experiment_bag["mlflow_experiment_id"]))

    # Logging setup
    run_logger = get_run_logger()
    # These are visible in the API Server
    run_logger.info("Hiii")
    # These are visible in the worker
    logging.info(mlflow.get_tracking_uri())
    logging.info(mlflow.get_registry_uri())

    #
    # Default values
    #
    if classifier_params is None:
        classifier_params = {
            "max_depth": 3,
            "max_features": 9,
            "class_weight": "balanced",
            "min_samples_leaf": 20,
        }

    # if clustering_params is None:
    #     clustering_params = {"n_clusters": 10, "n_init": "auto", "init": "k-means++"}

    # Some logging
    mlflow.log_params(classifier_params)
    # mlflow.log_params(clustering_params)

    #
    # Data Extraction
    #
    # NOTE(Participant): This one extracts the data from our database files and not DVC...
    dataset_ingestion_transformer = make_dataset_ingestion_transformer(
        target_choice, oldnames_newnames_dict)
    ingested_df = data_extraction(weather_db_file, dataset_ingestion_transformer)

    run_logger.info("Dataframe shape: %s", ingested_df.shape)

    #
    # Data Validation
    # TODO: translate it into deepchecks
    # validation_passed = data_validation(ingested_df) # great_expectations
    # if not validation_passed:
    #     run_logger.warning('Failed data validation. See artifacts or GX UI for more details.')

    #
    # Data preparation
    #
    # NOTE(Participant): This does not use our advanced training methods with tracking...
    # Option 1: train without tune
    remove_horizonless_rows_transformer = make_remove_horizonless_rows_transformer(target_choice)
    target_creation_transformer = make_target_creation_transformer(target_choice)
    dataset = data_preparation(
        ingested_df,
        remove_horizonless_rows_transformer,
        target_creation_transformer,
    )


    #data_transformer = build_transformer(params=clustering_params)
    #fit_transformer(data_transformer, dataset)

    #
    # Training
    #
    # df_train_transformed = transform_data(dataset, data_transformer, stage='train')
    classifier_obj = RandomForestClassifier(**classifier_params)
    model = train(
        dataset=dataset, 
        model=classifier_obj, 
        data_transformer=predictors_feature_engineering_transformer)

    #
    # Scoring
    #
    score_dict = score(
        metric=metric,
        transformer=predictors_feature_engineering_transformer,
        model=model,
        dataset=dataset,
    )

    logging.info("Logging to MLFLow...")
    # TODO: Check run
    log_metrics(score_dict)

    # TODO: Think of better heuristic to save
    save_model = True
    saved_model_name = "random_forest_from_oe_basic"
    if save_model:
        run_logger.info("Saving model named: %s", saved_model_name)
        save_model_mlflow(
            dataset, 
            predictors_feature_engineering_transformer, 
            classifier_obj,
        )

    should_deploy = False
    if should_deploy:
        deploy()
