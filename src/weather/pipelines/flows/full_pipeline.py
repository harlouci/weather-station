import logging
import os
import tempfile
import warnings
from time import time
from typing import Callable

import joblib
import mlflow
import prefect.context
import prefect.runtime.flow_run
from hyperopt import fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.models.signature import infer_signature
from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from sklearn.ensemble import RandomForestClassifier

from weather.data.minio_utilities import (
    delete_files_in_minio,
    extract_most_recent_filename_if_any,
    write_dataframe_to_minio,
)
from weather.data.prep_datasets import (
    Dataset,
    prepare_binary_classification_tabular_data,
    transform_dataset_and_create_target,
)
from weather.mlflow.registry import (
    get_model_version_by_stage,
    register_model_from_run,
    tag_model,
    transition_model_to_production,
    transition_model_to_staging,
)
from weather.mlflow.tracking import (
    Experiment,
    get_best_run,
)
from weather.models.skl_tracked_train_models import SKLModelWrapper
from weather.pipelines.common import (
    deploy,
    fit_transformer,
    load_artifacts_from_mlflow,
    log_metrics,
    make_mlflow_artifact_uri,
    raw_data_extraction,
    score,
    stop_mlflow_run,
    validate_ingested_data,
    validate_model,
)
from weather.pipelines.definitions import (
    MLFLOW_TRACKING_URI,
    feature_names,
    metric,
    oldnames_newnames_dict,
    target_choice,
)
from weather.transformers.skl_transformer_makers import (
    make_dataset_ingestion_transformer,
    make_predictors_feature_engineering_transformer,
    make_remove_horizonless_rows_transformer,
    make_target_creation_transformer,
)

warnings.filterwarnings("ignore")


def build_training_and_evaluation_func(
    model_family,
    data: Dataset,
    experiment_bag: dict,
    metric: Callable,
    random_state=1234,
):
    """Create a new evaluation function and returns it."""

    def train_eval_func(hparams):
        """Train, evaluate and log an sklearn model with given parameters by invoking MLflow run.

        Set an mlflow run with `experiment_bag` identifiers.
        Train a RF model with given `hparams`, `random_state`, `data`, `metric`.
        Eval the trained model and  return -score_dict["val"].
        """

        ## 1 - Set an mlflow run with `experiment_bag` identifiers
        with mlflow.start_run(
            experiment_id=experiment_bag["mlflow_experiment_id"]
        ), tempfile.TemporaryDirectory() as temp_d:
            # Utility method to make things shorter
            def tmp_fpath(fpath):
                return os.path.join(temp_d, fpath)

            # NOTE(Participant): This was added
            # N.B. We set a tag name so we can differentiate which Prefect run caused this
            #      Mlflow run. This will be useful to query models that were trained during
            #      this Prefect run.
            mlflow.set_tag("prefect_run_name", experiment_bag["prefect_run_name"])

            ## 2 - Train a RF model with given `hparams`, `random_state`, `data`, `metric`

            # Create, fit, and dump predictors_feature_engineering_transformer
            predictors_feature_engineering_transformer = make_predictors_feature_engineering_transformer(
                feature_names,
                target_choice,
            )
            # predictors_feature_engineering_transformer.fit(data.train_x)
            fit_transformer.fn(predictors_feature_engineering_transformer, dataset=data)
            joblib.dump(predictors_feature_engineering_transformer, tmp_fpath("predictors_feature_eng_pipeline.joblib"))

            # Pass the parameters used to train RF into dictionary
            (max_depth, max_features, class_weight, min_samples_leaf) = hparams
            rf_params = {
                "max_depth": max_depth,
                "max_features": max_features,
                "class_weight": class_weight,
                "min_samples_leaf": min_samples_leaf,
                "random_state": random_state,
            }

            # Create predictors features, set the RF model, with given `hparams`, train and dump the model
            train_inputs = predictors_feature_engineering_transformer.transform(data.train_x)
            classifier_obj = model_family(**rf_params)
            classifier_obj.fit(train_inputs, data.train_y)
            joblib.dump(classifier_obj, tmp_fpath("model.joblib"))

            ## 3 - Log trained model in MLflow Tracking Server

            # Get model signature
            sample = data.train_x.sample(3)
            signature = infer_signature(data.train_x.head(), classifier_obj.predict(train_inputs))
            # Log Model
            artifacts = {
                "feature_eng_path": tmp_fpath(
                    "predictors_feature_eng_pipeline.joblib"
                ),  # cannot change name of the dictionary because used by mlflow
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
            # Log params
            mlflow.log_params(
                {
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "class_weight": class_weight,
                    "min_samples_leaf": min_samples_leaf,
                    "random_state": random_state,
                }
            )

            # TODO(Participant): Reuse the method 'score' we defined above, but without the flow
            score_dict = score.fn(
                model=classifier_obj,
                transformer=predictors_feature_engineering_transformer,
                dataset=data,
                metric=metric,
            )
            # Track metrics
            log_metrics(score_dict)
            return -score_dict["val"]

    return train_eval_func


# N.B: We removed the feature_names
@task
def tune(
    model_family,
    data: Dataset,
    metric: Callable,  # alias for score
    max_runs,
    experiment_bag,
    ds_info,
) -> None:
    """
    Run hyperparameter optimization on space defined within the code, for parameters
    specific to RandomForestClassifer.
    TODO: Hyperparameters space defined within the code. Should be defined elsewhere.
    """

    # Just a shortcut to both:
    #    1) Set current experiment and
    #    2) save the variable for experiment_id
    # For now, this is unused (the local functions will call set_experiment themselves)
    _ = mlflow.set_experiment(experiment_id=experiment_bag["mlflow_experiment_id"]).experiment_id

    # Search space for RF
    space = [
        scope.int(hp.quniform("max_depth", 1, 30, q=1)),
        hp.uniform("max_features", 0.05, 0.8),
        hp.choice("class_weight", ["balanced", None]),
        scope.int(hp.quniform("min_samples_leaf", 5, 100, q=5)),
    ]
    # Optimisation function that takes parent id and search params as input
    fmin(
        fn=build_training_and_evaluation_func(
            model_family,
            data,
            experiment_bag,
            metric,
        ),
        space=space,
        algo=tpe.suggest,
        max_evals=max_runs,
    )


@flow(name="full-pipeline", on_completion=[stop_mlflow_run])
def automated_pipeline(
    dev_bucket: str = "dev",
    prod_bucket: str = "prod",
    max_runs: int = 10,
    mlflow_experiment_name: str = "tune_random_forest_with_full_pipeline",
):
    """TODO:
    max_runs: The higher and the better forhyperparameters exploration
    Replaces experiments with HP tuning."""

    ## MLFlow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    unique_experiment_name = mlflow_experiment_name + "_" + str(int(time()))
    current_experiment = mlflow.set_experiment(experiment_name=unique_experiment_name)

    # Create a configuration object we can pass around
    pipeline_experiment_bag = {}
    pipeline_experiment_bag["mlflow_experiment_name"] = current_experiment.name
    pipeline_experiment_bag["mlflow_experiment_id"] = current_experiment.experiment_id
    pipeline_experiment_bag["prefect_run_name"] = prefect.runtime.flow_run.get_name()

    # These are visible in the worker
    logging.info(mlflow.get_tracking_uri())

    ## Prefect setup
    create_link_artifact(make_mlflow_artifact_uri(pipeline_experiment_bag["mlflow_experiment_id"]))

    # Logging setup
    run_logger = get_run_logger()
    # These are visible in the API Server
    run_logger.info("Hi, I'm Prefect, your automated pipeline runner.")

    ## 1) Extract data from prod bucket
    df, ds_info = raw_data_extraction(prod_bucket)  # csv files of bucket prod, merged in a random order
    if df.empty:
        run_logger.info("The production bucket is empty.")
        return
    run_logger.info(f"The production bucket contains the following data files: {ds_info}")
    run_logger.info(f"df columns in step 1: {df.columns}")

    # 2) Ingest df
    dataset_ingestion_transformer = make_dataset_ingestion_transformer(target_choice, oldnames_newnames_dict)
    ingested_df = dataset_ingestion_transformer.transform(df)

    # 3) Data Validation with Deepchecks
    validation_passed = validate_ingested_data(ingested_df, feature_names, target_choice)
    if not validation_passed:
        run_logger.warning(
            "Failed data validation. See artifacts or deepchecks UI for more details."
        )  # TODO: create artifacts

    ## 4) Save df as last(##-##-##)_data.csv in dev_bucket (which now contains weather_dataset_raw_development.csv, last(##-##-##)_data.csv)
    df_filename = extract_most_recent_filename_if_any(ds_info, "data")
    write_dataframe_to_minio(df, dev_bucket, df_filename)
    # run_logger.info(f"Filepath: {os.path.join(dev_bucket, filename)}")
    run_logger.info(f"Content of prod bucket saved in dev bucket as {df_filename}")

    # 5) Clean prod bucket:
    delete_files_in_minio(prod_bucket, list(ds_info))
    run_logger.info(f"Files {list(ds_info)} deleted from prod bucket.")

    ## 6) Extract data from dev bucket
    df, ds_info = raw_data_extraction(
        dev_bucket
    )  # list_csv_files = [weather_dataset_raw_development.csv, last(##-##-##)_data.csv]
    run_logger.info(f"The development bucket contains the following data files: {ds_info}")
    run_logger.info(f"df columns in step 6: {df.columns}")

    ## 7) Save df as weather_dataset_raw_development.csv in dev_bucket
    df_filename = "weather_dataset_raw_development.csv"
    write_dataframe_to_minio(df, dev_bucket, df_filename)
    run_logger.info(f"File {df_filename} saved in dev bucket.")

    ## 8) clean dev bucket:
    del ds_info["weather_dataset_raw_development.csv"]
    delete_files_in_minio(dev_bucket, list(ds_info))
    run_logger.info(f"Files {list(ds_info)} deleted from dev bucket.")

    run_logger.info("STOP--STOP--STOP--STOP-STOP!!!!")

    ## 9) Ingest, transform,and split df

    # Ingest and transform
    remove_horizonless_rows_transformer = make_remove_horizonless_rows_transformer(target_choice)
    target_creation_transformer = make_target_creation_transformer(target_choice)
    transformed_data, created_target = transform_dataset_and_create_target(
        df,
        dataset_ingestion_transformer,
        remove_horizonless_rows_transformer,
        target_creation_transformer,
    )

    # Split
    dataset = prepare_binary_classification_tabular_data(
        transformed_data,
        created_target,
    )

    ## 10) Train with hyperparameter tuning
    #######################################
    tune(
        model_family=RandomForestClassifier,
        data=dataset,
        metric=metric,
        max_runs=max_runs,
        experiment_bag=pipeline_experiment_bag,
        ds_info=ds_info,
    )

    ## 11) Extract best_run,  score, save, register and stage in mlflow

    # Extract best_run
    current_experiment = Experiment(
        tracking_server_uri=mlflow.get_tracking_uri(),
        name=pipeline_experiment_bag["mlflow_experiment_name"],
    )
    best_run = get_best_run(
        experiment=current_experiment,
        filter_string="tags.prefect_run_name = '{}'".format(pipeline_experiment_bag["prefect_run_name"]),
    )

    # Score
    feat_eng_obj, best_classifier_obj = load_artifacts_from_mlflow(
        run=best_run
    )  # best(trans, model)= (trans, best model)
    score_dict = score(
        model=best_classifier_obj,
        dataset=dataset,
        transformer=feat_eng_obj,  # no hyperparameters research when fitting this transformer
        metric=metric,
    )
    run_logger.info(f"Best run score: {score_dict}")

    # Register and stage (to staging)
    save_model = True
    saved_model_name = "random_forest"
    if save_model:
        run_logger.info("Saving model named: %s", saved_model_name)
        register_model_from_run(current_experiment.tracking_server_uri, best_run, saved_model_name)
        model_version = get_model_version_by_stage(current_experiment.tracking_server_uri, saved_model_name, "None")
        transition_model_to_staging(current_experiment.tracking_server_uri, saved_model_name, model_version)

    # 12) Validate model
    results = validate_model(dataset, best_classifier_obj, feat_eng_obj, best_run.info.run_id)
    run_logger.info(f" {len(results.get_passed_checks())} of model tests are passed.")
    run_logger.info(f" {len(results.get_not_passed_checks())} of model tests are failed.")
    run_logger.info(f" {len(results.get_not_ran_checks())} of model tests are not runned.")
    if results.passed(fail_if_check_not_run=True, fail_if_warning=True):
        run_logger.info("The model validation succeeds.")
        tag_model(current_experiment.tracking_server_uri, saved_model_name, model_version, {"Model Tests": "PASSED"})
    else:
        run_logger.info("The model validation fails.")
        tag_model(current_experiment.tracking_server_uri, saved_model_name, model_version, {"Model Tests": "FAILED"})

    ## 13) Deploy model
    should_deploy = True
    if should_deploy:
        transition_model_to_production(current_experiment.tracking_server_uri, saved_model_name, model_version)
        model_info = deploy()
        run_logger.info(model_info)
