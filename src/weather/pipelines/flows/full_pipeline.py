import logging
import warnings

warnings.filterwarnings("ignore")
from time import time
from typing import Callable

import mlflow
import prefect.context
import prefect.runtime.flow_run
from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from sklearn.ensemble import RandomForestClassifier
from weather.data.prep_datasets import Dataset
from weather.mlflow.registry import (
    get_model_version_by_stage,
    register_model_from_run,
    transition_model_to_production,
    transition_model_to_staging,
)
from weather.mlflow.tracking import (
    Experiment,
    get_best_run,
)
from weather.pipelines.common import (
    #validate_model,  # deepchecks
    #data_validation, # deepchecks
    deploy,
    fit_transformer,
    load_artifacts_from_mlflow,
    log_metrics,
    make_mlflow_artifact_uri,
    prep_data_construction,
    raw_data_extraction,
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
from weather.transformers.skl_transformer_makers import (
    make_dataset_ingestion_transformer,
    make_predictors_feature_engineering_transformer,
    make_remove_horizonless_rows_transformer,
    make_target_creation_transformer,
)


# N.B.: Note that we removed the feature_names
# N.B.: Note that due to how the hyperparameter tuning is done, we need to repeat
#       most steps.
def build_evaluation_func(
    data: Dataset,
    experiment_bag: dict,
    metric: Callable,
    random_state = 1234,
):
    """
    Create a new evaluation function
    :return: new evaluation function.
    """

    def eval_func(hparams):
        """
        Train sklearn model with given parameters by invoking MLflow run.
        :param params: Parameters to the train script we optimize over
        :return: The metric value evaluated on the validation data.
        """
        import os
        import tempfile

        import joblib
        from weather.models.skl_tracked_train_models import SKLModelWrapper

        from mlflow.models.signature import infer_signature

        with mlflow.start_run(
            experiment_id=experiment_bag["mlflow_experiment_id"]
        ), tempfile.TemporaryDirectory() as temp_d:
            # Utility method to make things shorter
            tmp_fpath = lambda fpath: os.path.join(temp_d, fpath)

            # NOTE(Participant): This was added
            # N.B. We set a tag name so we can differentiate which Prefect run caused this
            #      Mlflow run. This will be useful to query models that were trained during
            #      this Prefect Run.
            mlflow.set_tag("prefect_run_name", experiment_bag["prefect_run_name"])

            # Params used to train RF
            (max_depth, max_features, class_weight, min_samples_leaf) = hparams
            predictors_feature_engineering_transformer = make_predictors_feature_engineering_transformer(
                feature_names,
                target_choice,
            )
            # predictors_feature_engineering_transformer.fit(data.train_x)
            fit_transformer.fn(
                predictors_feature_engineering_transformer, dataset=data)

            joblib.dump(predictors_feature_engineering_transformer,
                        tmp_fpath("predictors_feature_eng_pipeline.joblib"))
            # Pass the parameters into dictionary
            rf_params = {
                "max_depth": max_depth,
                "max_features": max_features,
                "class_weight": class_weight,
                "min_samples_leaf": min_samples_leaf,
                "random_state": random_state,
            }

            # Define model
            train_inputs = predictors_feature_engineering_transformer.transform(data.train_x)
            classifier_obj = RandomForestClassifier(**rf_params)
            classifier_obj.fit(train_inputs, data.train_y)
            joblib.dump(classifier_obj, tmp_fpath("model.joblib"))
            # Get Model Signature
            sample = data.train_x.sample(3)
            signature = infer_signature(data.train_x.head(),
                                        classifier_obj.predict(train_inputs))
            # Log Model
            artifacts = {
                "predictors_feature_eng_path": tmp_fpath("predictors_feature_eng_pipeline.joblib"),
                "model_path": tmp_fpath("model.joblib")}
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

    return eval_func


# N.B: We removed the feature_names
@task
def tune(
    data: Dataset,
    # feature_names: FeatureNames,
    # target_choice: TargetChoice,
    metric: Callable, # alias for score
    max_runs,
    experiment_bag,
    ds_info,

) -> None:
    """
    Run hyperparameter optimization.
    """
    from hyperopt import fmin, hp, tpe
    from hyperopt.pyll import scope

    # Just a shortcut to both:
    #    1) Set current experiment and
    #    2) save the variable for experiment_id
    # For now, this is unused (the local functions will call set_experiment themselves)
    experiment_id = mlflow.set_experiment(
        experiment_id=experiment_bag["mlflow_experiment_id"]).experiment_id
    # Search space for KMeans + RF
    space = [
        scope.int(hp.quniform("max_depth", 1, 30, q=1)),
        hp.uniform("max_features", 0.05, 0.8),
        hp.choice("class_weight", ["balanced", None]),
        scope.int(hp.quniform("min_samples_leaf", 5, 100, q=5)),
    ]
    # Optimisation function that takes parent id and search params as input
    fmin(
        fn=build_evaluation_func(
            data,
            #feature_names,
            #target_choice,
            experiment_bag,
            metric,
        ),
        space=space,
        algo=tpe.suggest,
        max_evals=max_runs,
    )


@flow(name="full-pipeline", on_completion=[stop_mlflow_run])
def automated_pipeline(
    ref_data_bucket: str = "reference-data",
    curr_data_bucket: str = "current-data",
    max_runs: int = 1,
    mlflow_experiment_name: str = "tune_randome_forest_with_full_pipeline",
):
    """Replaces experiments with HP tuning"""
    ######################################
    # Run setup
    ######################################
    # MLFlow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    unique_experiment_name = mlflow_experiment_name + "_" + str(int(time()))
    current_experiment = mlflow.set_experiment(experiment_name=unique_experiment_name)

    # Create a configuration object we can pass around
    pipeline_experiment_bag = {}
    pipeline_experiment_bag["mlflow_experiment_name"] = current_experiment.name
    pipeline_experiment_bag["mlflow_experiment_id"] = current_experiment.experiment_id
    pipeline_experiment_bag["prefect_run_name"] = prefect.runtime.flow_run.get_name()

    create_link_artifact(make_mlflow_artifact_uri(pipeline_experiment_bag["mlflow_experiment_id"]))

    # Logging setup
    run_logger = get_run_logger()
    # These are visible in the API Server
    run_logger.info("Hiii")
    # These are visible in the worker
    logging.info(mlflow.get_tracking_uri())

    ######################################
    # Data Extraction
    ######################################
    df = raw_data_extraction(curr_data_bucket) # "weather_dataset_raw_development.csv"

    # Data ingestion
    dataset_ingestion_transformer = make_dataset_ingestion_transformer(target_choice, oldnames_newnames_dict)
    ingested_df = dataset_ingestion_transformer.transform(df) # TODO: orphan
    ######################################
    # Data Validation
    ######################################
    # TODO: create  version deepchecks
    # validation_passed = data_validation(ingested_df)
    # if not validation_passed:
    #     run_logger.warning('Failed data validation. See artifacts or GX UI for more details.')

    ######################################
    # Data preparation
    ######################################

    remove_horizonless_rows_transformer = make_remove_horizonless_rows_transformer(target_choice)
    target_creation_transformer = make_target_creation_transformer(target_choice)

    dataset, ds_info = prep_data_construction(           # This dataset is the new dev, consisting of dev + 2011-01-01_prod
        ref_data_bucket,                                 # bucket dev
        curr_data_bucket,                                # buckets 2011-01-01-prod, 2011-01-02-prod,...
        dataset_ingestion_transformer,
        remove_horizonless_rows_transformer,
        target_creation_transformer,
    )
                                                                     #
    ######################################
    # Training with hyperparameter search
    #####################################
    tune(
        data=dataset,
        metric=metric,
        max_runs=max_runs,
        experiment_bag=pipeline_experiment_bag,
        ds_info=ds_info,
    )

    current_experiment = Experiment(
        tracking_server_uri=mlflow.get_tracking_uri(),
        name=pipeline_experiment_bag["mlflow_experiment_name"],
    )

    # Get the best run of the current experiment of the current Prefect Run
    # (we used a tag in MLFlow. We set the tag key to "prefect_run_name")
    best_run = get_best_run(
        experiment=current_experiment,
        filter_string="tags.prefect_run_name = '{}'".format(pipeline_experiment_bag["prefect_run_name"]),
    )

    ######################################
    # Scoring
    ######################################
    best_feat_eng_obj, best_classifier_obj = load_artifacts_from_mlflow(run=best_run)
    score_dict = score(
        model=best_classifier_obj,
        dataset=dataset,
        transformer=best_feat_eng_obj,
        metric=metric,
    )
    run_logger.info(score_dict)

    # TODO: HERE BELOW
    ######################################
    # Model register
    ######################################
    save_model = True
    saved_model_name = "random_forest_from_full_pipeline"
    if save_model:
        run_logger.info("Saving model named: %s", saved_model_name)
        register_model_from_run(current_experiment.tracking_server_uri, best_run, saved_model_name)
        model_version = get_model_version_by_stage(current_experiment.tracking_server_uri, saved_model_name, "None")
        transition_model_to_staging(current_experiment.tracking_server_uri, saved_model_name, model_version)

    ######################################
    # Model validation : WITH DEEPCHECK => TODO: Install deepchecks in env. Light modif to  validate_model()
    ######################################
    # result = validate_model(dataset, best_feat_eng_obj, best_classifier_obj, best_run.info.run_id)
    # run_logger.info(f" {len(result.get_passed_checks())} of Model tests are passed.")
    # run_logger.info(f" {len(result.get_not_passed_checks())} of Model tests are failed.")
    # run_logger.info(f" {len(result.get_not_ran_checks())} of Model tests are not runned.")
    # if result.passed(fail_if_check_not_run=True, fail_if_warning=True):
    #     run_logger.info("The Model validation succeeds")
    #     tag_model(current_experiment.tracking_server_uri, saved_model_name, model_version, {"Model Tests": "PASSED"})
    # else:
    #     run_logger.info("The Model validation fails")
    #     tag_model(current_experiment.tracking_server_uri, saved_model_name, model_version, {"Model Tests": "FAILED"})
    ######################################
    # Model deployment
    ######################################
    should_deploy = True
    if should_deploy:
        transition_model_to_production(current_experiment.tracking_server_uri, saved_model_name, model_version)
        model_info = deploy()
        run_logger.info(model_info)
