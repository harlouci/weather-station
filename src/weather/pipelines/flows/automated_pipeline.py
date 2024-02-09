import logging
from typing import Callable

import prefect.context
import prefect.runtime.flow_run
from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from sklearn.ensemble import RandomForestClassifier
from weather.data.prep_datasets import Dataset
from weather.mlflow.registry import register_model_from_run
from weather.mlflow.tracking import Experiment, get_best_run

#from weather.pipelines.blocks import dvc_block
from weather.pipelines.common import (
    data_preparation,
    deploy,
    fit_transformer,
    load_artifacts_from_mlflow,
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

import mlflow


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


@flow(name="automated-pipeline", on_completion=[stop_mlflow_run])
def automated_pipeline(
    #load_splits_from_dvc: bool = True,
    #reextract: bool = False,
    #db: str | None = None,
    #socio_eco_data_file: str | None = None,
    max_runs: int = 20,
    mlflow_experiment_name: str = "tune_random_forest_with_prefect",
):
    """Replaces experiments with HP tuning"""
    ######################################
    # Run setup
    ######################################
    #dvc_remote = "minio-mybucket-from-laptop"

    # MLFlow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    current_experiment = mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    # Create a configuration object we can pass around
    pipeline_experiment_bag = {}
    pipeline_experiment_bag["mlflow_experiment_name"] = current_experiment.name
    pipeline_experiment_bag["mlflow_experiment_id"] = current_experiment.experiment_id
    pipeline_experiment_bag["prefect_run_name"] = prefect.runtime.flow_run.get_name()

    create_link_artifact(make_mlflow_artifact_uri(
        pipeline_experiment_bag["mlflow_experiment_id"]))

    # Logging setup
    run_logger = get_run_logger()
    # These are visible in the API Server
    run_logger.info("Hiii")
    # These are visible in the worker
    logging.info(mlflow.get_tracking_uri())

    ######################################
    # Data Extraction
    ######################################
    # if reextract:
    #     run_logger.info("Re-extracting data from database")
    #     df = data_extraction(db=db, socio_eco_data_file=socio_eco_data_file)
    #     # TODO
    #     # Empty dict since fresh extraction
    #     ds_extract_info = {}
    # else:
    #     run_logger.info("Using DVC to load extraction dataframe")
    #     df, ds_extract_info = load_extraction_from_dvc(dvc_block=dvc_block, dvc_remote=dvc_remote)


    dataset_ingestion_transformer = make_dataset_ingestion_transformer(
        target_choice, oldnames_newnames_dict)
    ingested_df = data_extraction(weather_db_file, dataset_ingestion_transformer)

    ######################################
    # Data Validation
    ######################################
    # TODO: create version deepchecks of data_validation
    # validation_passed = data_validation(ingested_df) # Implements great expectation
    # if not validation_passed:
    #     run_logger.warning('Failed data validation. See artifacts or GX UI for more details.')

    ######################################
    # Data preparation: returns dataset with splits train, val, test
    ######################################

    remove_horizonless_rows_transformer = make_remove_horizonless_rows_transformer(target_choice)
    target_creation_transformer = make_target_creation_transformer(target_choice)
    dataset = data_preparation(
        ingested_df,
        remove_horizonless_rows_transformer,
        target_creation_transformer,
    )
    ds_info = {}

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
        filter_string="tags.prefect_run_name = '{}'".format(
            pipeline_experiment_bag["prefect_run_name"]),
    )

    best_feat_eng_obj, best_classifier_obj = load_artifacts_from_mlflow(run=best_run)

    # TODO(ERL)
    # Transition to staging
    # Load from mlflow
    # Model validation with deepchecks and transtition to prod if passed

    ######################################
    # Scoring
    ######################################
    metrics = score(metric = metric,
                    transformer=best_feat_eng_obj,
                    model=best_classifier_obj,
                    dataset=dataset,
                    )

    ######################################
    # Model saving
    ######################################
    # TODO: Think of logic to save
    save_model = True
    saved_model_name = "random_forest_from_automated_pipeline"
    if save_model:
        run_logger.info("Saving model named: %s", saved_model_name)
        register_model_from_run(current_experiment.tracking_server_uri,
                                best_run, saved_model_name)

    ######################################
    # Model deployment
    ######################################
    should_deploy = False
    if should_deploy:
        deploy()
