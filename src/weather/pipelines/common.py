import os
import tempfile
import warnings
from typing import Dict

import joblib
import pandas as pd
import requests
import sklearn.pipeline
from deepchecks.tabular import Dataset as DeepChecksDataset
from deepchecks.tabular.suites import (
    data_integrity,
    model_evaluation,
)
from minio import Minio
from weather.data.load_datasets import (
    load_prep_dataset_from_minio,
    load_raw_datasets_from_minio,
)
from weather.data.prep_datasets import (
    Dataset,
    prepare_and_merge_splits_to_dataset,
    prepare_binary_classification_tabular_data,
)
from weather.helpers.utils import (
    clean_temporary_dir,
    create_temporary_dir_if_not_exists,
)
from weather.mlflow.tracking import get_raw_artifacts_from_run
from weather.models.skl_train_models import score_evaluation_dict
from weather.pipelines.definitions import (
    MINIO_ACCESS_KEY,
    MINIO_API_HOST,
    MINIO_SECRET_KEY,
    SERVER_API_URL,
)

import mlflow
from prefect import task

warnings.filterwarnings("ignore")


def stop_mlflow_run(flow, flow_run, state):
    """Utilitary function to stop the current run after flow ends

    Note:
        Parameters are necessary, if not Prefect complains.
    """
    mlflow.end_run()


def make_mlflow_artifact_uri(experiment_id: str | None = None) -> str:
    """Generates the URI for the experiment

    Not that necessary.
    """
    import urllib.parse

    if experiment_id is None:
        run = mlflow.get_experiment()
        if run is not None:
            raise ValueError
        experiment_id = run.info.experiment_id

    return urllib.parse.urljoin(mlflow.get_tracking_uri(), f"/#/experiments/{experiment_id}")


def log_metrics(score_dict: Dict[str, float | str]):
    """Logs metrics to mlflow

    Not intended to be a prefect flow since we don't
    want to setup all the MLFlow setup. So we assume
    the MLFlow artifact tracking is set correctly.
    """
    mlflow.log_metric("train_" + score_dict["score_name"], score_dict["train"])
    mlflow.log_metric("val_" + score_dict["score_name"], score_dict["val"])
    mlflow.log_metric("test_" + score_dict["score_name"], score_dict["test"])


@task
def raw_data_extraction(curr_data_bucket: str) -> pd.DataFrame:
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    dataframes, ds_info = load_raw_datasets_from_minio(minio_client, curr_data_bucket)
    if not dataframes:
        return pd.DataFrame(), {}
    raw_df = pd.concat(dataframes, ignore_index=True)
    return raw_df, ds_info


@task
def prep_data_construction(
    ref_data_bucket: str,
    curr_data_bucket: str,
    dataset_ingestion_transformer,
    remove_horizonless_rows_transformer,
    target_creation_transformer,
) -> pd.DataFrame:
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    dataset = load_prep_dataset_from_minio(
        minio_client, ref_data_bucket
    )  # TODO: in "dev", fetch current splitted dataset, ready for training
    dataframes, ds_info = load_raw_datasets_from_minio(
        minio_client, curr_data_bucket
    )  # in "2011-01-01-prod", fetch 2011-01-01-weather_dataset_raw_production.csv in df
    from prefect import get_run_logger

    run_logger = get_run_logger()
    # These are visible in the API Server
    run_logger.info(str(dataframes[0].dtypes))
    run_logger.info(str(dataframes[0]))
    run_logger.info(str(len(dataframes)))
    dataset = prepare_and_merge_splits_to_dataset(
        dataset,  # dev dataset
        dataframes,  # [2011-01-01_raw_prod.df]
        dataset_ingestion_transformer,
        remove_horizonless_rows_transformer,
        target_creation_transformer,
    )
    return dataset, ds_info


@task
def data_preparation(
    ingested_df,
    remove_horizonless_rows_transformer,
    target_creation_transformer,
) -> Dataset:
    """Creates a Dataset out of our raw dataframes"""
    transformed_data = remove_horizonless_rows_transformer.transform(ingested_df)
    created_target = target_creation_transformer.transform(ingested_df)
    dataset = prepare_binary_classification_tabular_data(transformed_data, created_target)
    return dataset


@task
def fit_transformer(predictors_feature_engineering_transformer: sklearn.pipeline.Pipeline, dataset: Dataset):
    predictors_feature_engineering_transformer.fit(dataset.train_x)
    return predictors_feature_engineering_transformer


@task
def validate_ingested_data(ingested_df, feature_names, target_choice):
    """Run the data integrity suite on `ingested_df`. Return True if all tests pass, False otherwise."""

    # Populate Dataset parameters
    features = feature_names.numerical + feature_names.categorical + [target_choice.input_name]
    features = list(set(features))
    cat_features = [target_choice.input_name] + feature_names.categorical
    cat_features = list(set(cat_features))

    # Convert ingested_df into a deepchecks Dataset instance
    ds = DeepChecksDataset(ingested_df, features=features, cat_features=cat_features)

    # Run integrity suite
    integrity_suite = data_integrity()
    results = integrity_suite.run(ds)
    return results.passed()


@task
def score(model, transformer, dataset, metric) -> Dict[str, float | str]:
    """Returns the evaluation metrics"""
    score_dict = score_evaluation_dict(
        metric,  # alias score
        transformer,
        model,
        dataset,
    )
    return score_dict  # with keys "score_name", "train", "val", "test"


@task
def load_artifacts_from_mlflow(run):
    with tempfile.TemporaryDirectory() as d:
        best_feat_eng_obj, best_classifier_obj = get_raw_artifacts_from_run(
            mlflow.get_tracking_uri(), run=run, tmp_dir_path=d
        )
        return joblib.load(best_feat_eng_obj), joblib.load(best_classifier_obj)


@task
def validate_model(
    dataset, trained_model, trained_predictors_feature_engineering_transformer, run_id, excluded_check=5
):
    """Run the validation suite minus `WeekSegmentPerformance` on `dataset`. Return True if all tests pass, False otherwise."""
    # Populate train_ds
    x_train = trained_predictors_feature_engineering_transformer.transform(dataset.train_x)
    y_train = dataset.train_y
    train_ds = DeepChecksDataset(x_train, label=y_train, cat_features=[])
    # Populate test_ds
    x_test = trained_predictors_feature_engineering_transformer.transform(dataset.test_x)
    y_test = dataset.test_y
    test_ds = DeepChecksDataset(x_test, label=y_test, cat_features=[])
    # Run model validation suite
    evaluation_suite = model_evaluation()
    results = evaluation_suite.remove(excluded_check).run(train_ds, test_ds, trained_model)
    # Log results in json and html formats in "tests" subfolder of "artifats" in run `run_id`
    tmp_dir = create_temporary_dir_if_not_exists()

    def tmp_fpath(fpath):
        return os.path.join(tmp_dir, fpath)

    with open(tmp_fpath("deepchecks_report.json"), "w") as f:
        f.write(results.to_json())
    results.save_as_html(tmp_fpath("deepchecks_report.html"))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts("tmp", artifact_path="tests")
    clean_temporary_dir()
    return results


@task
def deploy():
    result = requests.get(SERVER_API_URL)
    return result.json()
