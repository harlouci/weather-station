import os
import warnings

warnings.filterwarnings("ignore")
import tempfile
from typing import Dict, Tuple

import joblib
import pandas as pd
import prefect.runtime.flow_run
import requests
import sklearn.pipeline
#from great_expectations.checkpoint.checkpoint import CheckpointResult
#from great_expectations.data_context import AbstractDataContext
#from minio import Minio
from prefect import task
from prefect.artifacts import create_markdown_artifact
from prefect.filesystems import RemoteFileSystem
from sklearn.cluster import KMeans
# from weather.data.load_datasets import (
#     extract_dataset_info,
#     get_extraction_url_from_dvc,
#     load_prep_dataset_from_minio,
#     load_raw_datasets_from_minio,
# )
from weather.data.prep_datasets import (
    Dataset,
    # prepare_and_merge_splits_to_dataset,
    prepare_binary_classification_tabular_data,
)
#from weather.features.skl_build_features import AdvFeatureNames, make_advanced_data_transformer
#from weather.gx.builders import get_context
#from weather.gx.runners import run_pipeline_checkpoint_from_df
from weather.helpers.utils import clean_temporary_dir, create_temporary_dir_if_not_exists
#from weather.mlflow.tracking import get_raw_artifacts_from_run
from weather.models.skl_train_models import score_evaluation_dict
#from weather.models.skl_validate_models import min_perf_validation
from weather.pipelines.definitions import (
    #COLUMNS,
    MINIO_ACCESS_KEY,
    MINIO_API_HOST,
    MINIO_SECRET_KEY,
    SERVER_API_URL,
    #cat_cols_wo_customer,
    #num_cols_wo_customer,
    #person_info_cols_cat,
    #person_info_cols_num,
)

import mlflow


def stop_mlflow_run(flow, flow_run, state):
    """Utilitary function to stop the current run after flow ends

    Note:
        Parameters are necessary, if not Prefect complains.
    """
    mlflow.end_run()


def make_mlflow_artifact_uri(experiment_id: str = None) -> str:
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


# def make_gx_markdown_from_results(results: CheckpointResult) -> str:
#     return f"```json\n{results.to_json_dict()}\n```"


def log_metrics(score_dict: Dict[str, float|str]):
    """Logs metrics to mlflow

    Not intended to be a prefect flow since we don't
    want to setup all the MLFlow setup. So we assume
    the MLFlow artifact tracking is set correctly.
    """
    mlflow.log_metric("train_"+score_dict["score_name"], score_dict["train"])
    mlflow.log_metric("val_"+score_dict["score_name"], score_dict["val"])
    mlflow.log_metric("test_"+score_dict["score_name"], score_dict["test"])
    # Or almost the same:
    # mlflow.log_metrics(metrics)


@task
def load_extraction_from_dvc(dvc_block: RemoteFileSystem, dvc_remote: str = None) -> Tuple[pd.DataFrame, dict]:
    url = get_extraction_url_from_dvc(remote=dvc_remote)
    with dvc_block.filesystem.open(url, "rb") as f:
        df_extract = pd.read_csv(f, sep=";")

    dvc_info = extract_dataset_info(data_dir="data", with_dvc_info=True, with_vcs_info=False)

    return df_extract, dvc_info


# @task
# def raw_data_extraction(curr_data_bucket: str) -> pd.DataFrame:
#     minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY,
#                          secret_key=MINIO_SECRET_KEY, secure=False)
#     dataframes, _ = load_raw_datasets_from_minio(minio_client, curr_data_bucket)
#     raw_data = pd.concat(dataframes, ignore_index=True)
#     return raw_data

# @task
# def prep_data_construction(ref_data_bucket: str, curr_data_bucket: str) -> pd.DataFrame:
#     minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY,
#                          secret_key=MINIO_SECRET_KEY, secure=False)
#     dataset = load_prep_dataset_from_minio(minio_client, ref_data_bucket)
#     dataframes, ds_info = load_raw_datasets_from_minio(minio_client, curr_data_bucket)
#     predictors = list(COLUMNS)
#     predictors.remove("curr_outcome")
#     predictors.remove("comm_duration")
#     predicted = "curr_outcome"
#     dataset = prepare_and_merge_splits_to_dataset(dataset, dataframes, predictors, predicted, pos_neg_pair=("yes", "no"))
#     return dataset, ds_info


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
def build_transformer(params: dict) -> sklearn.pipeline.Pipeline:
    adv_feature_names = AdvFeatureNames(
        person_info_cols_num, person_info_cols_cat, num_cols_wo_customer, cat_cols_wo_customer
    )
    transformer = make_advanced_data_transformer(adv_feature_names, KMeans, params)
    return transformer


@task
def fit_transformer(
    predictors_feature_engineering_transformer: sklearn.pipeline.Pipeline,
    dataset: Dataset
):
    predictors_feature_engineering_transformer.fit(dataset.train_x)
    return predictors_feature_engineering_transformer


# @task
# def data_validation(dataframe: pd.DataFrame, gx_expectation_suite_name: str = "latest_on_demand") -> bool:
#     """Runs data validation using a Greater Expectation specified expectation suite with provided dataframe

#     Args:
#         dataframe: The pandas dataframe against which to returns the expectations.
#         gx_expectation_suite_name: Name of the expectation suite

#     Returns:
#         Boolean value indicating if the expectation suite passed
#     """
#     context: AbstractDataContext = get_context(create=False)

#     results = run_pipeline_checkpoint_from_df(context, gx_expectation_suite_name, dataframe, run_name=prefect.runtime.flow_run.get_name())

#     create_markdown_artifact(markdown=make_gx_markdown_from_results(results), description="Results of Great Expectations data validation.")

#     return results.success


@task
def score(model, transformer, dataset, metric) -> Dict[str, float|str]:
    """Returns the evaluation metrics"""
    score_dict = score_evaluation_dict(
        metric, # alias score
        transformer,
        model,
        dataset,
    )
    return score_dict # with keys "score_name", "train", "val", "test"


@task
def load_artifacts_from_mlflow(run):
    with tempfile.TemporaryDirectory() as d:
        best_feat_eng_obj, best_classifier_obj = get_raw_artifacts_from_run(
            mlflow.get_tracking_uri(), run=run, tmp_dir_path=d
        )
        return joblib.load(best_feat_eng_obj), joblib.load(best_classifier_obj)

@task
def validate_model(dataset, data_transformer, classifier, run_id):
    cat_features = person_info_cols_cat + cat_cols_wo_customer
    result = min_perf_validation(dataset, data_transformer, classifier, cat_features)

    tmp_dir = create_temporary_dir_if_not_exists()
    tmp_fpath = lambda fpath: os.path.join(tmp_dir, fpath)

    with open(tmp_fpath("deepchecks_report.json"), "w") as f:
        f.write(result.to_json())
    result.save_as_html(tmp_fpath("deepchecks_report.html"))

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifacts("tmp", artifact_path="tests")

    clean_temporary_dir()

    return result

@task
def deploy():
    result = requests.get(SERVER_API_URL)
    return result.json()


