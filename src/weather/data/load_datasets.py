import subprocess
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from weather.data.prep_datasets import Dataset

#import dvc.api


def autodetect_commit() -> str:
    """Retrieves the current commit of the local repository

    Assumes the repository is local.
    Reads the HEAD value.

    Returns:
        The commit SHA
    """
    output = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--no-flags",
            "--tags",
            "HEAD",
        ],
        text=True,
    )
    return output.strip()


# def get_extraction_url_from_dvc(data_dir: str = "data", remote: str | None = None) -> str:
#     return dvc.api.get_url(f"{data_dir}/raw/extraction.csv", remote=remote)


def extract_dataset_info(data_dir: str = "data", with_dvc_info: bool = False, with_vcs_info: bool = False) -> dict:
    data_dir = Path(data_dir)

    dataset_info = {}

    if with_dvc_info:
        dvc_content = (data_dir / "splits.dvc").read_text()
        dataset_info["dvc"] = yaml.safe_load(dvc_content)
    if with_vcs_info:
        dataset_info["vcs"] = {}
        dataset_info["vcs"]["commit"] = autodetect_commit()

    return dataset_info


# def load_dataset_from_dvc(
#     data_dir: str = "data", with_dvc_info: bool = True, with_vcs_info: bool = True,
#     dvc_remote: str | None = None, pd_storage_options: dict | None = None,
# ) -> Tuple[Dataset, dict]:
#     """Loads a Dataset object coming from DVC


#     As of this implementation, reads the CSV files that are locally tracked
#     by DVC.

#     Args:
#         data_dir(str): Directory in which data resides. Defaults to "data/".
#         with_dvc_info(bool): If true, the DVC tracking information will be read
#                              and returned into a field in the dictionary.
#         with_vc_info(bool): If true, the VCS information will be read and returned
#                             into a field in the dictionary.
#     Returns:
#         A Tuple containing:
#             in the first position, the Dataset object
#             in the second position, the metadata information
#     """
#     data_dir = Path(data_dir)

#     dataset_info = extract_dataset_info(data_dir, with_dvc_info, with_vcs_info)

#     d = Dataset(
#         train_x=pd.read_csv(dvc.api.get_url(f"{data_dir}/splits/train_x.csv", remote=dvc_remote), sep=';', storage_options=pd_storage_options),
#         val_x=pd.read_csv(dvc.api.get_url(f"{data_dir}/splits/val_x.csv", remote=dvc_remote), sep=';', storage_options=pd_storage_options),
#         test_x=pd.read_csv(dvc.api.get_url(f"{data_dir}/splits/test_x.csv", remote=dvc_remote), sep=';', storage_options=pd_storage_options),
#         train_y=pd.read_csv(dvc.api.get_url(f"{data_dir}/splits/train_y.csv", remote=dvc_remote), sep=';', storage_options=pd_storage_options).iloc[:, 0],
#         val_y=pd.read_csv(dvc.api.get_url(f"{data_dir}/splits/val_y.csv", remote=dvc_remote), sep=';', storage_options=pd_storage_options).iloc[:, 0],
#         test_y=pd.read_csv(dvc.api.get_url(f"{data_dir}/splits/test_y.csv", remote=dvc_remote), sep=';', storage_options=pd_storage_options).iloc[:, 0],
#     )

#     return d, dataset_info


def load_dataset_from_localfs(
    data_dir: str = "data", with_dvc_info: bool = True, with_vcs_info: bool = True
) -> Tuple[Dataset, dict]:
    """Loads a Dataset object coming from DVC


    As of this implementation, reads the CSV files that are locally stored

    Args:
        data_dir(str): Directory in which data resides. Defaults to "data/".
    Returns:
        A Tuple containing:
            in the first position, the Dataset object
            in the second position, the metadata information
    """
    data_dir = Path(data_dir)

    dataset_info = {}

    d = Dataset(
        train_x=pd.read_csv(f"{data_dir}/splits/train_x.csv", sep=";"),
        val_x=pd.read_csv(f"{data_dir}/splits/val_x.csv", sep=";"),
        test_x=pd.read_csv(f"{data_dir}/splits/test_x.csv", sep=";"),
        train_y=pd.read_csv(f"{data_dir}/splits/train_y.csv", sep=";").iloc[:, 0],
        val_y=pd.read_csv(f"{data_dir}/splits/val_y.csv", sep=";").iloc[:, 0],
        test_y=pd.read_csv(f"{data_dir}/splits/test_y.csv", sep=";").iloc[:, 0]
    )

    return d, dataset_info


def load_prep_dataset_from_minio(
    minio_client,
    data_bucket: str = "reference-data",
) -> Dataset:

    d = Dataset(
        train_x=pd.read_csv(minio_client.get_object(data_bucket, "train_x.csv"), sep=",", index_col=0, parse_dates=True), # pd.DataFrame
        val_x=pd.read_csv(minio_client.get_object(data_bucket, "val_x.csv"), sep=",", index_col=0, parse_dates=True),
        test_x=pd.read_csv(minio_client.get_object(data_bucket, "test_x.csv"), sep=",", index_col=0, parse_dates=True),
        train_y=pd.read_csv(minio_client.get_object(data_bucket, "train_y.csv"), sep=",").iloc[:, 0], # pd.Series
        val_y=pd.read_csv(minio_client.get_object(data_bucket, "val_y.csv"), sep=",").iloc[:, 0],
        test_y=pd.read_csv(minio_client.get_object(data_bucket, "test_y.csv"), sep=",").iloc[:, 0],
    )

    return d


def load_raw_datasets_from_minio(
    minio_client,
    data_bucket: str = "current-data"):
    dataframes = []
    ds_info = {}
    objects = minio_client.list_objects(data_bucket)
    for obj in objects:
        name = obj.object_name
        obj = minio_client.get_object(
            data_bucket,
            obj.object_name,
        )
        df = pd.read_csv(obj)
        ds_info[name] = len(df)
        dataframes.append(df)
    return dataframes, ds_info
