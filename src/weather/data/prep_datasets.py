"""Includes functions to prepare datasets for ML applications."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class Dataset:
    """A dataclass used to represent a Dataset.

    Attributes
    ----------
    train_x : Pandas DataFrame
        the dataframe of input features w.r.t training split
    val_x : Pandas DataFrame
        the dataframe of input features w.r.t validation split
    test_x : Pandas DataFrame
        the dataframe of input features w.r.t testing split
    train_y : Pandas Series
        the series of output label w.r.t training split
    val_y : Pandas Series
        tthe series of output label w.r.t validation split
    test_y : Pandas Series
        the series of output label w.r.t testing split
    """

    train_x: pd.DataFrame
    val_x: pd.DataFrame
    test_x: pd.DataFrame
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series
    train_val_x: pd.DataFrame = None
    train_val_y: pd.Series = None

    def concatenate_train_and_val_splits(self):
        self.train_val_x = pd.concat([self.train_x, self.val_x])
        self.train_val_y = pd.concat([self.train_y, self.val_y])

    # TODO: remove this commented out chunks
    # def merge_in(self, dataset):
    #     self.train_x = pd.concat([self.train_x, dataset.train_x], axis=0)
    #     self.val_x = pd.concat([self.val_x, dataset.val_x], axis=0)
    #     self.test_x = pd.concat([self.test_x, dataset.test_x], axis=0)
    #     self.train_y = pd.concat([self.train_y, dataset.train_y], axis=0)
    #     self.val_y = pd.concat([self.val_y, dataset.val_y], axis=0)
    #     self.test_y = pd.concat([self.test_y, dataset.test_y], axis=0)

    # def apply_transformer(self, transformer):
    #     self.train_x = transformer.transform(self.train_x)
    #     self.val_x = transformer.transform(self.val_x)
    #     self.test_x = transformer.transform(self.test_x)
    #     self.train_y = transformer.transform(self.train_y)
    #     self.val_y = transformer.transform(self.val_y)
    #     self.test_y = transformer.transform(self.test_y)
    #     return self

    # def apply_transformer_to_test_split(self, transformer):
    #     self.test_x = transformer.transform(self.test_x)
    #     self.test_y = transformer.transform(self.test_y)
    #     return self

    def persist(self, dirpath):
        self.train_x.to_csv(Path(dirpath) / "train_x.csv", sep=";", index=False)
        self.train_y.to_csv(Path(dirpath) / "train_y.csv", sep=";", index=False)
        self.val_x.to_csv(Path(dirpath) / "val_x.csv", sep=";", index=False)
        self.val_y.to_csv(Path(dirpath) / "val_y.csv", sep=";", index=False)
        self.test_x.to_csv(Path(dirpath) / "test_x.csv", sep=";", index=False)
        self.test_y.to_csv(Path(dirpath) / "test_y.csv", sep=";", index=False)


def split_data(data: pd.DataFrame, split_size: Tuple[float] = (0.7, 0.1, 0.2)):
    """Split the dataframe in train/val/test datasets without shuffling the rows.
    The train, val, test datasets are chronologically ordered from past to present.
    """
    data_points = data.shape[0]
    training_points = int(split_size[0] * data_points)
    valid_points = int(split_size[1] * data_points)

    train_data = data[:training_points]
    val_data = data[training_points : training_points + valid_points]
    test_data = data[training_points + valid_points :]
    return train_data, val_data, test_data


def transform_dataset_and_create_target(
    data: pd.DataFrame,
    dataset_ingestion_transformer,
    remove_horizonless_rows_transformer,
    target_creation_transformer,
):
    ingested_data = dataset_ingestion_transformer.transform(data)
    transformed_data = remove_horizonless_rows_transformer.transform(ingested_data)
    created_target = target_creation_transformer.fit_transform(ingested_data)
    return transformed_data, created_target


def prepare_binary_classification_tabular_data(
    transformed_data: pd.DataFrame,
    created_target,
    split_size: Tuple[float] = (0.7, 0.1, 0.2),
):
    splitted_data: Tuple[pd.DataFrame] = split_data(transformed_data, split_size)
    splitted_target: Tuple[pd.Series] = split_data(created_target, split_size)

    dataset = Dataset(
        train_x=splitted_data[0],
        val_x=splitted_data[1],
        test_x=splitted_data[2],
        train_y=splitted_target[0],
        val_y=splitted_target[1],
        test_y=splitted_target[2],
    )
    return dataset
