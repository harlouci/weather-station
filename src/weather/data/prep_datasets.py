from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class Dataset:
    """A dataclass used to represent a Dataset

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

    # def merge_in(self, dataset):
    #     self.train_x = pd.concat([self.train_x, dataset.train_x], axis=0)
    #     self.val_x = pd.concat([self.val_x, dataset.val_x], axis=0)
    #     self.test_x = pd.concat([self.test_x, dataset.test_x], axis=0)
    #     self.train_y = pd.concat([self.train_y, dataset.train_y], axis=0)
    #     self.val_y = pd.concat([self.val_y, dataset.val_y], axis=0)
    #     self.test_y = pd.concat([self.test_y, dataset.test_y], axis=0)

    def persist(self, dirpath):
        self.train_x.to_csv(Path(dirpath)/"train_x.csv", sep=";", index=False)
        self.train_y.to_csv(Path(dirpath)/"train_y.csv", sep=";", index=False)
        self.val_x.to_csv(Path(dirpath)/"val_x.csv", sep=";", index=False)
        self.val_y.to_csv(Path(dirpath)/"val_y.csv", sep=";", index=False)
        self.test_x.to_csv(Path(dirpath)/"test_x.csv", sep=";", index=False)
        self.test_y.to_csv(Path(dirpath)/"test_y.csv", sep=";", index=False)


def spliting_data(data: pd.DataFrame, split_size: Tuple[float] = (0.7, 0.1, 0.2)):
    data_points = data.shape[0]
    training_points = int(split_size[0] * data_points)
    valid_points = int(split_size[1] * data_points)

    training_data = data[:training_points]
    valid_data = data[training_points : training_points + valid_points]
    test_data = data[training_points + valid_points :]
    return training_data, valid_data, test_data


def prepare_data(split_data: Tuple[pd.DataFrame], training_transform, test_transform):
    train = training_transform.fit_transform(split_data[0])
    valid = training_transform.transform(split_data[1])
    test = test_transform.fit_transform(split_data[2])
    return train, valid, test


def remove_last_n_rows(data: pd.DataFrame, n: int) -> pd.DataFrame:
    return data[:-n]


def make_dataset(
    data: pd.DataFrame,
    training_transform,
    test_transform,
    target_transform,
    remove_last_rows_transformer,
    split_size: Tuple[float] = (0.7, 0.1, 0.2),
):
    split_data: Tuple[pd.DataFrame] = spliting_data(data, split_size)
    split_data = prepare_data(split_data, training_transform, test_transform)
    target_train = target_transform.fit_transform(split_data[0])
    target_val = target_transform.transform(split_data[1])
    target_test = target_transform.transform(split_data[2])

    dataset = Dataset(
        train_x=remove_last_rows_transformer.transform(split_data[0]),
        val_x=remove_last_rows_transformer.transform(split_data[1]),
        test_x=remove_last_rows_transformer.transform(split_data[2]),
        train_y=remove_last_rows_transformer.transform(target_train),
        val_y=remove_last_rows_transformer.transform(target_val),
        test_y=remove_last_rows_transformer.transform(target_test),
    )
    return dataset
