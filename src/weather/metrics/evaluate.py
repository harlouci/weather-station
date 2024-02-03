import abc
from dataclasses import dataclass
from typing import Dict

from sklearn.metrics import accuracy_score
from weather.data.prep_datasets import Dataset
from weather.features.dataframe_transformer import SimpleCustomPipeline


@dataclass
class Score:

    score_name: str
    train: float
    valid: float
    test: float


def score_evaluation(score, classifier: abc.ABCMeta, data: Dataset, decimals: int = 3):
    score_name = score.__name__

    train_score = score(data.train_y.values, classifier.predict(data.train_x))
    val_score = score(data.val_y.values, classifier.predict(data.val_x))
    test_score = score(data.test_y.values, classifier.predict(data.test_x))
    return Score(score_name, round(train_score, decimals), round(val_score), round(test_score))


def accuracy_evaluation(
    data_transfomer: SimpleCustomPipeline, classifier: abc.ABCMeta, data: Dataset, decimals: int = 3
) -> Dict[str, float]:
    """Compute binary classification accuracy scores on training/validation/testing data splits
       by a given pair of data transformer and classifier.

    Args:
    ----
        data_transfomer (Pipeline): sklearn feature engineering pipeline
        classifier (abc.ABCMeta): sklearn classifier class
        data (Dataset): datasets (training/validation/test)
        decimals (int, optional): number decimal digits of precision. Defaults to 3.

    Returns:
    -------
        Dict[str, float]: (keys: splits names, values: accuracy scores)
    """
    train_accuracy = accuracy_score(data.train_y.values, classifier.predict(data_transfomer.transform(data.train_x)))
    val_accuracy = accuracy_score(data.val_y.values, classifier.predict(data_transfomer.transform(data.val_x)))
    test_accuracy = accuracy_score(data.test_y.values, classifier.predict(data_transfomer.transform(data.test_x)))
    return {
        "train": round(train_accuracy, decimals),
        "val": round(val_accuracy, decimals),
        "test": round(test_accuracy, decimals),
    }
