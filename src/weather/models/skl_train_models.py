"""This module includes the functions to train and evaluate scikit-learn models
for weather prediction ML applications.
"""

import abc
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from weather.data.prep_datasets import Dataset

# 1. Train-val-test score evaluation of a binary classification model


@dataclass
class Score:
    score_name: str
    train: float
    val: float
    test: float


def score_evaluation(
    score,
    predictors_feature_engineering_transformer: Pipeline,
    classifier: abc.ABCMeta,
    data: Dataset,
    decimals: int = 3,
) -> dataclass:
    """Compute binary classification scores on training/validation/testing data splits
       by a given pair of data transformer and classifier.

    Args:
    ----
        score: a scikit-learn metric function, e.g "accuracy_score", "f1_score".
        data_transfomer (Pipeline): sklearn feature engineering pipeline
        classifier (abc.ABCMeta): sklearn classifier class
        data (Dataset): datasets (training/validation/test)
        decimals (int, optional): number decimal digits of precision. Defaults to 3.

    Returns:
    -------
        dataclass: (keys: splits names, values: scores)
    """
    assert score.__name__ in [
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
    ], """
        The score name must be "accuracy_score", "precision_score", "recall_score" or "f1_score."""
    train_score = score(
        data.train_y.values, classifier.predict(predictors_feature_engineering_transformer.transform(data.train_x))
    )
    val_score = score(
        data.val_y.values, classifier.predict(predictors_feature_engineering_transformer.transform(data.val_x))
    )
    test_score = score(
        data.test_y.values, classifier.predict(predictors_feature_engineering_transformer.transform(data.test_x))
    )
    return Score(
        score.__name__,
        round(train_score, decimals),
        round(val_score, decimals),
        round(test_score, decimals),
    )


# 2.  Train-val-test confusion matrix


def confusion_matrix_evaluation(
    predictors_feature_engineering_transformer: Pipeline,
    classifier: abc.ABCMeta,
    data: Dataset,
    normalize: str | None = None,
) -> Dict[str, np.ndarray]:
    """Compute binary classification confusion matrices on training/validation/testing data splits
       by a given pair of data transformer and classifier.

    Args:
    ----
        predictors_feature_engineering_transformer (Pipeline): sklearn feature engineering pipeline
        classifier (abc.ABCMeta): sklearn classifier class
        data (Dataset): datasets (training/validation/test)
        normalize (string): {"true", "pred", "all", None}

    Returns:
    -------
        Dict[str, np.ndarray]: (keys: splits names, values: confusion matrices)
    """
    train_cm = confusion_matrix(
        data.train_y.values,
        classifier.predict(predictors_feature_engineering_transformer.transform(data.train_x)),
        labels=classifier.classes_,
        normalize=normalize,
    )
    val_cm = confusion_matrix(
        data.val_y.values,
        classifier.predict(predictors_feature_engineering_transformer.transform(data.val_x)),
        labels=classifier.classes_,
        normalize=normalize,
    )
    test_cm = confusion_matrix(
        data.test_y.values,
        classifier.predict(predictors_feature_engineering_transformer.transform(data.test_x)),
        labels=classifier.classes_,
        normalize=normalize,
    )
    return {
        "train": train_cm,
        "val": val_cm,
        "test": test_cm,
    }


def confusion_matrix_display(
    results: Dict[str, np.ndarray],
    classifier: abc.ABCMeta,
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    split_names = results.keys()
    for split_name, ax in zip(split_names, axes.flatten()):
        disp = ConfusionMatrixDisplay(confusion_matrix=results[split_name], display_labels=classifier.classes_)
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(split_name + "\n")
    plt.tight_layout()
    plt.show()


# 2. Train-val score evaluation of a binary classification model


@dataclass
class TrainValScore:
    score_name: str
    train: float
    val: float


def train_val_score_evaluation(
    score,
    predictors_feature_engineering_transformer: Pipeline,
    trained_classifier: abc.ABCMeta,
    data: Dataset,
    decimals: int = 3,
) -> dataclass:
    """ """
    assert score.__name__ in [
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
    ], """
        The score name must be "accuracy_score", "precision_score", "recall_score" or "f1_score."""
    train_score = score(
        data.train_y.values,
        trained_classifier.predict(predictors_feature_engineering_transformer.transform(data.train_x)),
    )
    val_score = score(
        data.val_y.values,
        trained_classifier.predict(predictors_feature_engineering_transformer.transform(data.val_x)),
    )
    return TrainValScore(
        score.__name__,
        round(train_score, decimals),
        round(val_score, decimals),
    )


# 3. Train-test score evaluation of a binary classification model


@dataclass
class TrainTestScore:
    score_name: str
    train_val: float
    test: float


def train_test_score_evaluation(
    score,
    predictors_feature_engineering_transformer: Pipeline,
    retrained_classifier: abc.ABCMeta,
    data: Dataset,
    decimals: int = 3,
) -> dataclass:
    """The  `train` and  `val` splits in  `data` are must be concatenated, and the classifier must
    be retrained on this split.
    """
    assert score.__name__ in [
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
    ], """
        The score name must be "accuracy_score", "precision_score", "recall_score" or "f1_score."""
    train_val_score = score(
        data.train_val_y.values,
        retrained_classifier.predict(predictors_feature_engineering_transformer.transform(data.train_val_x)),
    )
    test_score = score(
        data.test_y.values,
        retrained_classifier.predict(predictors_feature_engineering_transformer.transform(data.test_x)),
    )
    return TrainTestScore(
        score.__name__,
        round(train_val_score, decimals),
        round(test_score, decimals),
    )


#################################################### TODO: if not used, SEMLA code


# 3. Accuracy evaluation of a binary classification model


def accuracy_evaluation(
    data_transfomer: Pipeline, classifier: abc.ABCMeta, data: Dataset, decimals: int = 3
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


# 4. Train model and evaluate score with accuracy_evaluation()


def score_evaluation_dict(
    score,
    predictors_feature_engineering_transformer: Pipeline,
    classifier: abc.ABCMeta,
    data: Dataset,
    decimals: int = 3,
) -> dataclass:
    """Compute binary classification scores on training/validation/testing data splits
       by a given pair of data transformer and classifier.

    Args:
    ----
        score: a scikit-learn metric function, e.g "accuracy_score", "f1_score".
        data_transfomer (Pipeline): sklearn feature engineering pipeline
        classifier (abc.ABCMeta): sklearn classifier class
        data (Dataset): datasets (training/validation/test)
        decimals (int, optional): number decimal digits of precision. Defaults to 3.

    Returns:
    -------
        dataclass: (keys: splits names, values: scores)
    """
    assert score.__name__ in [
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
    ], """
        The score name must be "accuracy_score", "precision_score", "recall_score" or "f1_score."""
    train_score = score(
        data.train_y.values, classifier.predict(predictors_feature_engineering_transformer.transform(data.train_x))
    )
    val_score = score(
        data.val_y.values, classifier.predict(predictors_feature_engineering_transformer.transform(data.val_x))
    )
    test_score = score(
        data.test_y.values, classifier.predict(predictors_feature_engineering_transformer.transform(data.test_x))
    )
    return {
        "score_name": score.__name__,
        "train": round(train_score, decimals),
        "val": round(val_score, decimals),
        "test": round(test_score, decimals),
    }


def print_score_dict_results(results: List[Dict[str, str | float]] | Dict[str, str | float]) -> None:
    """Print the accuracy scoring results as pretty tables

    Args:
    ----
        results (List[Dict[str,str | float]] | Dict[str,str | float]): list of accuracy scores
    """
    tab = PrettyTable()
    if type(results) == dict:
        tab.field_names = list(results.keys())
        tab.add_row(list(results.values()))
    elif type(results) == list:
        tab.field_names = list(results[0].keys())
        tab.add_rows([list(result.values()) for result in results])
    print(tab)


def train_and_evaluate(
    data: Dataset,
    predictors_feature_engineering_transformer: Pipeline,
    classifers_list: List[abc.ABCMeta],
    score,
) -> List[Dict[str, str | float]]:
    """Train each classifier of the list on the training data then evaluate it on all the splits

    Args:
    ----
        data (Dataset): datasets (training/validation/test)
        predictors_feature_engineering_transformer (Pipeline): sklearn feature engineering pipeline
        classifers_list (List[abc.ABCMeta]): list of sklearn classifier classes

    Returns:
    -------
        List[Dict[str,str|float]]: A list of dictionaries, where each dictionary contains:
                                    'model': The name of the classifier class.
                                    'score_name': The name of the evaluation metric, e.g. f1_score
                                    Additional scores obtained from the score_evaluation_dict() function.
    """
    results = []
    for classifier in classifers_list:
        classifier_obj = classifier()
        classifier_obj.fit(
            predictors_feature_engineering_transformer.fit_transform(data.train_x),
            data.train_y,
        )
        results.append(
            {
                "model": classifier.__name__,
                **score_evaluation_dict(
                    score,
                    predictors_feature_engineering_transformer,
                    classifier_obj,
                    data,
                ),
            }
        )
    return results
