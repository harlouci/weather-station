from dataclasses import dataclass
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather.transformers.skl_transformer_utilities import (
    DataFrameColumnTransformer,
    SimpleCustomPipeline,
    TransformerToDataFrame,
)
from weather.transformers.skl_transformers import (
    AddColumnsYearMonthDayHourFromIndexTransformer,
    AddEmptyRowsAtMissingTimestampsTransformer,
    ConvertTimestampIntoDatetimeAndSetUTCtimezoneTransformer,
    CreateShiftedWeatherSeriesTransformer,
    FillInitialRowsWithBfillTransformer,
    ImputeOutliersTransformer,
    NaNsImputationTransformer,
    OneHotEncoderDataFrame,
    RemoveHorizonLessRowsTransformer,
    RemoveTimestampDuplicatesTransformer,
    RemoveUselessColumnsTransformer,
    RenameColumnsTransformer,
    TimestampAsIndexTransformer,
    TimestampOrderedTransformer,
    WeatherTransformer,
)


@dataclass
class FeatureNames:
    """A dataclass used to represent Feature Names for a Basic ML Model

    Attributes
    ----------
    numerical : List[str]
        the list of numerical feature names
    categorical : List[str]
        the list of categorical feature names

    Methods
    -------
    features()
        Returns the list of all features
    """

    numerical: List[str]
    categorical: List[str]

    def features(self) -> List[str]:
        """Returns the list of all features

        Returns
        -------
            List[str]: list of all features
        """
        return self.numerical + self.categorical


@dataclass
class TargetChoice:
    """
    a dataclass used to determine what is the target we want to predict

    input_name : str
        name of the variable we want to predict
    hours : int
        number of hours in the future we want to predict
    """

    input_name: str
    hours: int


def make_dataset_ingestion_transformer(
    target_choice: TargetChoice,
    oldnames_newnames_dict: dict,
    useless_column_names: tuple = ("S_No", "Location", "Apparent_temperature"),
):
    """This preprocessing pipeline would represent the data ingestion step of a pipeline.

    The dataset is stll raw, but with readable column names and a consistent timestamp set as the index.
    """
    dataset_ingestion_transformer = Pipeline(
        [
            ("rename_columns", RenameColumnsTransformer(oldnames_newnames_dict)),
            ("remove_useless_columns", RemoveUselessColumnsTransformer(useless_column_names)),
            ("timestamp_as_datetime_at_utc_timezone", ConvertTimestampIntoDatetimeAndSetUTCtimezoneTransformer()),
            ("order_timestamp", TimestampOrderedTransformer()),
            ("remove_timestamp_duplicates", RemoveTimestampDuplicatesTransformer()),
            ("timestamp_as_index", TimestampAsIndexTransformer()),
            ("add_empty_rows_at_missing_timestamps", AddEmptyRowsAtMissingTimestampsTransformer()),
        ]
    )
    return dataset_ingestion_transformer


def make_target_creation_transformer(target_choice: TargetChoice):
    """Creates the pipeline `target_creation_transformer` that produces the ground truth.

    `target_creation_transformer` must always be preceded by `dataset_transformer`."""
    target_creation_transformer = Pipeline(
        [
            ("weather", WeatherTransformer(target_choice.input_name)),
            ("ground_truth", CreateShiftedWeatherSeriesTransformer(target_choice.hours, target_choice.input_name)),
        ]
    )
    return target_creation_transformer


def make_remove_horizonless_rows_transformer(target_choice: TargetChoice):
    return RemoveHorizonLessRowsTransformer(target_choice.hours)


def make_predictors_feature_engineering_transformer(
    feature_names: FeatureNames,
    target_choice: TargetChoice,
    number_of_rows=1,
):
    # Impute outliers (0 values in "Pressure" and "Humidity")
    outliers_imputation_transformer = Pipeline(
        [
            ("inpute_humidity_outliers", ImputeOutliersTransformer("Humidity")),
            ("inpute_pressure_outliers", ImputeOutliersTransformer("Pressure")),
        ]
    )

    # For one-hot encoding of categorical columns
    categorical_transformer = SimpleCustomPipeline(  # Problem with Pipeline alone because of ColumnTransformer, which is used in DataFramColumnTransformer
        [
            ("onehot", OneHotEncoderDataFrame(handle_unknown="ignore")),
        ]
    )

    # For scaling numerical columns
    numerical_transformer = SimpleCustomPipeline(
        [
            ("scaler", TransformerToDataFrame(StandardScaler())),
        ]
    )

    merge_processor = DataFrameColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, feature_names.categorical),
            ("num", numerical_transformer, feature_names.numerical),
        ]
    )

    predictors_feature_engineering_transformer = SimpleCustomPipeline(
        [
            ("add_columns_year_month_day_hour", AddColumnsYearMonthDayHourFromIndexTransformer()),
            (
                "weather",
                WeatherTransformer("Weather"),
            ),  # Remove NaNs in "Weather" ; create "no_rain" ; encode "rain":1, "no_rain":0
            ("fill_initial_rows_nans", FillInitialRowsWithBfillTransformer(number_of_rows)),
            ("nans_imputation", NaNsImputationTransformer()),  # Impute NaNs in all columns with ffill()
            ("outliers", outliers_imputation_transformer),  # Humidity and Pressure columns
            ("one_hot_encoder_and_standard_scaler", merge_processor),
        ]
    )

    return predictors_feature_engineering_transformer
