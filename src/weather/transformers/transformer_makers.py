from dataclasses import dataclass
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from weather.transformers.transformer_utilities import (
    DataFrameColumnTransformer,
    SimpleCustomPipeline,
    TransformerToDataFrame,
)
from weather.transformers.transformers import (
    AddEmptyRowsAtMissingTimestampsTransformer,
    AddFromIndexTheColumnsYearMonthDayHourTransformer,
    ConvertTimestampIntoDatetimeAndSetUTCtimezoneTransformer,
    CreateShiftedWeatherSeriesTransformer,
    FillInitialRowsWithBfillTransformer,
    ImputeOutliersTransformer,
    NaNsImputationTransformer,
    OneHotEncoderDataFrame,
    RemoveHorizonLessRowsTransformer,
    RemoveTimestampDuplicatesTransformer,
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


def make_dataset_transformer(target_choice: TargetChoice, oldnames_newnames_dict, number_of_rows=2):
    dataset_transformer = Pipeline(
        [
            ("rename_columns", RenameColumnsTransformer(oldnames_newnames_dict)),
            ("fill_initial_rows_nans", FillInitialRowsWithBfillTransformer(number_of_rows)),
            ("timestamp_as_datetime_at_utc_timezone", ConvertTimestampIntoDatetimeAndSetUTCtimezoneTransformer()),
            ("timestamp_ordered", TimestampOrderedTransformer()),
            ("remove_timestamp_duplicates", RemoveTimestampDuplicatesTransformer()),
            ("timestamp_as_index", TimestampAsIndexTransformer()),
            ("add_empty_rows_at_missing_timestamps", AddEmptyRowsAtMissingTimestampsTransformer()),
        ]
    )
    return dataset_transformer


def make_predictors_feature_engineering_transformer(feature_names: FeatureNames, target_choice: TargetChoice):

    # Impute NaNs for all columns with ffill #
    nans_imputation_transformer = NaNsImputationTransformer()

    # Impute outliers (values 0 in "Pressure" and "Humidity")
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
            ("add_columns_year_month_day_hour", AddFromIndexTheColumnsYearMonthDayHourTransformer()),
            ("weather", WeatherTransformer("Weather")),
            ("NaNs", nans_imputation_transformer),
            ("outliers", outliers_imputation_transformer),
            ("basic", merge_processor),
        ]
    )

    return predictors_feature_engineering_transformer


def make_target_creation_transformer(target_choice: TargetChoice):
    target_transformer = Pipeline(
        [
            ("weather", WeatherTransformer(target_choice.input_name)),
            ("step", CreateShiftedWeatherSeriesTransformer(target_choice.hours, target_choice.input_name)),
        ]
    )
    return target_transformer


def make_remove_horizonless_rows_transformer(target_choice: TargetChoice):
    return RemoveHorizonLessRowsTransformer(target_choice.hours)
