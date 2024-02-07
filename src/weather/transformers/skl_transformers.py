from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


# Transformers for "dataset_transformer"


class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    """Rename the columns of the dataframe with nicely formatted names, e.g. remove units."""

    def __init__(self, oldnames_newnames_dict):
        self.oldnames_newnames_dict = oldnames_newnames_dict

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        data.rename(columns=self.oldnames_newnames_dict, inplace=True)
        return data

class RemoveUselessColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, useless_column_names):
        self.useless_column_names = useless_column_names

    def fit(self, x: pd.DataFrame = None, y=None):
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        for column_name in self.useless_column_names:
            data.drop([column_name], axis=1, inplace=True)
        return data
    
    
class ConvertTimestampIntoDatetimeAndSetUTCtimezoneTransformer(BaseEstimator, TransformerMixin):
    """Convert the dtype of the column "Timestamp" to pd.Timestamp, and set the timezone to UTC."""

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)
        return data


class TimestampOrderedTransformer(BaseEstimator, TransformerMixin):
    """Check that the dataframe  is ordered by `Timestamp`, and order it if necessary."""

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        if not data["Timestamp"].is_monotonic_increasing:      # True if there are NaNs in "Timestamp"
            data.sort_values(by="Timestamp", inplace=True)     # Rows with NaN at "Timestamp" are put at the end.
            data.dropna(subset=["Timestamp"], inplace=True)    # Remove rows with NaN at "Timestamp" 
        return data


class RemoveTimestampDuplicatesTransformer(BaseEstimator, TransformerMixin):
    """Remove rows of the dataframe with duplicate values in the column `Timestamp`."""

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # Performing the transformation
        data = x.copy()
        data = data.drop_duplicates(subset=["Timestamp"], keep="last")
        return data


class TimestampAsIndexTransformer(BaseEstimator, TransformerMixin):
    """Set the "Timestamp" column as the index of the dataframe."""

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        assert "Timestamp" in x.columns, "The `Timsetamp` column is missing in the dataframe."
        data = x.copy()
        data.set_index("Timestamp", inplace=True)  # TODO: Remove hardcoded "Timestamp"
        return data


class AddEmptyRowsAtMissingTimestampsTransformer(BaseEstimator, TransformerMixin):
    "Add np.nan rows add missing timestamps of the datafreame."

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex), "Wrong dataframe index type."
        data = x.copy()
        data_min_timestamp = data.index.min()
        data_max_timestamp = data.index.max()
        regular_timestamp_range = pd.date_range(start=data_min_timestamp, end=data_max_timestamp, freq="H")
        data = data.reindex(regular_timestamp_range, copy=True)
        return data


# Transformers for "predictors_feature_engineering_transformer"


def extract_year_month_day_hour_from_data_index(data: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex), "Wrong dataframe index type."
    data["Year"] = pd.Series(data.index).dt.year.to_numpy()
    data["Month"] = pd.Series(data.index).dt.month.to_numpy()
    data["Day"] = pd.Series(data.index).dt.day.to_numpy()
    data["Hour"] = pd.Series(data.index).dt.hour.to_numpy()
    return data


class AddColumnsYearMonthDayHourFromIndexTransformer(BaseEstimator, TransformerMixin):
    """Ensure the "Timestamp" column is present, convert it to a pd.Timestamp, derive from it and
    add to the dataframe the columns  "Year", "Month", "Day", "Hour".
    """

    def fit(self, x: pd.DataFrame, y=None) -> None:
        self.column_names = x.columns.tolist() + ["Year", "Month", "Day", "Hour"]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex), "Wrong dataframe index type."
        data = x.copy()
        data = extract_year_month_day_hour_from_data_index(data)
        return data

    def get_feature_names_out(self) -> List[str]:
        return self.column_names


class FillInitialRowsWithBfillTransformer(BaseEstimator, TransformerMixin):
    """From the dataframe, remove all rows containing at least a `NaN` value."""

    def __init__(self, number_of_rows):
        self.number_of_rows = number_of_rows

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        data[: self.number_of_rows] = data.bfill()[: self.number_of_rows]
        return data


class NaNsImputationTransformer(BaseEstimator, TransformerMixin):
    """Apply df.ffill() method to categorical and numerical columns.
    TODO: Look for  linear interpolation based on the previous two rows, 
    and  implement it for  numerical columns.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_transformed = x.ffill()
        return x_transformed


class ImputeOutliersTransformer(BaseEstimator, TransformerMixin):
    """Replace, in the dataframe, values equal to `value` in column `column_name`
    by the `ffill` method (aka by the previous value if it exists, or by "NaN").
    """

    def __init__(self, column_name, value=0.0):
        self.column_name = column_name
        self.value = value

    def fit(self, x: pd.DataFrame = None, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # Performing the transformation
        data = x.copy()
        data[self.column_name].replace(self.value, np.nan, inplace=True)
        data[self.column_name].ffill(inplace=True)
        return data


class OneHotEncoderDataFrame(BaseEstimator, TransformerMixin):
    """
    TODO: toarray() decompresses the sparse array...
    """

    def __init__(self, **kwargs):
        self.encoder = OneHotEncoder(**kwargs)
        self.column_names = None

    def fit(self, x: pd.DataFrame, y=None) -> None:
        self.encoder.fit(x)
        self.column_names = self.encoder.get_feature_names_out(x.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.encoder.transform(x).toarray(), columns=self.column_names)

    def get_feature_names_out(self) -> List[str]:
        return self.column_names



# Transformers for "target_creation_transformer"


class WeatherTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.no_rain_definition = {"snow": "no_rain", "clear": "no_rain"}
        self.label_codes = {'rain': 1, 'no_rain': 0}

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self.feature_name in x.columns, f"No {self.feature_name} column in the dataframe."
        data = x.copy()
        # Hypothetical NaN of `self.feature_name` columns first entry imputed with bfill().
        data[self.feature_name][:1] = data[self.feature_name].bfill()[:1] 
        # Nan labels of `self.feature_name` cannot be encoded by 1 or 0, we impute them with method ffill().
        data[self.feature_name] = data[self.feature_name].ffill() 
        # Binarization of the target
        data[self.feature_name] = data[self.feature_name].replace(self.no_rain_definition) 
        # Label encoding 
        data[self.feature_name] = data[self.feature_name].map(self.label_codes)               
        return data                                          


class CreateShiftedWeatherSeriesTransformer(BaseEstimator, TransformerMixin):
    """Create a series, the same length as the dataframe, with `feature_name`, "Weather",
    shiffted by `nb_of_hours` hours, so that the Series indicates at timestamp t the "Weather"
    of timestamp t + `number_of_hours`.
    """

    def __init__(self, number_of_hours: int, feature_name: str):
        self.number_of_hours = number_of_hours
        self.feature_name = feature_name

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.Series:
        data = x.copy()
        shifted_data = data.shift(-self.number_of_hours)
        ground_truth = shifted_data[self.feature_name].iloc[:-self.number_of_hours]
        return ground_truth # Series, thus no column name.


# Transformers for "remove_horizonless_rows_transformer"


class RemoveHorizonLessRowsTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that removes the last `n_rows` rows from a DataFrame.

    Parameters:
    - n_rows: int. The number of rows to remove from the end of the DataFrame.
    """
    def __init__(self, n_rows=1):
        self.n_rows = n_rows

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        assert isinstance(x, (pd.Series, pd.DataFrame)), "`x` must be a pd.Series or a pd.DataFrame."
        data = x.copy()
        shortened_data = data.iloc[: -self.n_rows]
        return shortened_data