import pandas as pd
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


def format_date(data:pd.DataFrame)->pd.DataFrame:
    # Convert the 'Timestamp' column to datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)

    # Extracting the relevant components directly from the datetime object
    data["Year"] = data['Timestamp'].dt.year
    data["Month"] = data['Timestamp'].dt.month
    data["Day"] = data['Timestamp'].dt.day
    data["Hour"] = data['Timestamp'].dt.hour  # Extracting just the hour

    return data


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None)->None:
        # This transformer does not need to learn anything from the data,
        # so the fit method just returns self.
        self.column_names = X.columns.tolist() + ["Year", "Month", "Day", "Hour"]
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        data = X.copy()
        # Check if X is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Ensure 'Timestamp' column is present
        if 'Timestamp' not in data.columns:
            raise ValueError("DataFrame must contain 'Timestamp' column for DateTransformer")

        # Convert 'Timestamp' to datetime and extract components
        data = format_date(data)

        return data

    def get_feature_names_out(self)->List[str]:
        return self.column_names


class StepTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, hours: int, feature_name: str):
        self.hours = hours
        self.feature_name = feature_name

    def fit(self, X:pd.DataFrame, y=None)->'StepTransformer':
        # This transformer does not need to learn anything from the data,
        # so the fit method just returns self.
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        data = X.copy()
        # Check if X is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        time_stamp_name = 'Timestamp'
        feature_name = self.feature_name

        # Ensure 'Timestamp' column is present

        if time_stamp_name not in data.columns:
            raise ValueError(f"DataFrame must contain {time_stamp_name} column for StepTransformer")

        # Convert 'Timestamp' to datetime if not already
        data[time_stamp_name] = pd.to_datetime(data[time_stamp_name], utc=True)

        # Compare current timestamp with the one 'steps' ahead
        future = data.shift(-self.hours)

        return future[feature_name]


class WeatherConditionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.label_encoder = LabelEncoder()
        self.feature_name = feature_name
        self.no_rain_definition = {"snow": "no_rain", "clear": "no_rain"}

    def fit(self, X:pd.DataFrame, y=None)->'WeatherConditionTransformer':
        data = X.copy()
        # Check if X is a DataFrame
        if not isinstance(data , pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Ensure 'Weather_conditions' column is present
        if self.feature_name not in data.columns:
            raise ValueError(f"DataFrame must contain {self.feature_name} column")

        # Fitting the LabelEncoder
        data[self.feature_name] = data[self.feature_name].ffill()
        data[self.feature_name] = data[self.feature_name].replace(self.no_rain_definition)
        self.label_encoder.fit(data[self.feature_name])

        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        # Performing the transformation
        data = X.copy()
        data[self.feature_name] = data[self.feature_name].ffill()
        data[self.feature_name] = data[self.feature_name].replace(self.no_rain_definition)
        encoded_weather = self.label_encoder.transform(data[self.feature_name])
        data.drop([self.feature_name], axis=1, inplace=True)
        data[self.feature_name] = encoded_weather

        return data


class RemoveNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame=None, y=None)->'RemoveNaTransformer':
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        # Performing the transformation
        data = X.copy()
        data.dropna(inplace=True)

        return data


class RemoveNoFuture(BaseEstimator, TransformerMixin):

    def __init__(self, hours: int):
        self.hours = hours

    def fit(self, X:pd.DataFrame, y=None):
        # This transformer does not need to learn anything from the data,
        # so the fit method just returns self.
        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        data = X.copy()
        # Check if X is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        time_stamp_name = 'Timestamp'

        # Ensure 'Timestamp' column is present
        if time_stamp_name not in data.columns:
            raise ValueError(f"DataFrame must contain {time_stamp_name} column")

        # Convert 'Timestamp' to datetime if not already
        data[time_stamp_name] = pd.to_datetime(data[time_stamp_name], utc=True)

        # Compare current timestamp with the one 'steps' ahead
        future = data.shift(-self.hours)

        is_future_exist = (future[time_stamp_name] - data[time_stamp_name]) == pd.Timedelta(hours=self.hours)
        data = data[is_future_exist]

        return data