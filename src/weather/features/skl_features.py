from dataclasses import dataclass
from typing import List

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from weather.data.data_transformers import TargetChoice
from weather.features.dataframe_transformer import SimpleCustomPipeline,TransformerToDataFrame, OneHotEncoderDataFrame, DataFrameColumnTransformer
from weather.features.feature_transformer import WeatherConditionTransformer, StepTransformer, RemoveNaTransformer, RemoveNoFuture, DateTransformer

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



def make_input_transformer(feature_names:FeatureNames, target_choice:TargetChoice):

    # For one-hot encoding of categorical columns
    categorical_transformer = SimpleCustomPipeline([
        ('imputer', TransformerToDataFrame(SimpleImputer(strategy='most_frequent'))),  # Handle missing values if any
        ('onehot', OneHotEncoderDataFrame(handle_unknown='ignore'))
    ])

    # For scaling numerical columns
    numerical_transformer = SimpleCustomPipeline([
        ('imputer', TransformerToDataFrame(SimpleImputer(strategy='mean'))),  # Handle missing values if any
        ('scaler', TransformerToDataFrame(StandardScaler()))
    ])

    merge_processor = DataFrameColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, feature_names.categorical),
            ('num', numerical_transformer, feature_names.numerical),
        ])

    input_transformer = SimpleCustomPipeline([
        ('time', DateTransformer()),
        ('weather', WeatherConditionTransformer('Weather_conditions')),
        ('basic', merge_processor),
    ])

    return input_transformer


