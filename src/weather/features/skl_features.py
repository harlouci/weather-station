
from dataclasses import dataclass
from typing import List


from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from src.weather.features.feature_transformer import WeatherConditionTransformer, StepTransformer, RemoveNaTransformer, RemoveNoFuture, DateTransformer
from src.weather.features.dataframe_transformer import SimpleCustomPipeline,TransformerToDataFrame, OneHotEncoderDataFrame, DataFrameColumnTransformer

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
    input_name:str
    hours:int

def make_input_transformer(feature_names:FeatureNames):

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


def make_target_transformer(target_choice: TargetChoice):
    target_transformer = Pipeline([
        ('weather', WeatherConditionTransformer(target_choice.input_name)),
        ('step', StepTransformer(target_choice.hours, target_choice.input_name))
    ])
    return target_transformer

def make_filter_transformer(target_choice: TargetChoice):
    training_data_filter = Pipeline([
        ('remove_na', RemoveNaTransformer()),
        ('remove_no_futur', RemoveNoFuture(target_choice.hours))
    ])
    return training_data_filter