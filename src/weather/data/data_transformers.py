from dataclasses import dataclass

from sklearn.pipeline import Pipeline



from src.weather.features.feature_transformer import WeatherConditionTransformer, StepTransformer, RemoveNaTransformer, RemoveNoFuture
from src.weather.features.dataframe_transformer import RemoveLastNRowsTransformer


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


def make_target_transformer(target_choice: TargetChoice):
    target_transformer = Pipeline([
        ('weather', WeatherConditionTransformer(target_choice.input_name)),
        ('step', StepTransformer(target_choice.hours, target_choice.input_name)),
    ])
    return target_transformer

def make_cleaning_transformer(target_choice: TargetChoice):
    training_data_filter = Pipeline([
        ('remove_na', RemoveNaTransformer()),
        ('remove_no_futur', RemoveNoFuture(target_choice.hours)),
    ])
    return training_data_filter

def make_remove_last_rows_transformer(target_choice: TargetChoice):
    return RemoveLastNRowsTransformer(target_choice.hours)
