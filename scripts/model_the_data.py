import joblib
import pandas as pd
from pathlib import Path
from weather.data.data_transformers import (
    TargetChoice,
    make_cleaning_transformer,
    make_remove_last_rows_transformer,
    make_target_transformer,
)
from weather.data.prep_datasets import make_dataset
from weather.features.skl_features import FeatureNames, make_input_transformer
from weather.models.models import models
from weather.models.skl_train_models import accuracy_evaluation, print_accuracy_results

df = pd.read_csv("../data/weather_dataset_raw.csv")

target_choice = TargetChoice("Weather_conditions", 4)
feature_names = FeatureNames(
    numerical=[
        "Temperature_C",
        "Humidity",
        "Wind_speed_kmph",
        "Wind_bearing_degrees",
        "Visibility_km",
        "Pressure_millibars",
    ],
    categorical=["Weather_conditions", "Month"],
)

cleaning_transformer = make_cleaning_transformer(target_choice)
target_tranformer = make_target_transformer(target_choice)
input_transformer = make_input_transformer(feature_names, target_choice)
remove_last_rows_transformer = make_remove_last_rows_transformer(target_choice)


df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
df.sort_values(by="Timestamp", ascending=True, inplace=True)

dataset = make_dataset(
    data=df,
    training_transform=cleaning_transformer,
    test_transform=cleaning_transformer,
    target_transform=target_tranformer,
    remove_last_rows_transformer=remove_last_rows_transformer,
)

print("percentage rain", dataset.train_y.mean())

models_score = {}
best_valid_score = 0
best_valid_model = None

model = models["RandomForest"]["model"]


input_transformer.fit(dataset.train_x)
model.fit(input_transformer.transform(dataset.train_x), dataset.train_y)
results = accuracy_evaluation(input_transformer, model, dataset)
print_accuracy_results(results)

model_folder = Path(__file__).resolve().parent.parent / "model"

joblib.dump(model, model_folder / "model.pkl")
joblib.dump(input_transformer, model_folder / "feature_eng_pipeline.pkl")
