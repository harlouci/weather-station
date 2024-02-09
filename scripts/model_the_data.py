import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from weather.data.prep_datasets import (
    prepare_binary_classification_tabular_data,
    transform_dataset_and_create_target,
)
from weather.helpers.utils import camel_to_snake
from weather.models.skl_train_models import (
    confusion_matrix_display,
    confusion_matrix_evaluation,
    score_evaluation,
)
from weather.transformers.skl_transformer_makers import (
    FeatureNames,
    TargetChoice,
    make_dataset_ingestion_transformer,
    make_predictors_feature_engineering_transformer,
    make_remove_horizonless_rows_transformer,
    make_target_creation_transformer,
)

data_dir =  Path.cwd().parent / "data"
models_dir = Path.cwd().parent / "models"
models_dir.mkdir(exist_ok=True)


# Select the predictors
feature_names = FeatureNames(
    numerical=[
        "Temperature",
        "Humidity",
        "Wind_speed",
        "Wind_bearing",
        "Visibility",
        "Pressure",
    ],
    categorical=[],  # Add or remove "Weather", "Month" to the predictors
)

# Set "Weather" within 4 hours as target
target_name = "Weather"
horizon = 4
target_choice = TargetChoice(target_name, horizon)


oldnames_newnames_dict = {
    "Temperature_C": "Temperature",
    "Apparent_Temperature_C": "Apparent_temperature",
    "Wind_speed_kmph": "Wind_speed",
    "Wind_bearing_degrees": "Wind_bearing",
    "Visibility_km": "Visibility",
    "Pressure_millibars": "Pressure",
    "Weather_conditions": "Weather"}

dataset_ingestion_transformer = make_dataset_ingestion_transformer(target_choice, oldnames_newnames_dict)
remove_horizonless_rows_transformer = make_remove_horizonless_rows_transformer(target_choice)
target_creation_transformer = make_target_creation_transformer(target_choice)
predictors_feature_engineering_transformer = make_predictors_feature_engineering_transformer(feature_names, target_choice)

# read the data
df = pd.read_csv(data_dir / "weather_dataset_raw_development.csv")
df.head(1)

# Transform the dataset and split it
# Three transformers: "dataset__ingestion_transformer", "remove_horizonless_rows_transformer", "target_creation_transformer"
transformed_data, created_target = transform_dataset_and_create_target(
    df,
    dataset_ingestion_transformer,
    remove_horizonless_rows_transformer,
    target_creation_transformer,
)

# Split the dataset
dataset = prepare_binary_classification_tabular_data(
    transformed_data,
    created_target,
)


random_state = 1234

models = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(max_depth=4, random_state=random_state),
    },
    "LinearSvc": {
        "model": LinearSVC(max_iter=10_000, random_state=random_state),
    },
    "LogisticRegression": {
        "model": LogisticRegression(),
    },
    "RandomForest": {
        "model": RandomForestClassifier(max_depth=4, random_state=random_state),
        #"param_grid": {"model__n_estimators": [5, 10], "model__max_depth": [None, 5, 10]},
    },
    "SvcWithRbfKernel": {
        "model": SVC(kernel="rbf", gamma=0.7, random_state=0),
    },
}

normalize = "all"  # for confusion matrices

# randoms forest
model_name = "RandomForest"
model = models[model_name]["model"]

# Train the model
model.fit(predictors_feature_engineering_transformer.fit_transform(dataset.train_x), dataset.train_y) # apply fit and transform, successively

# Evaluate and display metrics
print(score_evaluation(accuracy_score, predictors_feature_engineering_transformer, model, dataset))
print(score_evaluation(precision_score, predictors_feature_engineering_transformer, model, dataset))
print(score_evaluation(recall_score, predictors_feature_engineering_transformer, model, dataset))
print(score_evaluation(f1_score, predictors_feature_engineering_transformer, model, dataset))

# Evalutate and display CM
cm_results = confusion_matrix_evaluation(predictors_feature_engineering_transformer, model, dataset, normalize=normalize)
confusion_matrix_display(cm_results, model)

# Set the path for persistance
model_subdir  = models_dir / camel_to_snake(model_name)
model_subdir.mkdir(exist_ok=True)

# Persist the four transformers and the model
joblib.dump(dataset_ingestion_transformer, model_subdir / "dataset_ingestion_pipeline.pkl")
joblib.dump(remove_horizonless_rows_transformer, model_subdir / "remove_horizonless_rows_pipeline.pkl")
joblib.dump(target_creation_transformer, model_subdir / "target_creation_pipeline.pkl")
joblib.dump(predictors_feature_engineering_transformer, model_subdir / "predictors_feature_eng_pipeline.pkl")
joblib.dump(model, model_subdir / "model.pkl")

