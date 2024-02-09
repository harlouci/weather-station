import os

from weather.transformers.skl_transformer_makers import (
    FeatureNames,
    TargetChoice,
)
    
from sklearn.metrics import f1_score

# TODO: DRY
oldnames_newnames_dict = {
    "Temperature_C": "Temperature", 
    "Apparent_Temperature_C": "Apparent_temperature",
    "Wind_speed_kmph": "Wind_speed",
    "Wind_bearing_degrees": "Wind_bearing",
    "Visibility_km": "Visibility",
    "Pressure_millibars": "Pressure",
    "Weather_conditions": "Weather",
}
feature_names = FeatureNames(
    numerical=[
        "Temperature",
        "Humidity",
        "Wind_speed",
        "Wind_bearing",
        "Visibility",
        "Pressure",
    ],
    categorical=[],
)
target_choice = TargetChoice("Weather", 4)

metric = f1_score

# NOTE(Participant): This will eventually change
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") # mlflow
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio7777")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio8858")
MINIO_API_HOST = os.getenv("MINIO_API_HOST", "localhost:31975")       # MinIO
SERVER_API_URL = os.getenv("SERVER_API_URL", "http://localhost:6000") # fastAPI