import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from weather.data.minio_utilities import (
    delete_files_in_minio,
    find_files_in_minio,
    write_dataframe_to_minio,
    write_file_to_minio,
)

load_dotenv()

MAP_CSV_TO_BUCKETS = {
    "weather_dataset_raw_production.csv": os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_BUCKET"),
    "weather_dataset_raw_development.csv": os.getenv("START_DEV_RAW_DATA_MINIO_FILE_BUCKET"),
}

BUCKETS_TO_EMPTY = [
    os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_BUCKET"),
    os.getenv("START_DEV_RAW_DATA_MINIO_FILE_BUCKET"),
    os.getenv("START_PROD_RAW_DATA_MINIO_FILE_BUCKET"),
]

MAP_FILES_TO_BUCKETS = {
    "phones.txt": os.getenv("USER_MINIO_FILE_BUCKET"),
    "dataset_ingestion_pipeline.pkl": os.getenv("START_MODEL_MINIO_FILE_BUCKET"),
    "model.pkl": os.getenv("START_MODEL_MINIO_FILE_BUCKET"),
    "predictors_feature_eng_pipeline.pkl": os.getenv("START_MODEL_MINIO_FILE_BUCKET"),
}


def populate_buckets_with_csv(file_paths: Path, map_csv_files_to_buckets):
    """Save csv files by using project standard for `index` and `header` parameters."""
    for file_path in file_paths:
        if (filename := file_path.name) in map_csv_files_to_buckets:
            df = pd.read_csv(file_path)
            bucket = map_csv_files_to_buckets[filename]
            write_dataframe_to_minio(df, bucket, filename)


def populate_buckets_with_files(file_paths: Path, map_files_to_buckets):
    for file_path in file_paths:
        if (filename := file_path.name) in map_files_to_buckets:
            bucket = map_files_to_buckets[filename]
            write_file_to_minio(bucket, file_path)


def empty_buckets(buckets_to_empty):
    for bucket in buckets_to_empty:
        files = find_files_in_minio(bucket, "csv")
        if files:
            delete_files_in_minio(bucket, files)


if __name__ == "__main__":
    data_folder = Path("data")
    file_paths = [file_path for file_path in data_folder.glob("*") if file_path.is_file()]
    empty_buckets(BUCKETS_TO_EMPTY)
    populate_buckets_with_csv(file_paths, MAP_CSV_TO_BUCKETS)
    populate_buckets_with_files(file_paths, MAP_FILES_TO_BUCKETS)
