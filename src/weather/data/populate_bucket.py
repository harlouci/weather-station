import os

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
from weather.data.minio_utilities import write_dataframe_to_minio, find_files_in_minio, delete_files_in_minio, write_file_to_minio
CSV_TO_POPULATE = {'weather_dataset_raw_production.csv':os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_BUCKET"),
                   'weather_dataset_raw_development.csv':os.getenv("START_DEV_RAW_DATA_MINIO_FILE_BUCKET")}

BUCKET_TO_EMPTY =[os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_BUCKET"),
                  os.getenv("START_DEV_RAW_DATA_MINIO_FILE_BUCKET"),
                  os.getenv("START_PROD_RAW_DATA_MINIO_FILE_BUCKET")]

FILE_TO_POPULATE = {'phones.txt':os.getenv("USER_MINIO_FILE_BUCKET"),
                    'dataset_ingestion_pipeline.pkl':os.getenv("START_MODEL_MINIO_FILE_BUCKET"),
                    'model.pkl':os.getenv("START_MODEL_MINIO_FILE_BUCKET"),
                    'predictors_feature_eng_pipeline.pkl':os.getenv("START_MODEL_MINIO_FILE_BUCKET")}

def populate_csv(file_paths:Path):
    for file_path in file_paths:
        if (filename:=file_path.name) in CSV_TO_POPULATE:
            df = pd.read_csv(file_path)
            bucket = CSV_TO_POPULATE[filename]
            write_dataframe_to_minio(df, bucket, filename)

def populate_files(file_paths:Path):
    for file_path in file_paths:
        if (filename:=file_path.name) in FILE_TO_POPULATE:
            bucket = FILE_TO_POPULATE[filename]
            write_file_to_minio(bucket,file_path)

def empty_folder():
    for bucket in BUCKET_TO_EMPTY:
        files = find_files_in_minio(bucket, 'csv')
        if files:
            delete_files_in_minio(bucket, files)




if __name__ == "__main__":
    data_folder = Path('data')
    files_in_folder = [file_path for file_path in data_folder.glob("*") if file_path.is_file()]
    empty_folder()
    populate_csv(files_in_folder)
    populate_files(files_in_folder)



