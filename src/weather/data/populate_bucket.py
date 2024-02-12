import os

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
from weather.data.minio_utilities import write_dataframe_to_minio
CSV_TO_POPULATE = {'weather_dataset_raw_production.csv':os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_BUCKET")}

BUCKET_TO_EMPTY =[os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_BUCKET")]

def populate_csv(file_paths:Path):
    for file_path in file_paths:
        if (filename:=file_path.name) in CSV_TO_POPULATE:
            df = pd.read_csv(file_path)
            bucket = CSV_TO_POPULATE[filename]
            write_dataframe_to_minio(df, bucket, filename)




if __name__ == "__main__":
    data_folder = Path('data')
    files_in_folder = [file_path for file_path in data_folder.glob("*") if file_path.is_file()]
    populate_csv(files_in_folder)



