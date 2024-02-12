import fsspec
import os
import fsspec.implementations.local
import logging
import pandas as pd
import time
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Item(BaseModel):
    S_No:int
    Timestamp:str= ""
    Location:str
    Temperature_C: float = None
    Apparent_Temperature_C: float = None
    Humidity: float = None
    Wind_speed_kmph: float = None
    Wind_bearing_degrees: float = None
    Visibility_km: float = None
    Pressure_millibars: float = None
    Weather_conditions: str = ""


@dataclass
class DataChunk:

    df: pd.DataFrame = None
    predictions_sr: pd.Series = None
    ingested_df: pd.DataFrame = None

    def update(self, new_data):
        """Works on dataframes and series."""
        self.df = pd.concat([self.df, new_data.df], axis=0)
        self.predictions_sr =  pd.concat([self.predictions_sr, new_data.predictions_sr], axis=0)
        self.ingested_df =  pd.concat([self.ingested_df, new_data.ingested_df], axis=0)

def json_to_item_df(received_json):
    item = Item(**received_json)
    item_df = pd.DataFrame([item.dict()])
    return item_df

def predict_df(model, data_ingestion_pipeline, predictors_feature_eng_transformer, previous_item_df, new_item_df):
    df =  pd.concat([previous_item_df, new_item_df], axis=0)
    result_data = data_ingestion_pipeline.transform(df)
    y = model.predict(predictors_feature_eng_transformer.transform(result_data))
    return y[-1], result_data.head(-1)

def save_current_chunk(prod_bucket, current_chunk, date):
    """Save current raw data, current ingestion data and current predictions  in MinIO.
    
    Each file in a specific subfolder of `prod_bucket`, `raw_data`, `predictions` and
    `ingested_data` respectively."""
    if current_chunk.df is None:
        return
    
    logging.debug('Here the basic path 3 !!!!! : '+str(prod_bucket)) # TODO: remove or change message

    date = date.strftime("%Y-%m-%d")

    filename = f"{date}_weather_dataset_raw_production.csv"
    filepath = os.path.join(prod_bucket,  filename)
    logging.info(f"Filepath saving with Minio: {filepath}")
    with fsspec.open(filepath, mode="wb") as f: 
        current_chunk.df.to_csv(f, header=True, index=False)
         
def get_date(current_df):
    return pd.to_datetime(current_df["Timestamp"], utc=True)[0]

def load_data(data_file, max_retries=20, delay_seconds=1):
    """
        Attempt to load data from a file up to a maximum number of retries.

        Parameters:
        - data_file: The path to the data file to load.
        - max_retries: The maximum number of attempts to try loading the data.
        - delay_seconds: The delay between retry attempts in seconds.

        Returns:
        - The loaded DataFrame if successful, None otherwise.
        """
    for attempt in range(max_retries):
        try:
            with fsspec.open(data_file) as f:
                df = pd.read_csv(f)
            logging.info(f"Data loaded successfully on attempt {attempt + 1}.")
            return df
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                logging.error("Maximum retries reached. Failed to load data.")
    return None

def get_phones(path:Path):
    file_path = path / "phones.txt"

    # Open the file and read all lines
    with open(file_path) as file:
        phones = file.readlines()

    return phones

def send_messages(phones, twilio_client, twilio_phone):
    for phone in phones:
        message = twilio_client.messages.create(
            from_=twilio_phone,
            body="Dear docker, it will rain in 4 hours.",
            to=phone
        )
        