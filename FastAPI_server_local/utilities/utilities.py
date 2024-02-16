import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Item(BaseModel):
    S_No: int
    Timestamp: str = ""
    Location: str
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
        self.predictions_sr = pd.concat([self.predictions_sr, new_data.predictions_sr], axis=0)
        self.ingested_df = pd.concat([self.ingested_df, new_data.ingested_df], axis=0)


def json_to_item_df(received_json):
    item = Item(**received_json)
    item_df = pd.DataFrame([item.dict()])
    return item_df


def predict_df(model, data_ingestion_pipeline, predictors_feature_eng_transformer, previous_item_df, new_item_df):
    df = pd.concat([previous_item_df, new_item_df], axis=0)
    result_data = data_ingestion_pipeline.transform(df)
    y = model.predict(predictors_feature_eng_transformer.transform(result_data))
    return y[-1], result_data.head(-1)


def save_current_chunk(prod_bucket: Path, current_chunk, date, minio_client):
    """Save current raw data, current ingestion data and current predictions  in MinIO.

    Each file in a specific subfolder of `prod_bucket`, `raw_data`, `predictions` and
    `ingested_data` respectively."""
    if current_chunk.df is None:
        return

    date = date.strftime("%Y-%m-%d")

    filename = f"{date}_weather_dataset_raw_production.csv"
    # Path.mkdir(prod_bucket / "raw_data", exist_ok=True)
    # filepath = prod_bucket / "raw_data" / filename
    # filepath = prod_bucket / filename
    # with fsspec.open(filepath, mode="wb") as f:
    #     current_chunk.df.to_csv(f, header=True)
    with tempfile.TemporaryDirectory() as temp_d:
        local_filepath = os.path.join(temp_d, filename)
        current_chunk.df.to_csv(local_filepath, header=True)
        minio_client.fput_object("prod", filename, local_filepath)

    # filename = f"{date}_predictions.csv"
    # #Path.mkdir(prod_bucket / "predictions_data", exist_ok=True)
    # filepath = prod_bucket / "predictions_data" / filename
    # #filepath = prod_bucket / filename
    # with fsspec.open(filepath, mode="wb") as f:
    #     current_chunk.predictions_sr.to_csv(f, header=True)

    # filename = f"{date}_ingested_data.csv"
    # #Path.mkdir(prod_bucket / "ingested_data", exist_ok=True)
    # filepath = prod_bucket / "ingested_data" / filename
    # #filepath = prod_bucket / filename
    # with fsspec.open(filepath, mode="wb") as f:
    #     current_chunk.ingested_df.to_csv(f, header=True)


def get_date(current_df):
    return pd.to_datetime(current_df["Timestamp"], utc=True)[0]


def get_phones(path: Path):
    file_path = path / "phones.txt"

    # Open the file and read all lines
    with open(file_path) as file:
        phones = file.readlines()

    return phones


def send_messages(phones, twilio_client, twilio_phone):
    for phone in phones:
        _ = twilio_client.messages.create(from_=twilio_phone, body="Dear docker, it will rain in 4 hours.", to=phone)
