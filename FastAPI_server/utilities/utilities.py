import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel


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


def save_current_chunk(fastapi_dev_folder, current_chunk, date):
    if current_chunk.df is None:
        return

    date = date.strftime("%Y-%m-%d")
    current_folder = fastapi_dev_folder / date
    current_folder.mkdir(parents=True, exist_ok=True)

    filename = f"{date}_weather_dataset_raw_production.csv"
    current_chunk.df.to_csv(current_folder / filename, header=True)
    filename = f"{date}_predicted.csv"
    current_chunk.predictions_sr.to_csv(current_folder / filename, header=True)
    filename = f"{date}_ingested_data.csv"
    current_chunk.ingested_df.to_csv(current_folder / filename, header=True)
    filename = f"{date}_grouf_truth.csv"
    #current_chunk.ground_truth_sr.to_csv(current_folder / filename, header=True)


def get_date(current_df):
    return pd.to_datetime(current_df["Timestamp"], utc=True)[0]


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
