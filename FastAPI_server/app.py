import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from twilio.rest import Client

from utilities.utilities import (
    Item,
    DataChunk,
    get_phones,
    json_to_item_df,
    get_date,
    save_current_chunk,
    predict_df,
    send_messages,
)

# Load environment variables from .env file
load_dotenv(".env")
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone = os.getenv("TWILIO_PHONE")
twilio_client = Client(account_sid, auth_token)
send_message = os.getenv("SEND_MESSAGE")

app = FastAPI()

# Load model, data_ingestion_transformer, predictors_feature_eng_transformer
model_folder = Path(__file__).resolve().parent.parent / "models"
data_folder = Path(__file__).resolve().parent.parent / "data"
fastapi_dev_folder = Path(__file__).resolve().parent.parent / "fastapi_volume" / "dev"

model = joblib.load(model_folder / "model.pkl")
predictors_feature_eng_transformer = joblib.load(model_folder / "predictors_feature_eng_pipeline.pkl")
data_ingestion_transformer = joblib.load(model_folder / "dataset_ingestion_pipeline.pkl")

# TODO: Jules/docker's phone number
phones_folder = Path('.')
phones = get_phones(phones_folder)

# Initialization 
current_chunk = DataChunk()
previous_day = None
previous_date = None
previous_item_df = pd.read_csv(data_folder / "weather_dataset_raw_development.csv")[-1:]

# Create a POST endpoint to receive JSON data and return a response
@app.post("/predict/")
async def predict(item: Item):
    global current_chunk, previous_item_df, previous_day, previous_date

    new_item_df = json_to_item_df(item.dict())
    new_date = get_date(new_item_df)
    new_day = new_date.day

    if previous_day is not None and new_day != previous_day:
        save_current_chunk(
            fastapi_dev_folder,
            current_chunk,
            previous_date,
        )
        current_chunk = DataChunk()

    y, ingested_df = predict_df(
        model, 
        data_ingestion_transformer, 
        predictors_feature_eng_transformer, 
        previous_item_df, 
        new_item_df)

    current_chunk.update(DataChunk(new_item_df, pd.Series([y]), ingested_df))

    previous_item_df, previous_day, previous_date = new_item_df, new_day, new_date

    if y==1:
        if send_message:
            send_messages(phones, twilio_client, twilio_phone)
            print("It will rain!")
        return {"prediction": "rain"}
    else:
        return {"prediction": "no rain"}





