import fsspec
import fsspec.implementations.local
import os
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from twilio.rest import Client

import mlflow


from utilities.utilities import (
    Item,
    DataChunk,
    get_phones,
    json_to_item_df,
    get_date,
    save_current_chunk,
    predict_df,
    send_messages,
    load_data,
    load_model_by_stage,
)


# Load environment variables from .env file
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone = os.getenv("TWILIO_PHONE")
twilio_client = Client(account_sid, auth_token)
production_raw_data_minio_file_path = os.getenv("PRODUCTION_RAW_DATA_MINIO_FILE_PATH")
dev_raw_data_minio_file_path = os.getenv("DEV_RAW_DATA_MINIO_FILE_PATH")
prod_bucket = os.getenv("PROD_BUCKET")
send_message = os.getenv("SEND_MESSAGE")
model_registry_uri = os.getenv("MODEL_REGISTRY_URI")
model_stage = os.getenv("MODEL_STAGE")
model_name = os.getenv("MODEL_NAME")
model_folder = Path(os.getenv("MODEL_DIR")) #mounted folder for now, moving to mlflow soon

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Here the basic path 1 !!!!! : '+str(prod_bucket))
logging.info('Here the basic path 2 !!!!! : '+str(os.getenv("PROD_BUCKET")))

app = FastAPI()    

# Load model, data_ingestion_transformer, predictors_feature_eng_transforme
model = joblib.load(model_folder / "model.pkl")
predictors_feature_eng_transformer = joblib.load(model_folder / "predictors_feature_eng_pipeline.pkl")
data_ingestion_transformer = joblib.load(model_folder / "dataset_ingestion_pipeline.pkl")

# TODO: Jules/docker's phone number, transfer to Minio
phones_folder = Path('.')
phones = get_phones(phones_folder)

# Initialization 
current_chunk = DataChunk()
previous_day = None
previous_date = None
items_data = load_data(dev_raw_data_minio_file_path)
previous_item_df = items_data[-1:]


# Create a POST endpoint to receive JSON data and return a response
@app.post("/predict/")
async def predict(item: Item):
    global current_chunk, previous_item_df, previous_day, previous_date

    new_item_df = json_to_item_df(item.dict())
    new_date = get_date(new_item_df)
    new_day = new_date.day

    if previous_day is not None and new_day != previous_day:
        save_current_chunk(
            prod_bucket,
            current_chunk,
            previous_date
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

@app.get("/reload/")
async def reload():
    global model
    model = load_model_by_stage(model_registry_uri, model_name, model_stage)



