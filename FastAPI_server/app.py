import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from twilio.rest import Client

from data import Item
from util import get_phones, json_to_item_df, get_date, save_current_data, predict_df, update_current_data, send_messages

# Load environment variables from .env file
load_dotenv("api.env")

app = FastAPI()

model_folder = Path(__file__).resolve().parent.parent / "models"
data_folder = Path(__file__).resolve().parent.parent / "data"
fastapi_dev_folder = Path(__file__).resolve().parent.parent / "fastapi_volume" / "dev"

model = joblib.load(model_folder / "model.pkl")
predictors_feature_eng_transformer = joblib.load(model_folder / "predictors_feature_eng_pipeline.pkl")
data_ingestion_pipeline = joblib.load(model_folder / "dataset_ingestion_pipeline.pkl")

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone = os.getenv("TWILIO_PHONE")
twilio_client = Client(account_sid, auth_token)

SEND_MESSAGE = os.getenv("SEND_MESSAGE")



phones_folder = Path('.')
phones =get_phones(phones_folder)

current_df = None
current_predictions_sr = None
current_ground_truth_sr = None
current_predictor = None
previous_day = None
previous_date = None

previous_item_df = pd.read_csv(data_folder / "weather_dataset_raw_development.csv")[-1:]





# Create a POST endpoint to receive JSON data and return a response
@app.post("/predict/")
async def predict(item: Item):
    global current_df, current_predictions_sr, current_ground_truth_sr, previous_item_df, previous_day, current_predictor, previous_date

    new_item_df = json_to_item_df(item.dict())
    new_date = get_date(new_item_df)
    new_day = new_date.day

    if previous_day is not None and new_day != previous_day:
        save_current_data(fastapi_dev_folder, current_df, current_predictions_sr, current_predictor, current_ground_truth_sr, previous_date)
        current_df, current_predictions_sr, current_ground_truth_sr, current_predictor = None, None, None, None



    y, predictor = predict_df(model, data_ingestion_pipeline, predictors_feature_eng_transformer, previous_item_df, new_item_df)


    current_predictions_sr = update_current_data(current_predictions_sr, pd.Series([y]))
    current_df = update_current_data(current_df, new_item_df)
    current_predictor = update_current_data(current_predictor, predictor)

    previous_item_df, previous_day, previous_date = new_item_df, new_day, new_date

    if y==1:
        if SEND_MESSAGE:
            send_messages(phones, twilio_client, twilio_phone)
            print("It will rain!")
        return {"prediction": "rain"}
    else:
        return {"prediction": "no rain"}





