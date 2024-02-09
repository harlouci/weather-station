import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from twilio.rest import Client

from data import Item

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
client = Client(account_sid, auth_token)

SEND_MESSAGE = os.getenv("SEND_MESSAGE")

file_path = "phones.txt"

# Open the file and read all lines
with open(file_path) as file:
    phones = file.readlines()

def update_current_data(current_df, new_item_df):
    """work on df and series"""
    return pd.concat([current_df, new_item_df], axis=0)

def json_to_item_df(received_json):
    item = Item(**received_json)
    item_df = pd.DataFrame([item.dict()])
    return item_df

def predict_df(previous_item_df, new_item_df):
    df =  pd.concat([previous_item_df, new_item_df], axis=0)
    result_data = data_ingestion_pipeline.transform(df)
    y = model.predict(predictors_feature_eng_transformer.transform(result_data))
    return y[-1], result_data.head(-1)


def save_current_data(current_df, current_pred, current_predictor, current_ground_truth_sr, date):
    if current_df is None:
        return
    date = date.strftime("%Y-%m-%d")
    current_folder = fastapi_dev_folder / date
    current_folder.mkdir(parents=True, exist_ok=True)

    filename = f"{date}_weather_dataset_raw_production.csv"
    current_df.to_csv(current_folder / filename, header=True)
    filename = f"{date}_predicted.csv"
    current_pred.to_csv(current_folder / filename, header=True)
    filename = f"{date}_predictors.csv"
    current_predictor.to_csv(current_folder / filename, header=True)
    filename = f"{date}_grouf_truth.csv"
    #current_ground_truth_sr.to_csv(current_folder / filename, header=True)



def get_date(current_df):
    return pd.to_datetime(current_df["Timestamp"], utc=True)[0]



current_df = None
current_predictions_sr = None
current_ground_truth_sr = None
current_predictor = None
previous_item_df = pd.read_csv(data_folder / "weather_dataset_raw_development.csv")[-1:]
previous_day = None




# Create a POST endpoint to receive JSON data and return a response
@app.post("/predict/")
async def predict(item: Item):
    global current_df, current_predictions_sr, current_ground_truth_sr, previous_item_df, previous_day, current_predictor

    new_item_df = json_to_item_df(item.dict())
    new_date = get_date(new_item_df)
    new_day = new_date.day

    if previous_day is not None and new_day != previous_day:
        save_current_data(current_df, current_predictions_sr, current_predictor, current_ground_truth_sr, previous_day)
        current_df, current_predictions_sr, current_ground_truth_sr, current_predictor = None, None, None, None



    y, predictor = predict_df(previous_item_df, new_item_df)


    current_predictions_sr = update_current_data(current_predictions_sr, pd.Series([y]))
    current_df = update_current_data(current_df, new_item_df)
    current_predictor = update_current_data(current_predictor, predictor)
    previous_item_df = new_item_df
    previous_day = new_day

    if y==1:
        if SEND_MESSAGE:
            for phone in phones:
                message = client.messages.create(
                    from_=twilio_phone,
                    body="Dear docker, it will rain in 4 hours.",
                    to=phone
                )
            print("It will rain!")
        return {"prediction": "rain"}
    else:
        return {"prediction": "no rain"}





