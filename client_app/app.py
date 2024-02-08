from fastapi import FastAPI
from data import Item, json_to_dataframe_col
import pandas as pd
import joblib
import os
from pathlib import Path

from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("api.env")

app = FastAPI()

model_folder = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(model_folder / 'model.pkl')
input_transformer = joblib.load(model_folder / 'predictors_feature_eng_pipeline.pkl')
data_ingestion_pipeline = joblib.load(model_folder / 'dataset_ingestion_pipeline.pkl')

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone = os.getenv('TWILIO_PHONE')
client = Client(account_sid, auth_token)


file_path = 'phones.txt'

# Open the file and read all lines
with open(file_path, 'r') as file:
    phones = file.readlines()


useless_columns = ['S_No', 'Location', 'Apparent_temperature']

# Create a POST endpoint to receive JSON data and return a response
@app.post("/predict/")
async def predict(item: Item):
    df = pd.DataFrame([item.dict()])

    df.rename(columns=json_to_dataframe_col, inplace=True)
    for column_name in useless_columns:
        df[column_name] = ''
    df = data_ingestion_pipeline.transform(df)
    print('allo')
    print(list(df))
    print(input_transformer)
    df = input_transformer.transform(df)
    print(list(df))
    y = model.predict(input_transformer.transform(df))
    if y[0]==1:
        for phone in phones:
            message = client.messages.create(
                from_=twilio_phone,
                body="it's will rain in 4 hours!",
                to=phone
            )
        print("It's will rain!")
        return {"prediction": "rain"}
    else:
        return {"prediction": "no rain"}





