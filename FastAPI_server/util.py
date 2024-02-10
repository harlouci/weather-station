import pandas as pd
from data import Item
from pathlib import Path

def update_current_data(current_df, new_item_df):
    """work on df and series"""
    return pd.concat([current_df, new_item_df], axis=0)

def json_to_item_df(received_json):
    item = Item(**received_json)
    item_df = pd.DataFrame([item.dict()])
    return item_df

def predict_df(model, data_ingestion_pipeline, predictors_feature_eng_transformer, previous_item_df, new_item_df):
    df =  pd.concat([previous_item_df, new_item_df], axis=0)
    result_data = data_ingestion_pipeline.transform(df)
    y = model.predict(predictors_feature_eng_transformer.transform(result_data))
    return y[-1], result_data.head(-1)


def save_current_data(fastapi_dev_folder, current_df, current_pred, current_predictor, current_ground_truth_sr, date):
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
