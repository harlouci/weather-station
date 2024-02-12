import logging

import fsspec
import pandas as pd
import requests
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def pop_first_row(df:pd.DataFrame):
    if not df.empty:
        first_row = df.iloc[0].copy()
        df.drop(0, inplace=True)
        df.reset_index(inplace=True, drop=True)
        return first_row
    else:
        # Handle the case when the DataFrame is empty, e.g., return None or raise an exception.
        return None


def load_simulated_data(file_path, max_number_of_rows):
    with fsspec.open(file_path) as f:
        df = pd.read_csv(f, sep=",")
    df =df.head(max_number_of_rows)
    df.reset_index(inplace=True, drop=True)
    return df


def post_data(api_url, json):
    """Send a POST request to FastAPI."""
    try:
        response = requests.post(api_url, json=json)
    except requests.exceptions.RequestException as e:
        # Handle connection errors or exceptions
        logging.error(f"Error connecting to the API: {e}")
        response = None
    return response


def log_reponse(response):
    if response and response.status_code == 200:
        # API call was successful, process the response
        data = response.json()
        logging.info(f"API Response: {data}")
    elif response is not None:
        # Handle other HTTP status codes (e.g., 404, 500, etc.) as needed
        logging.error(f"API Request Failed. Status code: {response.status_code}")


def get_json_to_send_from(df):
    row = pop_first_row(df)
    if row is None:
        print("no data to send anymore!")
        return
    row = row.fillna("")
    item_data = Item(**row.to_dict())
    return item_data.dict()



