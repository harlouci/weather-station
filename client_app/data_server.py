import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from timeloop import Timeloop

from data import Item

# Load environment variables from .env file
load_dotenv("data.env")

TIME_LOOP = int(os.getenv("TIME_LOOP"))

data_file_name = os.getenv("DATA_FILE_NAME")
file_path = Path(__file__).resolve().parent.parent / "data" / data_file_name
# URL of your FastAPI API
api_url = os.getenv("API_URL")

def pop_first_row(df:pd.DataFrame):
    if not df.empty:
        first_row = df.iloc[0].copy()
        df.drop(0, inplace=True)
        df.reset_index(inplace=True, drop=True)
        return first_row
    else:
        # Handle the case when the DataFrame is empty, e.g., return None or raise an exception.
        return None



# Define the Pydantic model for the JSON data
df= pd.read_csv(file_path)

df =df.head(200)
df.reset_index(inplace=True, drop=True)

tl = Timeloop()




@tl.job(interval=timedelta(seconds=TIME_LOOP))
def send_new_data():
    row = pop_first_row(df)
    if row is None:
        print("no data to send anymore!")
        return
    row = row.fillna("")
    item_data = Item(**row.to_dict())

    # Send a POST request to the API
    try:
        response = requests.post(api_url, json=item_data.dict())
    except requests.exceptions.RequestException as e:
        # Handle connection errors or exceptions
        print("Error connecting to the API:", e)
        response = None

    if response and response.status_code == 200:
        # API call was successful, process the response
        data = response.json()
        print("API Response:", data)
    elif response is not None:
        # Handle other HTTP status codes (e.g., 404, 500, etc.) as needed
        print(f"API Request Failed. Status code: {response.status_code}")


# Start the Timeloop
if __name__ == "__main__":
    tl.start(block=True)
