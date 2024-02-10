import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from timeloop import Timeloop

from util import read_simulated_data, get_json_to_sent, post_data, log_reponse

# Load environment variables from .env file
load_dotenv("data.env")

TIME_LOOP = int(os.getenv("TIME_LOOP"))

data_file_name = os.getenv("DATA_FILE_NAME")
file_path = Path(__file__).resolve().parent.parent / "data" / data_file_name
# URL of your FastAPI API
api_url = os.getenv("API_URL")

df = read_simulated_data(file_path, nb_row=200)

tl = Timeloop()


@tl.job(interval=timedelta(seconds=TIME_LOOP))
def send_new_data():

    json = get_json_to_sent(df)
    if json is None:
        return

    response = post_data(api_url, json)
    log_reponse(response)



# Start the Timeloop
if __name__ == "__main__":
    tl.start(block=True)
