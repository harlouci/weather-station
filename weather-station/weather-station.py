import os
from datetime import timedelta

from dotenv import load_dotenv
from timeloop import Timeloop

from utilities.utilities import (
    load_simulated_data, 
    get_json_to_send_from, 
    post_data, 
    log_reponse,
)


# Load environment variables
load_dotenv(".env")
time_loop = int(os.getenv("TIME_LOOP"))
production_raw_data_minio_file_path = os.getenv("PRODUCTION_RAW_DATA_MINIO_FILE_PATH")
max_number_of_rows = int(os.getenv("MAX_NUMBER_OF_ROWS"))
api_url = os.getenv("FAST_API_URL")


# Read limited row number of production raw data
df= load_simulated_data(production_raw_data_minio_file_path, max_number_of_rows)


# Timeloop
tl = Timeloop()

@tl.job(interval=timedelta(seconds=time_loop))
def send_new_data():

    json = get_json_to_send_from(df)
    if json is None:
        return

    response = post_data(api_url, json)
    log_reponse(response)


# Start the Timeloop
if __name__ == "__main__":
    tl.start(block=True)
