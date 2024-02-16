import os
from datetime import timedelta

from dotenv import load_dotenv
from timeloop import Timeloop
from utilities.utilities import (
    get_json_to_send_from,
    load_simulated_data,
    log_reponse,
    post_data,
)

load_dotenv()

# Load environment variables
time_loop = int(os.getenv("TIME_LOOP"))
simulation_raw_data_minio_file_path = os.getenv("SIMULATION_RAW_DATA_MINIO_FILE_PATH")
max_number_of_rows = int(os.getenv("MAX_NUMBER_OF_ROWS"))
api_url = os.getenv("FAST_API_URL")

# Read limited row number of simulation raw data
df = load_simulated_data(simulation_raw_data_minio_file_path, max_number_of_rows)

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
