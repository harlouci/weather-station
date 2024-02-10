import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from timeloop import Timeloop



import logging
tl = Timeloop()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@tl.job(interval=timedelta(seconds=2))
def send_new_data():
    logging.debug(f"777777777777777777777777")

    # Start the Timeloop
if __name__ == "__main__":
    tl.start(block=True)
