import os

import pandas as pd
from prefect import flow
from weather.data.make_datasets import make_weather_prediction_dataframe


# NOTE(PARTICIPANT): We could have added the data validation in this flow
#                    and we would not have to repeat ourselves in each of
#                    the other pipelines.
@flow(name="data-extraction")
def data_extraction(weather_db_file: os.PathLike,dataset_ingestion_transformer,
    ) -> pd.DataFrame:
    return make_weather_prediction_dataframe(weather_db_file, dataset_ingestion_transformer)
