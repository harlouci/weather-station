"""This module includes the functions to make datasets for weather prediction ML applications.

# Examples
# --------
#     >>> from bank_marketing.data.make_datasets import make_bank_marketing_dataframe
#     >>> weather_db_file = r"/path/to/bank_marketing.db"
#     >>> socio_eco_data_file = r"/path/to/socio_economic_indices_data.csv"
#     >>> df = make_bank_marketing_dataframe(weather_db_file, socio_eco_data_file)

"""
import os

import pandas as pd

#from weather.helpers.file_loaders import load_fsspec_locally_temp

# def extract_credit_features(row: pd.Series) -> Tuple[str, str]:
#     """Deduce if the customer has any credit or any default of payment
#         based on two columns: status and default penalites that are present
#         in loans and mortgages data tables.

#     Args:
#     ----
#         row (pd.Series): mortgage/loan entry (row) for one customer

#     Returns:
#     -------
#         Tuple[str, str]: has loan (yes/no/unknown), had default (yes/no/unknown)
#     """
#     loan, default = None, None
#     if row["status"] == "paid":
#         loan = "no"
#     elif row["status"] == "ongoing":
#         loan = "yes"
#     elif row["status"] == "unknown":
#         loan = "unknown"
#     if pd.isna(row["default_penalties"]):
#         default = "unknown"
#     elif row["default_penalties"] == 0:
#         default = "no"
#     elif row["default_penalties"] > 0:
#         default = "yes"
#     return loan, default


# def merge_defaults(row: pd.Series) -> str:
#     """Merge two default columns resulting from the fusion of loans and mortgages dataframes

#     Args:
#     ----
#         row (pd.Series): entry (row) for one customer aggregated data

#     Returns:
#     -------
#         str: has default overall (yes/no/unknown)
#     """
#     if row["default_x"] == "yes" or row["default_y"] == "yes":
#         return "yes"
#     elif row["default_x"] == "unknown" or row["default_y"] == "unknown":
#         return "unknown"
#     elif row["default_x"] == "no" and row["default_y"] == "no":
#         return "no"
#     raise ValueError

# TODO
# def make_bank_marketing_dataframe_backup(weather_db_file: os.PathLike, socio_eco_data_file: os.PathLike) -> pd.DataFrame:
#     """Extract and build data from bank database and socio economical indices data files

#     Args:
#     ----
#         weather_db_file (os.PathLike): Bank database sqlite file path
#         socio_eco_data_file (os.PathLike): Socio-Economical CSV data file path

#     Returns:
#     -------
#         pd.DataFrame: customers dataframe (personal infos + loans +
#                                            mortgages + campaign missions +
#                                            socio economical indices)
#     """
#     socio_eco_df = pd.read_csv(socio_eco_data_file, sep=";")
#     bank_marketing_dl = BankMarketingDAL(weather_db_file)
#     loans_df = bank_marketing_dl.loans.fetch_all(to_dataframe=True)
#     mortgages_df = bank_marketing_dl.mortgages.fetch_all(to_dataframe=True)
#     mortgages_df[["housing", "default"]] = mortgages_df.apply(extract_credit_features, axis=1, result_type="expand")
#     loans_df[["loan", "default"]] = loans_df.apply(extract_credit_features, axis=1, result_type="expand")
#     customers_df = bank_marketing_dl.customers.fetch_all(to_dataframe=True)
#     campaign_missions_df = bank_marketing_dl.campaign_missions.fetch_all_done(to_dataframe=True)
#     dataframe = pd.merge(customers_df, campaign_missions_df, left_on="id", right_on="customer_id")
#     dataframe = pd.merge(dataframe, socio_eco_df, left_on="comm_date", right_on="date")
#     dataframe = pd.merge(dataframe, mortgages_df[["customer_id", "housing", "default"]], on="customer_id")
#     dataframe = pd.merge(dataframe, loans_df[["customer_id", "loan", "default"]], on="customer_id")
#     dataframe["default"] = dataframe.apply(merge_defaults, axis=1)
#     dataframe = dataframe.drop(columns=["default_x", "default_y"])
#     return dataframe

def make_weather_prediction_dataframe(
        weather_db_file: os.PathLike,
        dataset_ingestion_transformer,
        temp_dir: os.PathLike="") -> pd.DataFrame:
    """Extract and build data from the weather station raw data.

    # Args:
    # ----
    #     weather_db_file (os.PathLike): Bank database sqlite file path
    #     socio_eco_data_file (os.PathLike): Socio-Economical CSV data file path

    # Returns:
    # -------
    #     pd.DataFrame: customers dataframe (personal infos + loans +
    #                                        mortgages + campaign missions +
    #                                        socio economical indices)
    """
    # Load dataframe
    # with fsspec.open(socio_eco_data_file) as f:
    #     socio_eco_df = pd.read_csv(f, sep=";")

    # # Detect if weather_db_file is local or not, so we can acquire (download) the file if remote.
    # db_url = weather_db_file
    # _db = fsspec.open(weather_db_file)
    # db_downloaded_file = None
    # if not isinstance(_db.fs, fsspec.implementations.local.LocalFileSystem):
    #     db_downloaded_file = load_fsspec_locally_temp(weather_db_file, binary=True, temp_dir=temp_dir)
    #     db_url = db_downloaded_file.name

    # Fetch raw dataset
    df = pd.read_csv(weather_db_file)

    # Ingest raw dataset
    ingested_data = dataset_ingestion_transformer.transform(df)

    # # Split the dataset
    # dataset = prepare_binary_classification_tabular_data(
    #     transformed_data,
    #     created_target,
    # )

    # if db_downloaded_file is not None:
    #     os.remove(db_downloaded_file.name)

    return ingested_data

# def make_dataframe_splits_bimonthly(df, date_column:str, prefix:str, dirpath: os.PathLike) -> None:
#     """
#     Splits a DataFrame into bimonthly segments based on a specified date column
#     and saves the segments as CSV files in the provided directory.

#     Parameters:
#     - df (pandas.DataFrame): The DataFrame to be split.
#     - date_column (str): The column containing date information for segmentation.
#     - prefix (str): Prefix for the generated CSV filenames.
#     - dirpath (os.PathLike): Path to the directory to save the CSV files.
#     """
#     month_year_series = df[date_column].apply(lambda date:'-'.join(date.split('-')[:2]))
#     month_year_list = list(set(month_year_series.tolist()))
#     sorted_month_year_list = sorted(month_year_list, key=lambda myp: tuple(myp.split('-')))
#     for idx in range(0, len(sorted_month_year_list)-1, 2):
#         myp1 = sorted_month_year_list[idx]
#         myp2 = sorted_month_year_list[idx+1]
#         csv_fpath = f'{dirpath}/{prefix}_{myp2}.csv'
#         if not os.path.exists(csv_fpath):
#             data_at_myp1 = df[df[date_column].apply(lambda date:myp1 in date)]
#             data_at_myp2 = df[df[date_column].apply(lambda date:myp2 in date)]
#             data_to_myp2 = pd.concat([data_at_myp1, data_at_myp2], axis=0)
#             data_to_myp2.to_csv(csv_fpath, index=False)
