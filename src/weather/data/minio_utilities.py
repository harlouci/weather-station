import os
import re
from minio import Minio
#import fsspec
from weather.pipelines.definitions import (
    MINIO_ACCESS_KEY,
    MINIO_API_HOST,
    MINIO_SECRET_KEY,
)
from dotenv import load_dotenv
load_dotenv()
SCRATCH_DIR = os.environ.get('SCRATCH_DIR')

# Utilities to extract most recent filename if any
def extract_date(string): 
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    return re.search(date_pattern, string)

def extract_last_date_if_any(list_of_strings):
    extracted_dates = [extract_date(name).group(0) for name in list_of_strings if extract_date(name)]
    if extracted_dates:
        return sorted(extracted_dates)[-1]
    return ""

def extract_most_recent_filename_if_any(ds_info: dict, tag: str):
    last_date = extract_last_date_if_any(ds_info.keys())
    if last_date:
        return f"{last_date}_{tag}.csv"
    return ""

# File writer

def write_dataframe_to_minio(df, bucket, filename):
    assert filename != "", "`filename` is an empty string."
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    data_path = os.path.join(SCRATCH_DIR, filename)
    df.to_csv(data_path, header=True)
    found = minio_client.bucket_exists(bucket)
    if not found:
       minio_client.make_bucket(bucket)
    minio_client.fput_object(bucket, filename, data_path)

# Files deleter

def delete_files_in_minio(bucket, filenames):
    assert filenames != [], "`filenames` is an empty list."
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    for filename in filenames:
        try:
            minio_client.remove_object(bucket, filename)
            print(f"File '{filename}' deleted successfully from bucket '{bucket}'.")
        except Exception as e:
            print(f"Error deleting file '{filename}' from bucket '{bucket}': {e}")

#def delete_files_in_minio(bucket, filenames):
    #fs = fsspec.filesystem('s3', anon=False)
    # Set fs
    # minio_endpoint = os.path.join("http://", MINIO_API_HOST)
    # fs = fsspec.filesystem(
    #     's3',
    #     client_kwargs={'endpoint_url': minio_endpoint},
    #     key=MINIO_ACCESS_KEY,
    #     secret=MINIO_SECRET_KEY,
    #     use_ssl=False,  # Set to True if your MinIO server uses SSL/TLS
    # )
    # Loop through filenames and delete each one
    # for filename in filenames:
    #     filepath = os.path.join(bucket, filename)
    #     if fs.exists(filepath):  
    #         fs.rm(filepath)
    #         print(f"Deleted: {filepath}")
    #     else:
    #         print(f"File does not exist: {filepath}")
    # for filename in filenames:
    #     file_path = f"{bucket}/{filename}"  # Construct the full file path
    #     fs = fsspec.filesystem('file')  # Infer filesystem from path
    #     if fs.exists(file_path):  # Check if the file exists before trying to delete
    #         fs.rm(file_path)
    #         print(f"Deleted: {file_path}")
    #     else:
    #         print(f"File does not exist: {file_path}")



# def write_dataframe_to_minio(df, bucket, filename):
#     """Save `df` in bucket `bucket` as `filename`."""
#     filepath = os.path.join(bucket, filename)

#     with fsspec.open(filepath, mode="wb") as f:
#         df.to_csv(f, header=True)