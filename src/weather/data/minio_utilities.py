import os
import fsspec
from weather.pipelines.definitions import (
    MINIO_ACCESS_KEY,
    MINIO_API_HOST,
    MINIO_SECRET_KEY,
)

def extract_most_recent_filename(ds_info):
    last_date = sorted(list(ds_info))[-1][:10] # TODO: make it robust
    filename = f"{last_date}_data.csv"
    return filename

def write_dataframe_to_minio(df, bucket, filename):
    """Save `df` in bucket `bucket` as `filename`."""
    filepath = os.path.join(bucket, filename)
    
    with fsspec.open(filepath, mode="wb") as f:
        df.to_csv(f, header=True)

def delete_files_in_minio(bucket, filenames):
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
    for filename in filenames:
        file_path = f"{bucket}/{filename}"  # Construct the full file path
        fs = fsspec.filesystem('file')  # Infer filesystem from path
        if fs.exists(file_path):  # Check if the file exists before trying to delete
            fs.rm(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File does not exist: {file_path}")