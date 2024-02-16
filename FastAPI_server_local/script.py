import os

from dotenv import load_dotenv
from minio import Minio

load_dotenv(".env")

MINIO_ENDPOINT_URL = os.getenv("MINO_ACCESS_KEY")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

minio_client = Minio(
    MINIO_ENDPOINT_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,  # Set to True if using HTTPS
)

# Attempt a simple operation to verify connectivity
buckets = minio_client.list_buckets()
for bucket in buckets:
    print(bucket.name)
