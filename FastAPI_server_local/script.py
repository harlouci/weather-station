from minio import Minio

# Replace these with your actual credentials and URL
MINIO_ENDPOINT_URL = "http://127.0.0.1:31975"
MINIO_ACCESS_KEY = "minio7777"
MINIO_SECRET_KEY = "minio8858"

minio_client = Minio(
    MINIO_ENDPOINT_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to True if using HTTPS
)

# Attempt a simple operation to verify connectivity
buckets = minio_client.list_buckets()
for bucket in buckets:
    print(bucket.name)