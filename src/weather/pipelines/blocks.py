import os

from prefect.filesystems import RemoteFileSystem

# TODO(Everyone): Eventually handle credentials correctly
# N.B.: We could simply do os.getenv and launch the
#       prefect agent with the proper environment variable.
#       However, we'll later see the concept of Deployments
#       and infrastructure which will affect this setup.
#       For simplicity of the Automated ML pipelines lab, this
#       is set.
dvc_block = RemoteFileSystem(
    #basepath="s3://bank-marketing/dvc-data",
    basepath="s3://weather-prediction/dvc-data",
    settings={
        "key": "minio7777",
        "secret": "minio8858",
        # TODO(Participant): Make sure this is the proper value (change to yours, or set env)
        "client_kwargs": {"endpoint_url": os.getenv("MINIO_ENDPOINT","http://192.168.39.139:31975")},
        # "key": os.environ['MINIO_ACCESS_KEY'],
        # "secret": os.environ['MINIO_SECRET_KEY'],
        # "client_kwargs": {"endpoint_url": os.getenv('MINIO_ENDPOINT', "http://minio.minio.svc.cluster.local:9000")},
    },
)
