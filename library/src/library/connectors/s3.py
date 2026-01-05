import os
import boto3
from dotenv import load_dotenv
import logging

load_dotenv()


def get_s3_client():
    s3 = boto3.client(
        service_name="s3",
        endpoint_url="https://s3.fr-par.scw.cloud",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )
    return s3


def upload_to_s3(local_path: str, s3_key: str, s3_client = None):
    s3 = s3_client or get_s3_client()
    s3.upload_file(
        Filename=local_path,
        Bucket="sufficiency-library",
        Key=s3_key,
    )
    logging.info(f"{s3_key} saved to s3")
