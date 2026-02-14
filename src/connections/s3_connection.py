import boto3
import pandas as pd
import logging
from src.logger import logging
from io import StringIO

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import boto3
import pandas as pd
from io import StringIO
from src.logger import logging


class S3Operations:
    def __init__(self, bucket_name: str, region_name: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region_name)
        logging.info("S3 connection initialized")

    def fetch_file_from_s3(self, file_key: str) -> pd.DataFrame:
        try:
            logging.info(f"Fetching '{file_key}' from bucket '{self.bucket_name}'")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
            logging.info(f"Successfully loaded {len(df)} records from S3")
            return df
        except Exception as e:
            logging.exception(f"Failed to fetch file from S3: {e}")
            raise

# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual AWS credentials and S3 details
#     BUCKET_NAME = "bucket-name"
#     AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
#     AWS_SECRET_KEY = "AWS_SECRET_KEY"
#     FILE_KEY = "data.csv"  # Path inside S3 bucket

#     data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame