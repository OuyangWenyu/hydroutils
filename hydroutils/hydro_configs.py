import os
from pathlib import Path
import boto3
import s3fs
import yaml
from minio import Minio
import psycopg2


def read_setting(setting_path):
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r", encoding="utf-8") as file:  # 指定编码为 UTF-8
        setting = yaml.safe_load(file)

    example_setting = (
        "minio:\n"
        "  server_url: 'http://minio.waterism.com:9090' # Update with your URL\n"
        "  client_endpoint: 'http://minio.waterism.com:9000' # Update with your URL\n"
        "  access_key: 'your minio access key'\n"
        "  secret: 'your minio secret'\n\n"
        "local_data_path:\n"
        "  root: 'D:\\data\\waterism' # Update with your root data directory\n"
        "  datasets-origin: 'D:\\data\\waterism\\datasets-origin'\n"
        "  datasets-interim: 'D:\\data\\waterism\\datasets-interim'\n"
        "postgres:\n"
        "  server_url: your_postgres_server_url\n"
        "  port: 5432\n"
        "  username: your_postgres_username\n"
        "  password: your_postgres_secret_code\n"
        "  database: your_postgres_database\n"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\nExample configuration:\n{example_setting}"
        )

    # Define the expected structure
    expected_structure = {
        "minio": ["server_url", "client_endpoint", "access_key", "secret"],
        "local_data_path": ["root", "datasets-origin", "datasets-interim"],
        "postgres": ["server_url", "port", "username", "password", "database"],
    }

    # Validate the structure
    try:
        for key, subkeys in expected_structure.items():
            if key not in setting:
                raise KeyError(f"Missing required key in config: {key}")

            if isinstance(subkeys, list):
                for subkey in subkeys:
                    if subkey not in setting[key]:
                        raise KeyError(f"Missing required subkey '{subkey}' in '{key}'")
    except KeyError as e:
        raise ValueError(
            f"Incorrect configuration format: {e}\n\nExample configuration:\n{example_setting}"
        ) from e

    return setting


SETTING_FILE = os.path.join(Path.home(), "hydro_setting.yml")
try:
    SETTING = read_setting(SETTING_FILE)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")

LOCAL_DATA_PATH = SETTING["local_data_path"]["root"]

MINIO_PARAM = {
    "endpoint_url": SETTING["minio"]["client_endpoint"],
    "key": SETTING["minio"]["access_key"],
    "secret": SETTING["minio"]["secret"],
}

FS = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": MINIO_PARAM["endpoint_url"]},
    key=MINIO_PARAM["key"],
    secret=MINIO_PARAM["secret"],
    use_ssl=False,
)

# remote_options parameters for xr open_dataset from minio
RO = {
    "client_kwargs": {"endpoint_url": MINIO_PARAM["endpoint_url"]},
    "key": MINIO_PARAM["key"],
    "secret": MINIO_PARAM["secret"],
    "use_ssl": False,
}


# Set up MinIO client
S3 = boto3.client(
    "s3",
    endpoint_url=SETTING["minio"]["client_endpoint"],
    aws_access_key_id=MINIO_PARAM["key"],
    aws_secret_access_key=MINIO_PARAM["secret"],
)
MC = Minio(
    SETTING["minio"]["client_endpoint"].replace("http://", ""),
    access_key=MINIO_PARAM["key"],
    secret_key=MINIO_PARAM["secret"],
    secure=False,  # True if using HTTPS
)
STATION_BUCKET = "stations"
STATION_OBJECT = "sites.csv"

GRID_INTERIM_BUCKET = "grids-interim"

PS = psycopg2.connect(
    database=SETTING["postgres"]["database"],
    user=SETTING["postgres"]["username"],
    password=SETTING["postgres"]["password"],
    host=SETTING["postgres"]["server_url"],
    port=SETTING["postgres"]["port"],
)
