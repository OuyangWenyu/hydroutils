# hydro_s3

The `hydro_s3` module provides utilities for interacting with S3-compatible storage services, supporting both MinIO and AWS S3.

## MinIO Functions

### minio_upload_file

```python
def minio_upload_file(client: Minio, bucket_name: str, object_name: str, file_path: str) -> list
```

Uploads a file to MinIO S3-compatible storage.

**Example:**
```python
from minio import Minio
client = Minio('play.min.io', access_key='...', secret_key='...')
objects = minio_upload_file(client, 'mybucket', 'data.csv', './data.csv')
print(f"Bucket contents: {objects}")
```

### minio_download_file

```python
def minio_download_file(client: Minio, bucket_name: str, object_name: str, file_path: str, version_id: str = None) -> None
```

Downloads a file from MinIO S3-compatible storage.

**Example:**
```python
from minio import Minio
client = Minio('play.min.io', access_key='...', secret_key='...')
minio_download_file(client, 'mybucket', 'data.csv', './downloaded.csv')
```

## AWS S3 Functions

### boto3_upload_file

```python
def boto3_upload_file(client, bucket_name: str, object_name: str, file_path: str) -> list
```

Uploads a file to AWS S3 using boto3.

**Example:**
```python
import boto3
client = boto3.client('s3')
objects = boto3_upload_file(client, 'mybucket', 'data.csv', './data.csv')
print(f"Bucket contents: {objects}")
```

### boto3_download_file

```python
def boto3_download_file(client, bucket_name: str, object_name: str, file_path: str) -> None
```

Downloads a file from AWS S3 using boto3.

**Example:**
```python
import boto3
client = boto3.client('s3')
boto3_download_file(client, 'mybucket', 'data.csv', './downloaded.csv')
```

## Common Features

- Automatic bucket creation if not exists
- UTF-8 text file handling
- Version control support (MinIO)
- List bucket contents after upload
- Simple and consistent API for both services

## API Reference

::: hydroutils.hydro_s3