"""
Author: Wenyu Ouyang
Date: 2023-10-27 15:08:16
LastEditTime: 2023-10-27 15:31:13
LastEditors: Wenyu Ouyang
Description: Some functions to deal with s3 file system
FilePath: /hydroutils/hydroutils/hydro_s3.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
from minio import Minio


def minio_upload_file(client, bucket_name, object_name, file_path):
    """Upload a file to MinIO S3-compatible storage.

    This function uploads a local file to MinIO storage. If the specified bucket
    doesn't exist, it will be created automatically. After upload, it returns a
    list of all objects in the bucket.

    Args:
        client (minio.Minio): Initialized MinIO client instance.
        bucket_name (str): Name of the bucket to upload to.
        object_name (str): Name to give the object in MinIO storage.
        file_path (str): Path to the local file to upload.

    Returns:
        list[str]: List of all object names in the bucket after upload.

    Note:
        - Creates bucket if it doesn't exist
        - Uses fput_object for efficient file upload
        - Lists all objects recursively after upload

    Example:
        >>> client = Minio('play.min.io',
        ...               access_key='access_key',
        ...               secret_key='secret_key')
        >>> objects = minio_upload_file(client,
        ...                            'mybucket',
        ...                            'data/file.csv',
        ...                            '/local/path/file.csv')
        >>> print(objects)
        ['data/file.csv', 'data/other.csv']
    """
    # Make a bucket
    bucket_names = [bucket.name for bucket in client.list_buckets()]
    if bucket_name not in bucket_names:
        client.make_bucket(bucket_name)
    # Upload an object
    client.fput_object(bucket_name, object_name, file_path)
    # List objects
    objects = client.list_objects(bucket_name, recursive=True)
    return [obj.object_name for obj in objects]


def minio_download_file(
    client: Minio, bucket_name, object_name, file_path: str, version_id=None
):
    """Download a file from MinIO S3-compatible storage.

    This function downloads an object from MinIO storage to a local file. It
    supports versioned objects and handles UTF-8 encoded text files. The function
    ensures proper cleanup of resources after download.

    Args:
        client (Minio): Initialized MinIO client instance.
        bucket_name (str): Name of the bucket containing the object.
        object_name (str): Name of the object to download.
        file_path (str): Local path where the file should be saved.
        version_id (str, optional): Version ID for versioned objects.
            Defaults to None.

    Note:
        - Assumes UTF-8 encoding for text files
        - Properly closes and releases connection after download
        - Uses context managers for file handling
        - Handles cleanup in finally block for robustness

    Example:
        >>> client = Minio('play.min.io',
        ...               access_key='access_key',
        ...               secret_key='secret_key')
        >>> minio_download_file(client,
        ...                    'mybucket',
        ...                    'data/file.csv',
        ...                    '/local/path/file.csv')
    """
    try:
        response = client.get_object(bucket_name, object_name, version_id)
        res_csv: str = response.data.decode("utf8")
        with open(file_path, "w+") as fp:
            fp.write(res_csv)
    finally:
        response.close()
        response.release_conn()


def boto3_upload_file(client, bucket_name, object_name, file_path):
    """Upload a file to S3 using boto3.

    This function uploads a local file to S3 storage using the boto3 client.
    If the specified bucket doesn't exist, it will be created automatically.
    After upload, it returns a list of all objects in the bucket.

    Args:
        client (boto3.client): Initialized boto3 S3 client instance.
        bucket_name (str): Name of the bucket to upload to.
        object_name (str): Name to give the object in S3 storage.
        file_path (str): Path to the local file to upload.

    Returns:
        list[str]: List of all object keys in the bucket after upload.

    Note:
        - Creates bucket if it doesn't exist
        - Uses upload_file for efficient file upload
        - Lists all objects in bucket after upload
        - Handles bucket listing and creation using boto3's API

    Example:
        >>> import boto3
        >>> client = boto3.client('s3',
        ...                      endpoint_url='http://localhost:9000',
        ...                      aws_access_key_id='access_key',
        ...                      aws_secret_access_key='secret_key')
        >>> objects = boto3_upload_file(client,
        ...                            'mybucket',
        ...                            'data/file.csv',
        ...                            '/local/path/file.csv')
        >>> print(objects)
        ['data/file.csv', 'data/other.csv']
    """
    # Make a bucket
    bucket_names = [dic["Name"] for dic in client.list_buckets()["Buckets"]]
    if bucket_name not in bucket_names:
        client.create_bucket(Bucket=bucket_name)
    # Upload an object
    client.upload_file(file_path, bucket_name, object_name)
    return [dic["Key"] for dic in client.list_objects(Bucket=bucket_name)["Contents"]]


def boto3_download_file(client, bucket_name, object_name, file_path: str):
    """Download a file from S3 using boto3.

    This function downloads an object from S3 storage to a local file using
    the boto3 client. It provides a simple wrapper around boto3's download_file
    method.

    Args:
        client (boto3.client): Initialized boto3 S3 client instance.
        bucket_name (str): Name of the bucket containing the object.
        object_name (str): Name of the object to download.
        file_path (str): Local path where the file should be saved.

    Example:
        >>> import boto3
        >>> client = boto3.client('s3',
        ...                      endpoint_url='http://localhost:9000',
        ...                      aws_access_key_id='access_key',
        ...                      aws_secret_access_key='secret_key')
        >>> boto3_download_file(client,
        ...                    'mybucket',
        ...                    'data/file.csv',
        ...                    '/local/path/file.csv')
    """
    client.download_file(bucket_name, object_name, file_path)
