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
    """upload a file to minio

    Parameters
    ----------
    client : _type_
        the minio client
    bucket_name : _type_
        the bucket name
    object_name : _type_
        the object name
    file_path : _type_
        the local file path
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
    """_summary_

    Parameters
    ----------
    client : Minio
        _description_
    bucket_name : _type_
        _description_
    object_name : _type_
        _description_
    file_path : str
        absolute file
    version_id : _type_, optional
        _description_, by default None
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
    """upload a file to minio

    Parameters
    ----------
    client : _type_
        the minio client
    bucket_name : _type_
        the bucket name
    object_name : _type_
        the object name
    file_path : _type_
        the local file path
    """
    # Make a bucket
    bucket_names = [dic["Name"] for dic in client.list_buckets()["Buckets"]]
    if bucket_name not in bucket_names:
        client.create_bucket(Bucket=bucket_name)
    # Upload an object
    client.upload_file(file_path, bucket_name, object_name)
    return [dic["Key"] for dic in client.list_objects(Bucket=bucket_name)["Contents"]]


def boto3_download_file(client, bucket_name, object_name, file_path: str):
    client.download_file(bucket_name, object_name, file_path)
