"""
Author: Wenyu Ouyang
Date: 2023-10-25 15:16:21
LastEditTime: 2023-10-27 18:02:46
LastEditors: Wenyu Ouyang
Description: Tests for preprocess
FilePath: /hydroutils/tests/test_s3.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
from io import StringIO
import os.path
import pathlib
import pytest
from minio import Minio
import boto3
import numpy as np
import pandas as pd
import tempfile
from hydroutils.hydro_s3 import boto3_download_file, boto3_upload_file, minio_upload_file


@pytest.fixture()
def minio_paras():
    minio_param = {
        "endpoint_url": "",
        "access_key": "",
        "secret_key": "",
        "bucket_name": "test-private-data",
    }
    home_path = str(pathlib.Path.home())
    if not os.path.exists(os.path.join(home_path, ".wisminio")):
        raise FileNotFoundError(
            "Please create a file called .wisminio in your home directory"
        )
    for line in open(os.path.join(home_path, ".wisminio")):
        key = line.split("=")[0].strip()
        value = line.split("=")[1].strip()
        if key == "endpoint_url":
            minio_param["endpoint_url"] = value
        if key == "access_key":
            minio_param["access_key"] = value
        elif key == "secret_key":
            minio_param["secret_key"] = value
    return minio_param


@pytest.fixture()
def mc(minio_paras):
    minio_server = minio_paras["endpoint_url"]
    return Minio(
        minio_server.replace("http://", ""),
        access_key=minio_paras["access_key"],
        secret_key=minio_paras["secret_key"],
        secure=False,
    )


@pytest.fixture()
def s3(minio_paras):
    minio_server = minio_paras["endpoint_url"]
    return boto3.client(
        "s3",
        endpoint_url=minio_server,
        aws_access_key_id=minio_paras["access_key"],
        aws_secret_access_key=minio_paras["secret_key"],
    )


def create_random_data(rows=100):
    """return DataFrame"""
    dates = pd.date_range("20230101", periods=rows)
    data = {"dates": dates, "values": np.random.randn(rows)}
    csv_buffer = StringIO()
    pd.DataFrame(data).to_csv(csv_buffer)
    data_ = csv_buffer.getvalue().encode("utf-8")

    # create temp csv file
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "wb") as temp_file:
        temp_file.write(data_)
    return temp_path


def test_upload_csv(minio_paras, mc, s3):
    bucket_name = minio_paras["bucket_name"]
    local_file = create_random_data()
    boto3_upload_file(
        s3,
        bucket_name,
        "test_hydroutils_boto3.csv",
        local_file,
    )
    minio_upload_file(
        mc,
        bucket_name,
        "test_hydroutils_minio.csv",
        local_file,
    )


def test_download_csv_boto3(minio_paras, s3):
    bucket_name = minio_paras["bucket_name"]
    local_file = create_random_data()
    boto3_upload_file(
        s3,
        bucket_name,
        "test_hydroutils_boto3.csv",
        local_file,
    )
    boto3_download_file(s3, bucket_name, "test_hydroutils_boto3.csv", local_file)
