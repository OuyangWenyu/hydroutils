"""
Author: Wenyu Ouyang
Date: 2024-08-15 10:08:59
LastEditTime: 2025-02-02 06:27:44
LastEditors: Wenyu Ouyang
Description: some methods for file operations
FilePath: \\hydroutils\\hydroutils\\hydro_file.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from collections import OrderedDict
import fnmatch
import io
import os
import platform
import json
import logging
from pathlib import Path
import pickle
import re
import tempfile
import zipfile
from urllib import parse
import urllib
import numpy as np
import requests
import async_retriever as ar


def zip_extract(the_dir) -> None:
    """Extract the downloaded zip files in the specified directory.

    Args:
        the_dir (Path): The directory containing zip files to extract.

    Returns:
        None
    """
    for f in the_dir.glob("*.zip"):
        with zipfile.ZipFile(f) as zf:
            # extract files to a directory named by f.stem
            zf.extractall(the_dir.joinpath(f.stem))


def unzip_file(data_zip, path_unzip):
    """Extract a zip file to the specified directory.

    Args:
        data_zip (str): Path to the zip file to extract.
        path_unzip (str): Directory where the contents will be extracted.

    Returns:
        None
    """
    with zipfile.ZipFile(data_zip, "r") as zip_temp:
        zip_temp.extractall(path_unzip)


def unzip_nested_zip(dataset_zip, path_unzip):
    """
    Extract a zip file including any nested zip files
    If a file's name is "xxx_", it seems the "extractall" function in the "zipfile" lib will throw an OSError,
    so please check the unzipped files manually when this occurs.
    Parameters
    ----------
    dataset_zip: the zip file
    path_unzip: where it is unzipped
    """

    with zipfile.ZipFile(dataset_zip, "r") as zfile:
        try:
            zfile.extractall(path=path_unzip)
        except OSError as e:
            logging.warning(
                "Please check the unzipped files manually. There may be some missed important files."
            )
            logging.warning(f"The directory is: {path_unzip}")
            logging.warning(f"Error message: {e}")
    for root, dirs, files in os.walk(path_unzip):
        for filename in files:
            if re.search(r"\.zip$", filename):
                file_spec = os.path.join(root, filename)
                new_dir = os.path.join(root, filename[:-4])
                unzip_nested_zip(file_spec, new_dir)


def zip_file_name_from_url(data_url, data_dir):
    data_url_str = data_url.split("/")
    filename = parse.unquote(data_url_str[-1])
    zipfile_path = os.path.join(data_dir, filename)
    unzip_dir = os.path.join(data_dir, filename[:-4])
    return zipfile_path, unzip_dir


def is_there_file(zipfile_path, unzip_dir):
    """if a file has existed"""
    if os.path.isfile(zipfile_path):
        if os.path.isdir(unzip_dir):
            return True
        unzip_nested_zip(zipfile_path, unzip_dir)
        return True


def download_one_zip(data_url, data_dir):
    """Download one zip file from URL and extract it.

    Args:
        data_url (str): The URL of the file to download.
        data_dir (str): Directory where the file will be downloaded and extracted.

    Returns:
        None

    Note:
        The function will create the target directory if it doesn't exist.
    """

    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.makedirs(unzip_dir)
        r = requests.get(data_url, stream=True)
        with open(zipfile_path, "wb") as py_file:
            for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
                if chunk:
                    py_file.write(chunk)
        unzip_nested_zip(zipfile_path, unzip_dir), download_small_file


def download_zip_files(urls, the_dir: Path):
    """Download multiple files from multiple URLs.

    Args:
        urls (list): List of URLs to download files from.
        the_dir (Path): Directory where downloaded files will be stored.

    Returns:
        None

    Note:
        Uses a temporary directory for caching during download.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_names = tmpdir.joinpath(f"{the_dir.stem}.sqlite")
        r = ar.retrieve(urls, "binary", cache_name=cache_names, ssl=False)
        files = [the_dir.joinpath(url.split("/")[-1]) for url in urls]
        [files[i].write_bytes(io.BytesIO(r[i]).getbuffer()) for i in range(len(files))]


def download_small_zip(data_url, data_dir):
    """Download a small zip file and extract it.

    Args:
        data_url (str): URL of the zip file to download.
        data_dir (str): Directory where the file will be downloaded and extracted.

    Returns:
        None

    Note:
        Uses urllib.request for downloading small files.
    """
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)
        unzip_nested_zip(zipfile_path, unzip_dir)


def download_small_file(data_url, temp_file):
    """Download a small file from URL.

    Args:
        data_url (str): URL of the file to download.
        temp_file (str): Path where the downloaded file will be saved.

    Returns:
        None

    Note:
        Uses requests library for downloading.
    """
    r = requests.get(data_url)
    with open(temp_file, "w") as f:
        f.write(r.text)


def download_excel(data_url, temp_file):
    """Download an Excel file from URL.

    Args:
        data_url (str): URL of the Excel file to download.
        temp_file (str): Path where the Excel file will be saved.

    Returns:
        None

    Note:
        Only downloads if the file doesn't already exist locally.
    """
    if not os.path.isfile(temp_file):
        urllib.request.urlretrieve(data_url, temp_file)


def download_a_file_from_google_drive(drive, dir_id, download_dir):
    """Download files from Google Drive.

    Args:
        drive: Google Drive API instance.
        dir_id (str): ID of the Google Drive directory.
        download_dir (str): Local directory to save downloaded files.

    Returns:
        None

    Note:
        Handles both files and folders recursively.
        Skips already downloaded files.
    """
    file_list = drive.ListFile(
        {"q": f"'{dir_id}' in parents and trashed=false"}
    ).GetList()
    for file in file_list:
        print(f'title: {file["title"]}, id: {file["id"]}')
        file_dl = drive.CreateFile({"id": file["id"]})
        print(f'mimetype is {file_dl["mimeType"]}')
        if file_dl["mimeType"] == "application/vnd.google-apps.folder":
            download_dir_sub = os.path.join(download_dir, file_dl["title"])
            if not os.path.isdir(download_dir_sub):
                os.makedirs(download_dir_sub)
            download_a_file_from_google_drive(drive, file_dl["id"], download_dir_sub)
        else:
            # download
            temp_file = os.path.join(download_dir, file_dl["title"])
            if os.path.isfile(temp_file):
                print("file has been downloaded")
                continue
            file_dl.GetContentFile(os.path.join(download_dir, file_dl["title"]))
            print("Downloading file finished")


def serialize_json(my_dict, my_file, encoding="utf-8", ensure_ascii=True):
    """Serialize a dictionary to a JSON file.

    Args:
        my_dict (dict): Dictionary to serialize.
        my_file (str): Path to the output JSON file.
        encoding (str, optional): File encoding. Defaults to "utf-8".
        ensure_ascii (bool, optional): If True, ensure ASCII output. Defaults to True.

    Returns:
        None
    """
    with open(my_file, "w", encoding=encoding) as FP:
        json.dump(my_dict, FP, ensure_ascii=ensure_ascii, indent=4)


def unserialize_json_ordered(my_file):
    """Load a JSON file into an OrderedDict, preserving key order.

    Args:
        my_file (str): Path to the JSON file to read.

    Returns:
        OrderedDict: Dictionary with preserved key order from the JSON file.
    """
    with open(my_file, "r") as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    return m_dict


def unserialize_json(my_file):
    """Load a JSON file into a Python object.

    Args:
        my_file (str): Path to the JSON file to read.

    Returns:
        object: Python object (typically dict or list) loaded from the JSON file.
    """
    with open(my_file, "r") as fp:
        my_object = json.load(fp)
    return my_object


class NumpyArrayEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and scalar types.

    This encoder converts NumPy arrays and scalar types to Python native types
    that can be serialized by the standard JSON encoder.
    """

    def default(self, obj):
        """Convert NumPy types to JSON serializable objects.

        Args:
            obj: Object to encode.

        Returns:
            JSON serializable object.
        """
        if isinstance(obj, np.ndarray):
            return self.convert_ndarray(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

    def convert_ndarray(self, array):
        """Convert a NumPy array to a nested list.

        Args:
            array (np.ndarray): NumPy array to convert.

        Returns:
            list or scalar: Python native type equivalent of the array.
        """
        if array.ndim == 0:
            return array.item()
        return [
            (
                self.convert_ndarray(element)
                if isinstance(element, np.ndarray)
                else element
            )
            for element in array
        ]


def serialize_json_np(my_dict, my_file):
    """Serialize a dictionary containing NumPy arrays to a JSON file.

    Args:
        my_dict (dict): Dictionary containing NumPy arrays to serialize.
        my_file (str): Path to the output JSON file.

    Returns:
        None

    Note:
        Uses NumpyArrayEncoder to handle NumPy types.
    """
    with open(my_file, "w") as FP:
        json.dump(my_dict, FP, cls=NumpyArrayEncoder)


def serialize_pickle(my_object, my_file):
    """Serialize an object to a pickle file.

    Args:
        my_object (object): Python object to serialize.
        my_file (str): Path to the output pickle file.

    Returns:
        None
    """
    with open(my_file, "wb") as f:
        pickle.dump(my_object, f)


def unserialize_pickle(my_file):
    """Load an object from a pickle file.

    Args:
        my_file (str): Path to the pickle file to read.

    Returns:
        object: Python object loaded from the pickle file.
    """
    with open(my_file, "rb") as f:
        my_object = pickle.load(f)
    return my_object


def serialize_numpy(my_array, my_file):
    """Save a NumPy array to a binary file.

    Args:
        my_array (np.ndarray): NumPy array to save.
        my_file (str): Path to the output file.

    Returns:
        None
    """
    np.save(my_file, my_array)


def unserialize_numpy(my_file):
    """Load a NumPy array from a binary file.

    Args:
        my_file (str): Path to the NumPy array file.

    Returns:
        np.ndarray: NumPy array loaded from the file.
    """
    return np.load(my_file)


def get_lastest_file_in_a_dir(dir_path):
    """Get the last file in a directory

    Parameters
    ----------
    dir_path : str
        the directory

    Returns
    -------
    str
        the path of the weight file
    """
    pth_files_lst = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if fnmatch.fnmatch(file, "*.pth")
    ]
    return get_latest_file_in_a_lst(pth_files_lst)


def get_latest_file_in_a_lst(lst):
    """Get the most recently modified file from a list of files.

    Args:
        lst (list): List of file paths.

    Returns:
        str: Path of the most recently modified file.
    """
    lst_ctime = [os.path.getctime(file) for file in lst]
    sort_idx = np.argsort(lst_ctime)
    return lst[sort_idx[-1]]


def get_cache_dir(app_name="hydro"):
    """Get the appropriate cache directory for the current operating system.

    Args:
        app_name (str, optional): Name of the application. Defaults to "hydro".

    Returns:
        str: Path to the cache directory.

    Note:
        Creates the directory if it doesn't exist.
        Follows OS-specific conventions:
        - Windows: %LOCALAPPDATA%/app_name/Cache
        - macOS: ~/Library/Caches/app_name
        - Linux: ~/.cache/app_name
    """
    home = os.path.expanduser("~")
    system = platform.system()

    if system == "Windows":
        cache_dir = os.path.join(home, "AppData", "Local", app_name, "Cache")
    elif system == "Darwin":
        cache_dir = os.path.join(home, "Library", "Caches", app_name)
    else:
        cache_dir = os.path.join(home, ".cache", app_name)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir
