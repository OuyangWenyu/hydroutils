from collections import OrderedDict
import fnmatch
import os
import json
import logging
import pickle
import re
import zipfile
from urllib import parse
import urllib
import numpy as np
import requests


def unzip_file(data_zip, path_unzip):
    """extract a zip file"""
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
    """
    download one zip file from url as data_file
    Parameters
    ----------
    data_url: the URL of the downloading website
    data_dir: where we will put the data
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


def download_small_zip(data_url, data_dir):
    """download zip file and unzip"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)
        unzip_nested_zip(zipfile_path, unzip_dir)


def download_small_file(data_url, temp_file):
    """download data from url to the temp_file"""
    r = requests.get(data_url)
    with open(temp_file, "w") as f:
        f.write(r.text)


def download_excel(data_url, temp_file):
    """download a excel file according to url"""
    if not os.path.isfile(temp_file):
        urllib.request.urlretrieve(data_url, temp_file)


def download_a_file_from_google_drive(drive, dir_id, download_dir):
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


def serialize_json(my_dict, my_file):
    with open(my_file, "w") as FP:
        json.dump(my_dict, FP, indent=4)


def unserialize_json_ordered(my_file):
    # m_file = os.path.join(my_file, 'master.json')
    with open(my_file, "r") as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    return m_dict


def unserialize_json(my_file):
    with open(my_file, "r") as fp:
        my_object = json.load(fp)
    return my_object


def serialize_pickle(my_object, my_file):
    with open(my_file, "wb") as f:
        pickle.dump(my_object, f)


def unserialize_pickle(my_file):
    with open(my_file, "rb") as f:
        my_object = pickle.load(f)
    return my_object


def serialize_numpy(my_array, my_file):
    np.save(my_file, my_array)


def unserialize_numpy(my_file):
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
    """get the latest file in a list

    Parameters
    ----------
    lst : list
        list of files

    Returns
    -------
    str
        the latest file
    """
    lst_ctime = [os.path.getctime(file) for file in lst]
    sort_idx = np.argsort(lst_ctime)
    return lst[sort_idx[-1]]
