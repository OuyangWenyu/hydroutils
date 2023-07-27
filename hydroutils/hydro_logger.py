import json
import os
import re
import zipfile
import datetime as dt, datetime
from typing import Union
import pickle
import smtplib
import ssl
from collections import OrderedDict
import numpy as np
import urllib
from urllib import parse

import requests
import matplotlib.pyplot as plt
from itertools import combinations

import threading
import functools

import tqdm

import logging


def get_hydro_logger(log_level_param):
    logger = logging.getLogger(__name__)
    # StreamHandler
    stream_handler = logging.StreamHandler()  # console stream output
    stream_handler.setLevel(level=log_level_param)
    logger.addHandler(stream_handler)
    return logger


log_level = logging.INFO
HydroLogger = get_hydro_logger(log_level)


# ------------------------------------------------progress bar----------------------------------------------------
def provide_progress_bar(
    function, estimated_time, tstep=0.2, tqdm_kwargs={}, args=[], kwargs={}
):
    """
    Tqdm wrapper for a long-running function
    Parameters
    -----------
    function
        function to run
    estimated_time
        how long you expect the function to take
    tstep
        time delta (seconds) for progress bar updates
    tqdm_kwargs
        kwargs to construct the progress bar
    args
        args to pass to the function
    kwargs
        keyword args to pass to the function
    Returns
    --------
    function(*args, **kwargs)
    """
    ret = [None]  # Mutable var so the function can store its return value

    def myrunner(function, ret, *args, **kwargs):
        ret[0] = function(*args, **kwargs)

    thread = threading.Thread(
        target=myrunner, args=(function, ret) + tuple(args), kwargs=kwargs
    )
    pbar = tqdm.tqdm(total=estimated_time, **tqdm_kwargs)

    thread.start()
    while thread.is_alive():
        thread.join(timeout=tstep)
        pbar.update(tstep)
    pbar.close()
    return ret[0]


def progress_wrapped(estimated_time, tstep=0.2, tqdm_kwargs={}):
    """Decorate a function to add a progress bar"""

    def real_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return provide_progress_bar(
                function,
                estimated_time=estimated_time,
                tstep=tstep,
                tqdm_kwargs=tqdm_kwargs,
                args=args,
                kwargs=kwargs,
            )

        return wrapper

    return real_decorator


def setup_log(tag="VOC_TOPICS"):
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    return logger


# -------------------------------------------------- notification tools--------------------------------------------
def send_email(subject, text, receiver="hust2014owen@gmail.com"):
    sender = "hydro.wyouyang@gmail.com"
    password = "D4VEFya3UQxGR3z"
    context = ssl.create_default_context()
    msg = f"Subject: {subject}\n\n{text}"
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(from_addr=sender, to_addrs=receiver, msg=msg)
