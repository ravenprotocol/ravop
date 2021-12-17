import glob
import json
import os
import shutil

import numpy as np
import requests

from .config import DATA_FILES_PATH, RAVSOCK_SERVER_URL
from .socket_client import SocketClient


def save_data_to_file(data_id, data):
    """
    Method to save data in a pickle file
    """
    file_path = os.path.join(DATA_FILES_PATH, "data_{}.json".format(data_id))

    if os.path.exists(file_path):
        os.remove(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        if isinstance(data, np.ndarray):
            data = data.tolist()
        json.dump(data, f)

    return file_path


def load_data_from_file():
    pass


def delete_data_file(data_id):
    file_path = os.path.join(DATA_FILES_PATH, "data_{}.json".format(data_id))
    if os.path.exists(file_path):
        os.remove(file_path)


class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)


def dump_data(data_id, value):
    """
    Dump ndarray to file
    """
    file_path = os.path.join(DATA_FILES_PATH, "data_{}.pkl".format(data_id))
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    value.dump(file_path)
    return file_path


def copy_data(source, destination):
    try:
        shutil.copy(source, destination)
        print("File copied successfully.")
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")
    # For other errors
    except:
        print("Error occurred while copying file.")


def inform_server():
    socket_client = SocketClient(server_url=RAVSOCK_SERVER_URL).connect()
    socket_client.emit("inform_server", data={"type": "event"}, namespace="/ravop")


def reset():
    for file_path in glob.glob("files/*"):
        if os.path.exists(file_path):
            os.remove(file_path)

    if not os.path.exists("files"):
        os.makedirs("files")

    # Delete and create database
    reset_database()

    # Clear redis queues
    from .db import clear_redis_queues
    clear_redis_queues()


def make_request(endpoint, method, payload={}, headers=None):
    if method == "post":
        return requests.post(
            "{}{}".format(RAVSOCK_SERVER_URL, endpoint), json=payload, headers=headers
        )
    elif method == "get":
        return requests.get(
            "{}{}".format(RAVSOCK_SERVER_URL, endpoint), headers=headers
        )


def convert_to_ndarray(x):
    if isinstance(x, str):
        x = np.array(json.loads(x))
    elif isinstance(x, list) or isinstance(x, tuple) or isinstance(x, int) or isinstance(x, float):
        x = np.array(x)

    return x


def convert_ndarray_to_str(x):
    return str(x.tolist())
