import os
from os.path import expanduser

BASE_DIR = os.path.join(expanduser("~"), "rdf/ravop")
DATA_FILES_PATH = os.path.join(BASE_DIR, "files")

RAVOP_LOG_FILE = os.environ.get(BASE_DIR, "ravop.log")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_FILES_PATH, exist_ok=True)

RAVSOCK_SERVER_URL = os.environ.get("RAVSOCK_SERVER_URL", "http://0.0.0.0:9999/")

LOCAL_COMPUTE = False
