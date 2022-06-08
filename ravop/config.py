import os
from os.path import expanduser

BASE_DIR = os.path.join(expanduser("~"), "ravenverse/ravop")
DATA_FILES_PATH = os.path.join(BASE_DIR, "files")

TEMP_FILES_BASE_PATH = os.path.join(expanduser("~"), os.getcwd())
TEMP_FILES_PATH = os.path.join(TEMP_FILES_BASE_PATH, "temp_files")

RAVOP_LOG_FILE = os.environ.get(BASE_DIR, "ravop.log")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_FILES_PATH, exist_ok=True)
os.makedirs(TEMP_FILES_PATH, exist_ok=True)

RAVENVERSE_URL = os.environ.get("RAVENVERSE_URL")

LOCAL_COMPUTE = False

RAVENVERSE_FTP_HOST = os.environ.get("RAVENVERSE_FTP_HOST")

FTP_BLOCKSIZE = 8192 * 10  # bytes
