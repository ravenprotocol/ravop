import os

from ravcom.config import BASE_DIR

RAVOP_LOG_FILE = os.environ.get(BASE_DIR, "ravop.log")
