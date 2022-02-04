import os
from os.path import expanduser

BASE_DIR = os.path.join(expanduser("~"), "rdf")
DATA_FILES_PATH = os.path.join(BASE_DIR, "files")

RAVOP_LOG_FILE = os.environ.get(BASE_DIR, "ravop.log")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_FILES_PATH, exist_ok=True)

RDF_REDIS_HOST = os.environ.get("RDF_REDIS_HOST", "localhost")
RDF_REDIS_PORT = os.environ.get("RDF_REDIS_PORT", "6379")
RDF_REDIS_DB = os.environ.get("RDF_REDIS_DB", "0")

QUEUE_HIGH_PRIORITY = "queue:high_priority"
QUEUE_LOW_PRIORITY = "queue:low_priority"
QUEUE_COMPUTING = "queue:computing"

RAVSOCK_SERVER_URL = os.environ.get("RAVSOCK_SERVER_URL", "http://0.0.0.0:9999/")

RDF_DATABASE_URI = "sqlite:///{}/rdf.db?check_same_thread=False".format(BASE_DIR)
# RDF_DATABASE_URI = "mysql://root:qwerty12345.A@localhost/rdf"
RDF_REDIS_URI = os.environ.get("RDF_REDIS_URI", "redis://localhost:6379?db=0")

LOCAL_COMPUTE = True
