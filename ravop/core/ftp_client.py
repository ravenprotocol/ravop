from ftplib import FTP

from ..config import RAVENVERSE_FTP_HOST
from ..globals import globals as g
from tqdm import tqdm
import os


class FTPClient:
    def __init__(self, host, user, passwd):
        self.ftp = FTP(host)
        # self.ftp.set_debuglevel(2)
        self.ftp.set_pasv(True)
        self.ftp.login(user, passwd)

    def download(self, filename, path):
        self.ftp.retrbinary('RETR ' + path, open(filename, 'wb').write)

    def upload(self, filename, path):
        with open(filename, 'rb') as f:
            filesize = os.path.getsize(filename)
            with tqdm(unit = 'b', unit_scale = True, leave = False, miniters = 1, desc = 'Uploading Op Chunk', total = filesize) as tqdm_instance:
                self.ftp.storbinary('STOR ' + path, f, blocksize=g.ftp_upload_blocksize, callback=lambda sent: tqdm_instance.update(len(sent)))

    def list_server_files(self):
        self.ftp.retrlines('LIST')

    def close(self):
        self.ftp.quit()


def get_client(username, password):
    g.logger.debug("FTP User credentials:{} {} {}".format(RAVENVERSE_FTP_HOST, username, password))
    return FTPClient(host=RAVENVERSE_FTP_HOST, user=username, passwd=password)


def check_credentials(username, password):
    try:
        FTPClient(host=RAVENVERSE_FTP_HOST, user=username, passwd=password)
        return True
    except Exception as e:
        g.logger.debug("Error:{}".format(str(e)))
        return False
