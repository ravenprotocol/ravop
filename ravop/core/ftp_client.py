from ftplib import FTP

from ..config import RAVENVERSE_FTP_HOST
from ..globals import globals as g


class FTPClient:
    def __init__(self, host, user, passwd):
        self.ftp = FTP(host)
        # self.ftp.set_debuglevel(2)
        self.ftp.set_pasv(True)
        self.ftp.login(user, passwd)

    def download(self, filename, path):
        print('Downloading')
        self.ftp.retrbinary('RETR ' + path, open(filename, 'wb').write)
        print("Downloaded")

    def upload(self, filename, path):
        self.ftp.storbinary('STOR ' + path, open(filename, 'rb'), blocksize=g.ftp_upload_blocksize)

    def list_server_files(self):
        self.ftp.retrlines('LIST')

    def close(self):
        self.ftp.quit()


def get_client(username, password):
    print("FTP User credentials:", RAVENVERSE_FTP_HOST, username, password)
    return FTPClient(host=RAVENVERSE_FTP_HOST, user=username, passwd=password)


def check_credentials(username, password):
    try:
        FTPClient(host=RAVENVERSE_FTP_HOST, user=username, passwd=password)
        return True
    except Exception as e:
        print("Error:{}".format(str(e)))
        return False
