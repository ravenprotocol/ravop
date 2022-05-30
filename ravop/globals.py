from .singleton_utils import Singleton


@Singleton
class Globals(object):
    def __init__(self):
        self._default_graph_id = 1
        self._default_sub_graph_id = 1
        self._eager_mode = False
        self._ftp_upload_blocksize = 8192
        self._ravenverse_token = None

    @property
    def graph_id(self):
        return self._default_graph_id

    @property
    def sub_graph_id(self):
        return self._default_sub_graph_id

    @property
    def eager_mode(self):
        return self._eager_mode

    @eager_mode.setter
    def eager_mode(self, mode):
        self._eager_mode = mode

    @property
    def ravop_log_file(self):
        return self._ravop_log_file

    @graph_id.setter
    def graph_id(self, id):
        self._default_graph_id = id

    @graph_id.deleter
    def graph_id(self):
        del self._default_graph_id

    @sub_graph_id.setter
    def sub_graph_id(self, id):
        self._default_sub_graph_id = id

    @sub_graph_id.deleter
    def sub_graph_id(self):
        del self._default_sub_graph_id

    @property
    def ftp_upload_blocksize(self):
        return self._ftp_upload_blocksize

    @ftp_upload_blocksize.setter
    def ftp_upload_blocksize(self, ftp_upload_blocksize):
        self._ftp_upload_blocksize = ftp_upload_blocksize

    @property
    def ravenverse_token(self):
        return self._ravenverse_token

    @ravenverse_token.setter
    def ravenverse_token(self, ravenverse_token):
        self._ravenverse_token = ravenverse_token


globals = Globals.Instance()
