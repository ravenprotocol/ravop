from .utils import Singleton


@Singleton
class Globals(object):
    def __init__(self):
        self._default_graph_id = 1
        self._default_sub_graph_id = 1
        self._eager_mode = False

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


globals = Globals.Instance()
