from ravop.utils import Singleton


@Singleton
class Globals(object):
    def __init__(self):
        self._default_graph_id = None

    @property
    def graph_id(self):
        return self._default_graph_id

    @property
    def ravop_log_file(self):
        return self._ravop_log_file

    @graph_id.setter
    def graph_id(self, id):
        self._default_graph_id = id

    @graph_id.deleter
    def graph_id(self):
        del self._default_graph_id


globals = Globals.Instance()

