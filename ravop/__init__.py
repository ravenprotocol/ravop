from .config import RAVOP_LOG_FILE


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


@Singleton
class Globals(object):
    def __init__(self):
        self._default_graph_id = None
        self._ravop_log_file = RAVOP_LOG_FILE

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
