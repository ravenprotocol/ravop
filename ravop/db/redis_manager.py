import redis
from ..config import RDF_REDIS_HOST, RDF_REDIS_PORT, RDF_REDIS_DB, QUEUE_LOW_PRIORITY, \
    QUEUE_HIGH_PRIORITY, QUEUE_COMPUTING
from ..utils import Singleton


@Singleton
class RedisManager(object):
    def __init__(self):
        self.r = redis.Redis(host=RDF_REDIS_HOST, port=RDF_REDIS_PORT, db=RDF_REDIS_DB, decode_responses=True)

    def connect(self):
        return self.r


class RavQueue(object):
    def __init__(self, name):
        self.queue_name = name
        redis_manager = RedisManager.Instance()
        self.r = redis_manager.connect()

    def push(self, value):
        if self.search(value) == -1:
            return self.r.rpush(self.queue_name, value)
        else:
            return -1

    def pop(self):
        return self.r.lpop(self.queue_name)

    def __len__(self):
        return self.r.llen(self.queue_name)

    def remove(self, value):
        self.r.lrem(self.queue_name, count=0, value=value)

    def delete(self):
        return self.r.delete(self.queue_name)

    def get(self, index):
        return self.r.lindex(self.queue_name, index)

    def set(self, index, value):
        return self.r.lset(self.queue_name, index, value)

    def search(self, value):
        if type(value).__name__ != "str":
            value = str(value)
        elements = self.r.lrange(self.queue_name, 0, -1)
        try:
            return elements.index(bytes(value, "utf-8"))
        except ValueError as e:
            return -1


def clear_redis_queues():
    r = RavQueue(QUEUE_HIGH_PRIORITY)
    r.delete()
    r1 = RavQueue(QUEUE_LOW_PRIORITY)
    r1.delete()
    r2 = RavQueue(QUEUE_COMPUTING)
    r2.delete()
