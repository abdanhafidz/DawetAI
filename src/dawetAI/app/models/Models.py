from django_redis import get_redis_connection
class Models:
    def __init__(self):
        self.redis_con = get_redis_connection("default")
    def getRedisDB(self):
        return self.redis_con
    