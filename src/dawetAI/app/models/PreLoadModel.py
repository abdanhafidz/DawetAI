from . import Models
class PreLoadModel(Models):
    def __init__(self):
        self.redis_con = Models().getRedisDB()
        self.model = self.redis_con.get("model")
        self.tokenizer = self.redis_con.get("tokenizer")
    def setModel(self, model):
        if(model != None):
            self.model = model
            self.redis_con.set("model", self.model)
    def setTokenizer(self, tokenizer):
        if(tokenizer != None):
            self.tokenizer = tokenizer
            self.redis_con.set("tokenizer", self.tokenizer)