from django.db import models
class ChatModel(models.Model):
    chat_id = models.IntegerField()
    prompt = models.CharField(max_length=4096)
    session_id = models.IntegerField()
    response = models.CharField(max_length=4096)
