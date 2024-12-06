from django.contrib import admin
from django.urls import path, include
from app.controllers import *
def chatController():
    return ChatController().show()
urlpatterns = [
    path("chat/", chatController, name="chat"),
]
print(ChatController().show())
