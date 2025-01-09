from django.urls import path , include
from .consumer import PredictConsumer

websocket_urlpatterns = [
    path(r"predict" , PredictConsumer.as_asgi()) , 
]