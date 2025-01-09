from django.contrib import admin
from django.urls import path
from app.controllers import *

urlpatterns = [
    path("admin/", admin.site.urls),
    path("chat/", ChatController().show),
]
