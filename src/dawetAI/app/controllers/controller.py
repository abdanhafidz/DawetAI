
from django.http import HttpResponse
from abc import ABC, abstractmethod
import app.views as views
class Controller():
    def __init__(self, services, request=False, ):
        self.__response = HttpResponse(services(request))
    def loadViews():
        return views
    def getResponse(self):
        try:
            return self.__response
        except Exception as e:
            print(e)
    @abstractmethod
    def show():
        pass
    @abstractmethod
    def create():
        pass
    @abstractmethod
    def update():
        pass
    @abstractmethod
    def delete():
        pass

            

