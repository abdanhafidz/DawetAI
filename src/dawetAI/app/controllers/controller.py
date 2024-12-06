
from django.http import HttpResponse
from abc import ABC, abstractmethod
import app.services as services
import app.views as views
class Controller():
    def __init__(self, services = False, request=False, ):
        if(services):
            self.response = services(request)
        else:
            self.response = "Hello World"
    def services(req):
        return services
    def loadViews(request):
        return views
    def getResponse(self):
        try:
            return HttpResponse(self.response)
        except Exception as e:
            print(e)

            

