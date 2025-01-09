
from django.template import loader
class Views:
    def __init__(self, templates_name=False):
        if(templates_name != False):
            self.renderTemplate = loader.get_template(templates_name).render()
    def getRenderTemplate(self, request):
        return self.renderTemplate