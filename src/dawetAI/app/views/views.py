
from django.template import loader
class Views:
    def __init__(self, templates_name=False):
        if(templates_name):
            self.renderTemplate = loader.get_template(templates_name).render()
    def getRenderTemplate(self):
        return self.renderTemplate