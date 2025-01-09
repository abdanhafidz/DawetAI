from .Controller import Controller
class ChatController(Controller):
    global ctrl
    def __init__(self):
        super().__init__(self.loadViews().ChatView().getRenderTemplate)
    def show(self, request):
        return self.getResponse()
