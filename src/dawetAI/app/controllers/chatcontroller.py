from .controller import Controller
class ChatController(Controller):
    def __init__(self):
       super().__init__(self.loadViews().ChatView().getRenderTemplate().render)
    
    def show(self):
        return self.getResponse()
