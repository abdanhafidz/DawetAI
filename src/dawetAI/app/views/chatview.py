from .Views import Views
class ChatView(Views):
    def __init__(request):
        super().__init__("chat.html")

