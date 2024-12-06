from . import Controller
class PredictController(Controller):
    def __init__(self, request):
        super().__init__(
            self.services().PredictService().Predict, 
            request.POST.get('InputTexts')
        )
        return self.getResponse()