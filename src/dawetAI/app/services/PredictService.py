from abc import abstractmethod
from . import Service
from app.config import Config
class PredictService(Service):
    def __init__(self, inputTexts = None):
        self.__inputTexts = inputTexts
    def Predict(self, inputTexts):
        if(inputTexts != None):self.__inputTexts = inputTexts
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
                max_seq_length = Config.MAX_SEQ_LENGTH,
                dtype = None,
                load_in_4bit = True,
            )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        messages = [
            {"role": "user", "content": self.__inputTexts},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")

        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        result = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                        temperature = 0.7)
        return result
    

