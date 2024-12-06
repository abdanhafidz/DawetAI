from unsloth import FastLanguageModel
from config import config
class Predict:
    def __init__(self, inputs):
        self.__messages = [{"role": "user", "content": inputs},]
        
    def getResult(inputs):
        
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = config.MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
        )

        self.__inputs = tokenizer.apply_chat_template(
            self.__messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        
