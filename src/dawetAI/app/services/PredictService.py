# app/services/PredictService.py
from abc import abstractmethod
from . import Service
from app.config import Config
import os

class PredictService(Service):
    def __init__(self):
        model_path = os.path.abspath(os.path.join(os.getcwd(), "trained_model", "lora_model"))

        from transformers import TextStreamer
        from unsloth import FastLanguageModel
        import torch
        print("Loading model ... ")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,  # Ganti dengan path model yang sesuai
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)  # Enable faster inference

    def getModel(self):
        return self.model

    def getTokenizer(self):
        return self.tokenizer


# Inisialisasi PredictService sekali di proses utama
predict_service = PredictService()
