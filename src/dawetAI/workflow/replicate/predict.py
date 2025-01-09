
from cog import BasePredictor, Input, ConcatenateIterator, Path
from huggingface_hub import snapshot_download
from threading import Thread
import numpy as np
import torch
import json
import os
from unsloth import FastLanguageModel
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
SYSTEM_PROMPT = "Kamu adalah DawetAI sebuah model yang merupakan hasil dari fine-tuning Large Language Models (LLM) Llama Llama-3.2-1B-Instruct-bnb-4bit, dan kamu akan menyediakan layanan percakapan yang berdasarkan datasets Cendol dalam berbagai bahasa di Asia terutama Indonesia"
class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
    def predict(
        self,
        prompt: str = Input(description="Instruction for model"),
        system_prompt: str = Input(
            description="System prompt for the model, helps guides model behaviour.",
            default=SYSTEM_PROMPT,
        ),
        prompt_template: str = Input(
            description="Template to pass to model. Override if you are providing multi-turn instructions.",
            default=PROMPT_TEMPLATE,
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens to generate.", default=512
        ),
        top_p: float = Input(description="Top P", default=0.95),
        top_k: int = Input(description="Top K", default=10),
        min_p: float = Input(description="Min P", default=0),
        typical_p: float = Input(description="Typical P", default=1.0),
        tfs: float = Input(description="Tail-Free Sampling", default=1.0),
        frequency_penalty: float = Input(
            description="Frequency penalty", ge=0.0, le=2.0, default=0.0
        ),
        presence_penalty: float = Input(
            description="Presence penalty", ge=0.0, le=2.0, default=0.0
        ),
        repeat_penalty: float = Input(
            description="Repetition penalty", ge=0.0, le=2.0, default=1.1
        ),
        temperature: float = Input(description="Temperature", default=0.8),
        seed: int = Input(description="Seed", default=None),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        full_prompt = prompt_template.replace("{prompt}", prompt).replace(
            "{system_prompt}", system_prompt
        )

        if seed:
                set_seed(seed)
                print(f"Retrieved seed: {seed}")

        input_ids = self.tokenizer(
                full_prompt,
                add_generation_prompt = True, # kurang yakin apakah wajib, ini unsloth-specific
                return_tensors="pt").to(
                self.model.device
            )


        streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    input_ids,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    typical_p=typical_p,
                    tfs=tfs,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repeat_penalty,
                    temperature=temperature,
                    streamer=streamer,
                ),
            )
        thread.start()

        for new_token in streamer:
            if "<|im_end|>" in new_token:
                break
            yield new_token
        thread.join()      