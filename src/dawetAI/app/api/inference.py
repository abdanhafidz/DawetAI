from unsloth import FastLanguageModel
import os
import time
model_path = os.path.abspath(os.path.join(os.getcwd(), "app", "trained-model", "lora_model"))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../../trained-model/lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference



def predict(prompt:str, history):
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    output_ids = model.generate(
        inputs=inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,  # Sampling untuk hasil yang lebih variatif, atau gunakan `do_sample=False` untuk greedy decoding
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    for i in range(len(output_text)):
        time.sleep(0.1)
        yield "You typed: " + output_text[:i+1]