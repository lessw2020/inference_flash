import torch
import time
import torch.nn.functional as F

# test = F.scaled_dot_product_attention()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

inputs = tokenizer(
    "A step by step recipe to make pizza:",
    return_tensors="pt",
)

start = time.perf_counter()

outputs = model.generate(max_new_tokens=300, **inputs)
stop = time.perf_counter()
total_time = round(stop - start, 5)
print(f"==== Total time inference {total_time} seconds")
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
