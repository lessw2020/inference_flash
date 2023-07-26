import torch
import time
import torch.nn.functional as F

# test = F.scaled_dot_product_attention()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model.to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
# tokenizer.to("cuda:0")

torch.cuda.manual_seed(2020)
torch.manual_seed(2020)
torch.backends.cuda.enable_flash_sdp(enabled=False)
torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
torch.backends.cuda.enable_math_sdp(enabled=False)

inputs = tokenizer(
    # "A step by step recipe to make pizza:",
    # "translate English to German: The weather is beautiful today!",
    "translate English to German: A step by step recipe to make cherry pie",
    return_tensors="pt",
)
inputs.to("cuda:0")

test_count = 3
accum = 0
questions = [
    "A step by step recipe to make pizza",
    "translate English to German: A step by step recipe to make cherry pie",
    "translate English to German: The weather is beautiful today.",
]


def get_inputs(input):
    inputs = tokenizer(input, return_tensors="pt")
    inputs.to("cuda:0")
    return inputs


# warmup
start = time.perf_counter()
outputs = model.generate(max_new_tokens=300, **inputs)
stop = time.perf_counter()
total_time = round(stop - start, 5)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
"""
print(f"==== Warmup time inference {total_time} seconds.")
start = time.perf_counter()
outputs = model.generate(max_new_tokens=300, **inputs)
stop = time.perf_counter()
total_time = round(stop - start, 5)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
print(f"==== Warmup time inference {total_time} seconds.")
start = time.perf_counter()
outputs = model.generate(max_new_tokens=300, **inputs)
stop = time.perf_counter()
total_time = round(stop - start, 5)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
print(f"==== Warmup time inference {total_time} seconds.")
"""
print(f"Starting testing....")
for i in range(test_count):
    start = time.perf_counter()
    inputs = get_inputs(questions[i])
    outputs = model.generate(max_new_tokens=300, **inputs)
    stop = time.perf_counter()
    total_time = round(stop - start, 5)
    accum += total_time
    print(f"Test {i} completed...{total_time} seconds")
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    print(f"=======")
avg_time = round(accum / test_count, 5)
print(f"==== Average time inference {avg_time} seconds")

# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
