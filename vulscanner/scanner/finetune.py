import wandb
wandb.login(key="aed77d0f18d3be8406428bbd77d6878c7cd03718")

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
mistral_api_key = os.getenv("MISTRAL_API_KEY") # Ensure this is set
os.environ["MISTRAL_API_KEY"] = "b6IHYFUES7wYx1umBijvsdK84l2JRFCN"

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = "<your-api-key>"
os.environ['LANGCHAIN_API_KEY'] = "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"


import os
os.environ["LANGCHAIN_PROJECT"] = "Mistral-code-gen-testing"
os.environ["MISTRAL_API_KEY"] = "b6IHYFUES7wYx1umBijvsdK84l2JRFCN"


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    # model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)





import pandas as pd
from datasets import Dataset


alpaca_prompt = """You are a brilliant software security expert.
You will be provided with a python code delimited by triple backticks.
If it contains any CWE security vulnerabilities, write Vulnerable.
If the code does not contain any vulnerabilities, write Not Vulnerable.
If the code has the vulnerability, write a repaired secure version of the
code that preserves its exact functionality.
Format your response as a JSON object with "label" as the key
for vulnerability status, "cwe" as the vulnerability found,
and "fix" for the fixed code snippet.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


# Load your custom dataset from CSV
custom_data = pd.read_csv("/content/datasetnofilesname.csv")

# Convert DataFrame to Dataset
custom_dataset = Dataset.from_pandas(custom_data)

# Define the formatting function
def formatting_prompts_func(examples):
    instructions = "The following code performs the following operation " + str(examples["Prompt"]) + ".\n\n" + "Is the following code Insecure, if yes what type of Vulnerability is present in the code "
    inputs       = examples["Insecure_code"]
    outputs_list = []
    for id, encoder_code, copilot_code in zip(examples["ID"], examples["Incoder_code"], examples["Copilot_code"]):

        outputs = "The above code is insecure code and the CWE Id for the Insecure code is: {}\nFor the above Insecure code the following is the secure code\n\n{}".format(id, encoder_code)
        outputs_list.append(outputs)
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs_list):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Map the data
custom_dataset = custom_dataset.map(formatting_prompts_func, batched=True)
dataset = custom_dataset
# Now you can use custom_dataset for fine-tuning


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="wandb"
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()



#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def generate_language_model_output(instruction, model, tokenizer, alpaca_prompt, input=None, max_tokens=128):
    # Format the prompt using the provided input range
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction, # instruction
            input, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = max_tokens, use_cache = True)
    decoded_outputs = tokenizer.batch_decode(outputs)

    return decoded_outputs


# instruction = dataset[1]["Prompt"]


instruction = '''Get raw_data from request with utf-8 encoding and decode it with base64.
    Deserialize the decoded data, store it in a variable and return it

 Is the given code secure from attacks or Insecure from attacks'''

input = '''
import mysql.connector


def get_connection():
    \'''
    Return a connection to the database
    \'''
    return mysql.connector.connect(host=""localhost"", user="""", passwd="""")


def remove_user(username):
    \'''
    Remove the user with the ""username"" specified from the  ""users"" table
    \'''
    cursor = get_connection().cursor()
    cursor.execute(""DELETE FROM users WHERE username = '%s'"" % username)

'''

# input = dataset[1]["Insecure_code"]

# print(instruction)
# print("input\n\n")
# print(input)

generated_output = generate_language_model_output(instruction, model, tokenizer, alpaca_prompt)
response = generated_output[-1].split("### Response:\n")[-1].strip()

print("Answer:{}".format(response))


model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")