import torch
from unsloth import FastLlamaModel
import os

# Check if CUDA (GPU) is available
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. This script requires a GPU.")
print("CUDA is available. Ready to merge and quantize.")

max_seq_length = 8192
lora_model_dir = "lora_model"

if not os.path.exists(lora_model_dir):
    print(f"Error: LoRA model directory '{lora_model_dir}' not found.")
    exit()

# --- 1. Load the BASE Model & Tokenizer ---
print("Loading base model 'unsloth/llama-3.1-8b-Instruct'...")
model, tokenizer = FastLlamaModel.from_pretrained(
    model_name = "unsloth/llama-3.1-8b-Instruct",
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False,
)

# --- 2. Create the Unsloth PEFT/LoRA Model ---
print("Applying Unsloth PEFT model patch...")
model = FastLlamaModel.get_peft_model(
    model,
    r = 32, # Must match the 'r' you trained with
    lora_alpha = 64, # Must match the 'lora_alpha' you trained with
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Must match the 'target_modules' you trained with
    lora_dropout = 0.05, # Must match
    bias = "none",
    random_state = 42,
)

# --- 3. Load Your Trained Adapter Weights ---
print(f"Loading adapter weights from '{lora_model_dir}'...")
model.load_adapter(lora_model_dir, adapter_name = "default")
print("Adapter weights loaded!")

# --- 4. Merge and Save to GGUF ---
output_folder = "gguf_model"
print(f"Merging and saving to GGUF in folder: {output_folder}")

model.save_pretrained_gguf(
    output_folder, 
    tokenizer,
    quantization_method = "q4_k_m"
)

print(f"✅ All done! Your GGUF file is saved inside the '{output_folder}' folder.")
print(f"Look for a file like 'llama-3.1-8b-Instruct.Q4_K_M.gguf' inside.")
