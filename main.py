import torch
from unsloth import FastLlamaModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Check if CUDA (GPU) is available
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. This script requires a GPU.")
print("CUDA is available. Let's get this bread. 🔥")

# 1. --- Load Model & Tokenizer (with Unsloth) ---
# depends on how long your avg conversations are...
max_seq_length = 8192
dtype = None
# you can choose a different model but,you probably would need to change some configs 
model_name = "unsloth/llama-3.1-8b-Instruct"

model, tokenizer = FastLlamaModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    # if you wanna load in 4 bit 
    load_in_4bit = True,
)

# 2. --- Configure LoRA (PEFT) ---
model = FastLlamaModel.get_peft_model(
    model,
    r = 32,
    lora_alpha = 64,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 42,
)

# 3. --- Load Your Cleaned Dataset ---
train_file = "train_data.jsonl"
val_file = "val_data.jsonl"

dataset = load_dataset("json", data_files={
    "train": train_file,
    "validation": val_file
})

print(f"Original dataset loaded. Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")


# 4. --- Define the BATCH-AWARE Formatting Function ---
def formatting_func(batch):
    # 'batch['messages']' is a LIST of conversations
    output_texts = [
        tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        ) 
        for conversation in batch['messages']
    ]
    return { "text": output_texts }


# 5. --- PRE-PROCESS THE DATASET (THE FIX) ---
# We manually .map() our function *before* passing it to the trainer.
# This creates a new dataset with a 'text' column.
print("Formatting dataset...")
processed_dataset = dataset.map(
    formatting_func,
    batched=True,
    num_proc=2, # Use the same num_proc as SFTConfig
    remove_columns=list(dataset["train"].features), # Drop old columns
)

print(f"Dataset formatted. New features: {processed_dataset['train'].features}")


# 6. --- Set Up SFTConfig ---
training_args = SFTConfig(
    # SFT-specific args
    max_seq_length = max_seq_length,
    dataset_num_proc = 2, 
    packing = False, 

    # Standard TrainingArguments
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    warmup_steps = 10,
    num_train_epochs = 1,
    learning_rate = 2e-4,
    bf16 = True,
    logging_steps = 1,
    optim = "paged_adamw_8bit",
    seed = 42,
    do_eval = True,
    eval_strategy = "steps",
    eval_steps = 50,
    save_strategy = "steps",
    save_steps = 50,
    save_total_limit = 2,
    load_best_model_at_end = True,
)

# 7. --- Set Up Trainer ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = processed_dataset["train"],
    eval_dataset = processed_dataset["validation"],
    args = training_args,
    dataset_text_field = "text",
)

# 8. --- Train the Model ---
print("Starting training...")
trainer.train()

print("Training finished!")

# 9. --- Save the Final LoRA Adapters ---
output_dir = "lora_model"
trainer.save_model(output_dir)

print(f"✅ Training complete. Your final LoRA adapters are saved in: {output_dir}")
