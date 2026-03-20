## 🔥 Llama 3.1 Personality Clone Pipeline 🔥

This is the full end-to-end MLOps pipeline to build a personality-cloned AI of yourself.

No cap, this isn't just a simple script. This is a full-stack AI workflow that takes your raw, messy chat data, fine-tunes a state-of-the-art LLM (Llama 3.1 8B), and deploys it as a quantized GGUF file that runs locally on your own machine with Ollama.

From a cloud GPU to your laptop, this is the whole journey. 🚀
💻 The Stack

This pipeline is built on the 2025 meta for efficient LLM engineering:

    Model: Llama 3.1 8B Instruct

    Performance: Unsloth (for 2x+ faster training & 70% less VRAM)

    Fine-Tuning: QLoRA (4-bit quantization)

    Deployment: GGUF (q4_k_m)

    Runner: Ollama

    Orchestration: Bash Scripting

⚡ The Full Pipeline: How to Run This

This whole repo is automated with a single bash script.
Step 0: Get Your Data

This is the one manual step. You need your chat logs.

    Go to your Instagram Account Center > "Your information and permissions" > "Download your information".

    Request a download. Select JSON format.

    You only need to select "Messages".

    Unzip the file. You're looking for the messages/inbox folder. This is your CHATS_FOLDER_PATH.

Step 1: Run the Pipeline Script

The run_pipeline.sh script automates everything.

First, make it executable:

chmod +x run.sh

Then, just run it:

./run.sh

The script will:

    Set up a Python virtual environment (venv) and install all dependencies.

    Stop and ask you to manually edit convert_meta_chats.py to add your MY_NAME and CHATS_FOLDER_PATH. (This is the only time you'll need to do anything).

    Run the data conversion script to create your train_cleaned.jsonl.

    Run the training script (this is the long part) to fine-tune the model and create your lora_model.

    Run the merge script to combine your LoRA with the base model and quantize it to a GGUF file.

    Create your Modelfile for you, automatically pointing to the new GGUF.

Step 2: Run Your Clone

After the pipeline is done, your gguf_model folder and Modelfile are ready.

    Move the gguf_model folder and the Modelfile to your local laptop (the one with your 4060).

    Open your terminal and cd into that folder.

    Create the Ollama model:

    ollama create your-clone -f ./Modelfile

    Run your clone. It's time to talk to yourself.

    ollama run your-clone

🛠️ Script Breakdown

This pipeline is built from a few key files:

    run.sh: The main orchestrator. It runs everything in order and stops if anything fails.

    data_converter.py: The (arguably) most important script. It handles the nightmare of Meta's JSON, fixes the notorious latin-1 encoding bug, and creates clean, ShareGPT-formatted training data.

    main.py: The core Unsloth/QLoRA training script. This is where the model learns your personality. It's set up to pre-process the data and handle all the tricky formatting_func logic.

    merge_lora_adapter.py : The final MLOps step. It loads the 16-bit base model, correctly attaches your trained adapters, and uses Unsloth's tools to merge and quantize the final product into a GGUF.

    pyproject.toml: All the pip dependencies.

    Modelfile: The "recipe" for Ollama, created for you by the bash script.

This project is a high-level demo of a full, data-to-deployment AI workflow.
