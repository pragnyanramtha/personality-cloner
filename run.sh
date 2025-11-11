#!/bin/bash

# --- Stop the script if any command fails ---
set -e

# --- Color Codes for pretty output ---
C_CYAN='\033[0;36m'
C_GREEN='\033[0;32m'
C_RED='\033[0;31m'
C_YELLOW='\033[0;33m'
C_NONE='\033[0m'

# --- List of required Python scripts ---
SCRIPT_CONVERT="data_conveter.py"
SCRIPT_TRAIN="main.py"
SCRIPT_MERGE="merge_lora_adapter.py"
REQUIREMENTS="pyproject.toml"

# --- Function to check if a file exists ---
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${C_RED}Error: File not found: $1${C_NONE}"
        echo -e "${C_YELLOW}Please make sure all scripts exist"
        exit 1
    fi
}

echo -e "${C_CYAN}===== STARTING END-TO-END FINE-TUNING PIPELINE =====${C_NONE}"
echo -e "${C_YELLOW}This will run all steps. Grab a coffee, this will take a few hours.${C_NONE}"

# --- Check for all required files first ---
check_file $SCRIPT_CONVERT
check_file $SCRIPT_TRAIN
check_file $SCRIPT_MERGE
check_file $REQUIREMENTS

# --- Step 0: Set up Python Environment ---
echo -e "\n${C_GREEN}>>> Step 0/5: Setting up Python environment...${C_NONE}"
if [ -d ".venv" ]; then
    echo "Virtual environment '.venv' already exists. Activating..."
source .venv/bin/activate
echo "Installing dependencies from requirements.txt..."
uv sync
echo "Environment is ready."

# --- Step 1: Data Conversion ---
echo -e "\n${C_GREEN}>>> Step 1/5: Preparing Data Conversion...${C_NONE}"
echo -e "${C_YELLOW}!!! IMPORTANT !!!${C_NONE}"
echo -e "${C_YELLOW}You MUST edit '${C_CYAN}$SCRIPT_CONVERT${C_YELLOW}' right now.${C_NONE}"
echo -e "${C_YELLOW}Open it in a separate terminal and set your${C_RED} 'MY_NAME' ${C_YELLOW}and${C_RED} 'CHATS_FOLDER_PATH' ${C_YELLOW}variables.${C_NONE}"
read -p "Have you edited the file and saved it? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${C_RED}Exiting. Please edit the script and run again.${C_NONE}"
    exit 1
fi
echo "Running data conversion..."
python $SCRIPT_CONVERT
echo "Data conversion complete."

# --- Step 2: Model Training ---
echo -e "\n${C_GREEN}>>> Step 2/5: Starting Model Training...${C_NONE}"
echo -e "${C_YELLOW}This is the longest step. The L4 GPU will be at 100%.${C_NONE}"
python $SCRIPT_TRAIN
echo "Training complete. LoRA adapters saved to 'lora_model'."

# --- Step 3: Merge and Quantize ---
echo -e "\n${C_GREEN}>>> Step 3/5: Merging Adapters and Quantizing to GGUF...${C_NONE}"
echo -e "${C_YELLOW}This will take 15-20 minutes. It will build llama.cpp.${C_NONE}"
python $SCRIPT_MERGE
echo "GGUF quantization complete. Model saved to 'gguf_model' folder."

# --- Step 4: Find GGUF and Create Modelfile ---
echo -e "\n${C_GREEN}>>> Step 4/5: Creating Ollama Modelfile...${C_NONE}"
# Find the GGUF file path
GGUF_PATH=$(find ./gguf_model -name "*.gguf" -print -quit)

if [ -z "$GGUF_PATH" ]; then
    echo -e "${C_RED}Error: Could not find a .gguf file in the 'gguf_model' directory.${C_NONE}"
    exit 1
fi

echo "Found GGUF at: $GGUF_PATH"



# --- Step 5: All Done! ---
echo -e "\n${C_GREEN}>>> Step 5/5: ALL DONE! 🔥${C_NONE}"
echo -e "${C_GREEN}The entire pipeline is complete. Your clone is ready.${C_NONE}"
echo -e "Your next steps are on your local machine:"
echo -e "1. Move the ${C_YELLOW}'Modelfile'${C_NONE} and the ${C_YELLOW}'$GGUF_PATH'${C_NONE} to your laptop."
echo -e "2. In your terminal, 'cd' to that folder and run:"
echo -e "${C_CYAN}   ollama create yourname-clone -f ./Modelfile(needs editing)${C_NONE}"
echo -e "3. After it's created, run your clone:"
echo -e "${C_CYAN}   ollama run yourname-clone${C_NONE}"
echo -e "\n${C_GREEN}===== PIPELINE FINISHED =====${C_NONE}"
