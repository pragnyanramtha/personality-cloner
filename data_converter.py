import json
import os
import sys

# --- 1. CONFIGURE YOUR SETTINGS ---

MY_NAME = input("Your exact instagram username: ") 


CHATS_FOLDER_PATH = input("path to your chat data (instagram/Whatsapp): ")

# 3. Name of the final output file
OUTPUT_FILE_NAME = "train_data.jsonl"

# 4. How many messages to include in each training example.
#    This "sliding window" is what we used before. 5 is a good default.
CONTEXT_WINDOW_SIZE = 5

# --- END OF CONFIGURATION ---


def fix_encoding(text):
    """
    Fixes the notorious Instagram latin-1 encoding bug.
    
    Meta's JSONs are "utf-8" files, but the 'content' strings
    are encoded as 'latin-1' *within* them. This function
    reverses that, giving you correct UTF-8 strings with emojis.
    """
    try:
        # Re-encode the 'utf-8' (mojibake) string back to its original 'latin-1' bytes
        # Then, decode those bytes as 'utf-8' to get the correct string.
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If it's already a valid utf-8 string, just return it.
        return text

def process_chat_file(file_path, my_name):
    """
    Reads a single message_1.json file and returns a list
    of all messages in chronological order with correct roles.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [!] Error reading {file_path}: {e}")
        return []

    # Check if this is a valid chat file
    if 'messages' not in data or 'participants' not in data:
        print(f"  [*] Skipping {file_path}: Not a valid chat file.")
        return []
    
    # Check for group chats and warn the user
    if len(data['participants']) > 2:
        print(f"  [!] Warning: {file_path} is a group chat.")
        print("      All other participants will be mapped to the 'user' role.")
    
    # Meta exports are newest-to-oldest. We must reverse
    # to get a chronological conversation.
    messages = data['messages']
    messages.reverse()
    
    processed_messages = []
    for msg in messages:
        # Skip system messages, reactions, or messages with no content
        if 'content' not in msg or 'sender_name' not in msg:
            continue
            
        sender = msg['sender_name']
        
        # Apply the encoding fix
        content = fix_encoding(msg['content'])
        
        # Assign the role
        role = "assistant" if sender == my_name else "user"
        
        processed_messages.append({"role": role, "content": content})
        
    return processed_messages

def create_sliding_windows(all_messages, window_size):
    """
    Creates sliding windows from a long list of messages.
    Each window is a training example.
    """
    windows = []
    for i in range(len(all_messages) - window_size + 1):
        window = all_messages[i : i + window_size]
        
        # CRITICAL: We only want examples where "I" (the assistant)
        # am the one responding. This is what the model learns.
        if window[-1]['role'] == 'assistant':
            windows.append({"messages": window})
            
    return windows

def main():
    print("Starting chat conversion...")
    
    if MY_NAME == "Your Name Here":
        print("!!! ERROR: Please update 'MY_NAME' in the script first.")
        sys.exit(1)
        
    if not os.path.exists(CHATS_FOLDER_PATH):
        print(f"!!! ERROR: Chat folder not found at: {CHATS_FOLDER_PATH}")
        print("    Please update 'CHATS_FOLDER_PATH' to the correct path.")
        sys.exit(1)

    all_conversations = []
    
    # Use os.walk to go through all subfolders
    for dirpath, dirnames, filenames in os.walk(CHATS_FOLDER_PATH):
        for filename in filenames:
            # We only care about the JSON files
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                print(f"[+] Processing {file_path}...")
                
                # 1. Get all messages from this file
                chronological_messages = process_chat_file(file_path, MY_NAME)
                
                if not chronological_messages:
                    continue
                    
                # 2. Create the sliding windows
                conversation_windows = create_sliding_windows(chronological_messages, CONTEXT_WINDOW_SIZE)
                
                all_conversations.extend(conversation_windows)

    if not all_conversations:
        print("!!! ERROR: No conversations were found!")
        print("    Did you set the correct 'MY_NAME'?")
        print(f"    Is the '{CHATS_FOLDER_PATH}' folder correct?")
        sys.exit(1)

    # 3. Write all windows to the final .jsonl file
    try:
        with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
            for conv in all_conversations:
                # ensure_ascii=False keeps our fixed emojis
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"!!! ERROR writing to output file: {e}")
        sys.exit(1)
        
    print("\n" + "="*30)
    print("✅ Conversion Complete!")
    print(f"   Total conversations found: {len(all_conversations)}")
    print(f"   Output file created: {OUTPUT_FILE_NAME}")
    print("="*30)


if __name__ == "__main__":
    main()
