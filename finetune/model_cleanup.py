# hf_model_cleanup.py

import os
import gc
import subprocess
import sys

def free_python_memory():
    print("\nFreeing model and tokenizer from Python RAM...")
    try:
        del model
        del tokenizer
    except NameError:
        print("No model/tokenizer objects found in this session.")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU VRAM (if used).")
    except ImportError:
        pass
    print("RAM cleanup complete (as much as Python allows).")

def scan_hf_cache():
    print("\nScanning Hugging Face cache...")
    try:
        subprocess.run(["huggingface-cli", "scan-cache"])
    except FileNotFoundError:
        print("huggingface-cli not found. Please install huggingface_hub[cli] with:")
        print('pip install -U "huggingface_hub[cli]"')

def interactive_delete_cache():
    print("\nLaunching interactive Hugging Face cache cleanup...")
    try:
        subprocess.run(["huggingface-cli", "delete-cache"])
    except FileNotFoundError:
        print("huggingface-cli not found. Please install huggingface_hub[cli] with:")
        print('pip install -U "huggingface_hub[cli]"')

def noninteractive_delete_cache():
    print("\nLaunching non-interactive Hugging Face cache cleanup...")
    try:
        subprocess.run(["huggingface-cli", "delete-cache", "--disable-tui"])
    except FileNotFoundError:
        print("huggingface-cli not found. Please install huggingface_hub[cli] with:")
        print('pip install -U "huggingface_hub[cli]"')

def manual_delete_model_folder():
    print("\nManual deletion of model or all cache files.")
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "huggingface", "hub")
    print(f"\nDefault Hugging Face cache dir: {cache_dir}")
    model_name = input("Enter the model subfolder (e.g., models--meta-llama--Llama-3.2-1B), or leave blank to delete all: ").strip()
    if model_name:
        target = os.path.join(cache_dir, model_name)
    else:
        target = cache_dir
        confirm = input("Are you sure you want to DELETE ALL Hugging Face cache? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
    print(f"Deleting: {target}")
    if os.path.exists(target):
        subprocess.run(["rm", "-rf", target])
        print("Deletion complete.")
    else:
        print("Specified folder does not exist.")

def delete_custom_model_folder():
    path = input("\nEnter path to your custom model folder to delete: ").strip()
    if os.path.exists(path):
        subprocess.run(["rm", "-rf", path])
        print(f"Deleted: {path}")
    else:
        print("Folder not found.")

def main():
    while True:
        print("\n===== Hugging Face Model Cleanup Menu =====")
        print("1. Scan Hugging Face cache (list size & models)")
        print("2. Free model/tokenizer from Python RAM")
        print("3. Interactive Hugging Face cache deletion (TUI)")
        print("4. Non-interactive cache deletion (text editor)")
        print("5. Manually delete a specific model/all from cache dir")
        print("6. Delete custom model folder from disk")
        print("0. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            scan_hf_cache()
        elif choice == "2":
            free_python_memory()
        elif choice == "3":
            interactive_delete_cache()
        elif choice == "4":
            noninteractive_delete_cache()
        elif choice == "5":
            manual_delete_model_folder()
        elif choice == "6":
            delete_custom_model_folder()
        elif choice == "0":
            print("Exiting cleanup tool.")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
