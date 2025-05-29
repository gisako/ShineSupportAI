import os
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "../../multiwoz-master/parsed/distilgpt-finetuned-multiwoz"

# Load model and tokenizer
print("Loading model and tokenizer. This may take a moment...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
print("Model loaded successfully!\n")

# System prompt with greeting example to encourage friendly behavior
SYSTEM_PROMPT = (
    "<|system|> You are a friendly and polite assistant for hotel and restaurant bookings in Cambridge. "
    "<|user|> hello <|assistant|> Hello! How can I assist you today?"
)

# List of greeting keywords for manual interception
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]

def extract_assistant_response(full_text):
    # Only return the response after the last <|assistant|>, stopping at next tag or end
    parts = full_text.split("<|assistant|>")
    if len(parts) > 1:
        return parts[-1].split("<|user|>")[0].split("<|system|>")[0].strip()
    else:
        return full_text.strip()

def generate_response(prompt, max_new_tokens=60):
    encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_assistant_response(full_response)

def menu():
    while True:
        print("\n--- Customer Support AI Testing Menu ---")
        print("1. Test a single prompt (stateless)")
        print("2. Test a batch of sample prompts (stateless)")
        print("3. Interactive chat with context and greetings")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            user_prompt = input("\nEnter a customer question: ").strip()
            # Manual greeting interception
            if user_prompt.lower() in GREETINGS:
                print("\nAI Response: Hello! How can I assist you today?")
            else:
                prompt = f"{SYSTEM_PROMPT} <|user|> {user_prompt} <|assistant|>"
                response = generate_response(prompt)
                print("\nAI Response:", response)

        elif choice == "2":
            test_prompts = [
                "How do I reset my password?",
                "The app keeps crashing when I open it. What should I do?",
                "How can I change my billing information?",
                "I want to cancel my subscription.",
                "Can I transfer my account to another email?",
                "hello"
            ]
            print("\n--- Batch Testing ---")
            for user_prompt in test_prompts:
                # Manual greeting interception
                if user_prompt.lower() in GREETINGS:
                    response = "Hello! How can I assist you today?"
                else:
                    prompt = f"{SYSTEM_PROMPT} <|user|> {user_prompt} <|assistant|>"
                    response = generate_response(prompt)
                print(f"Prompt: {user_prompt}")
                print(f"AI Response: {response}")
                print("-" * 50)

        elif choice == "3":
            print("\nType 'exit' or 'quit' to end the chat.")
            dialogue_history = []
            while True:
                user_input = input("Customer: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Ending interactive chat.\n")
                    break
                # Manual greeting interception
                if user_input.lower() in GREETINGS:
                    print("AI: Hello! How can I assist you today?")
                    dialogue_history.append(f"<|user|> {user_input}")
                    dialogue_history.append("<|assistant|> Hello! How can I assist you today?")
                    continue
                # Add user message to history
                dialogue_history.append(f"<|user|> {user_input}")
                # Use system prompt + last 4 turns for brevity
                recent_history = dialogue_history[-4:]
                prompt = f"{SYSTEM_PROMPT} {' '.join(recent_history)} <|assistant|>"
                response = generate_response(prompt)
                print("AI:", response)
                # Add model response to history for next round
                dialogue_history.append(f"<|assistant|> {response}")

        elif choice == "4":
            print("Exiting. Have a great day!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    menu()
