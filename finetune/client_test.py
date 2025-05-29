import requests

history = [
    "Hello, I want to book a train from London to Cambridge.",
    "Sure, what date do you want to travel?",
    "Next Friday.",
    "What time do you prefer?"
]
user_message = "Around 9am."

resp = requests.post("http://localhost:8000/chat", json={
    "history": history,
    "user_message": user_message,
    "max_new_tokens": 50
})
print(resp.json())
