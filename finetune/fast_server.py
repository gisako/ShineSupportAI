from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio

MODEL_PATH = "./distilgpt-finetuned-multiwoz"

app = FastAPI(
    title="Async + WebSocket MultiWOZ Chatbot"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

class ChatRequest(BaseModel):
    history: list[str]
    user_message: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/chat")
async def chat(request: ChatRequest):
    # Build prompt
    prompt = ""
    for i, msg in enumerate(request.history):
        speaker = "User" if i % 2 == 0 else "Agent"
        prompt += f"{speaker}: {msg}\n"
    prompt += f"User: {request.user_message}\nAgent:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded[len(prompt):].strip()
    reply = reply.split('\n')[0]

    return {
        "history": request.history + [request.user_message, reply],
        "response": reply
    }

# --- WebSocket endpoint for streaming chat ---
@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    history = []
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("user_message", "")
            # Build prompt from history + latest user input
            prompt = ""
            for i, msg in enumerate(history):
                speaker = "User" if i % 2 == 0 else "Agent"
                prompt += f"{speaker}: {msg}\n"
            prompt += f"User: {user_message}\nAgent:"

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            # Stream tokens one-by-one
            output_ids = input_ids
            generated = ""
            for _ in range(64):  # max tokens, adjust as needed
                with torch.no_grad():
                    next_token_id = model.generate(
                        output_ids,
                        max_new_tokens=1,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True
                    )[:, -1:]

                new_token = next_token_id[0, 0].item()
                if new_token == tokenizer.eos_token_id:
                    break
                output_ids = torch.cat([output_ids, next_token_id], dim=-1)
                token_str = tokenizer.decode([new_token])
                generated += token_str
                await websocket.send_json({"token": token_str, "partial": generated})

                # This sleep is to yield control for async and simulate streaming pace
                await asyncio.sleep(0.02)

            reply = generated.strip().split('\n')[0]
            history += [user_message, reply]
            await websocket.send_json({"done": True, "reply": reply, "history": history})
    except WebSocketDisconnect:
        pass
