import asyncio
import json
import os
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mlx_lm import load, stream_generate

MODEL_ID = os.getenv("BONSAI_MODEL_ID", "prism-ml/Bonsai-8B-mlx-1bit")
DEFAULT_MAX_TOKENS = int(os.getenv("BONSAI_MAX_TOKENS", "200"))

app = FastAPI(title="Bonsai 8B Local API Wrapper")

model, tokenizer = load(MODEL_ID)


@app.get("/")
async def root():
    return {
        "service": "Bonsai 8B Local API Wrapper",
        "model": MODEL_ID,
        "websocket": "/ws",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


def build_prompt(user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_payload(data: dict) -> dict:
    if data.get("json") is not None:
        return data["json"]
    if data.get("text") is not None:
        return json.loads(data["text"])
    return {}


def stream_response(prompt: str, max_tokens: int, token_queue: queue.Queue) -> None:
    # Suppress model reasoning only when it appears inside explicit think tags.
    in_think = False
    think_done = False
    pre_think_buf = ""

    try:
        generator = stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        for chunk in generator:
            token = chunk.text

            if think_done:
                token_queue.put(("token", token))
                continue

            pre_think_buf += token

            if not in_think:
                if "<think>" in pre_think_buf:
                    in_think = True
                else:
                    token_queue.put(("token", token))
            elif "</think>" in pre_think_buf:
                think_done = True
                after = pre_think_buf.split("</think>", 1)[1]
                if after:
                    token_queue.put(("token", after))

        token_queue.put(("end", None))

    except Exception as exc:
        token_queue.put(("error", str(exc)))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive()

            try:
                payload = parse_payload(data)
            except json.JSONDecodeError:
                await ws.send_json({"event": "error", "detail": "Invalid JSON payload"})
                continue

            user_prompt = payload.get("prompt", "")
            if not user_prompt:
                await ws.send_json({"event": "error", "detail": "Missing prompt"})
                continue

            max_tokens = int(payload.get("max_tokens", DEFAULT_MAX_TOKENS))
            prompt = build_prompt(user_prompt)

            token_queue = queue.Queue()
            worker = asyncio.create_task(
                asyncio.to_thread(stream_response, prompt, max_tokens, token_queue)
            )

            while True:
                msg_type, msg_value = await asyncio.to_thread(token_queue.get)
                if msg_type == "token":
                    await ws.send_text(msg_value)
                elif msg_type == "end":
                    await ws.send_json({"event": "end"})
                    break
                elif msg_type == "error":
                    print("Error in generation thread:", msg_value)
                    await ws.send_json({"event": "error", "detail": msg_value})
                    break

            await worker

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("Error:", e)
