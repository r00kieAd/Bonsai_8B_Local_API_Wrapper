# Bonsai 8B Local API Wrapper

A small FastAPI WebSocket wrapper for running PrismML's Bonsai 8B 1-bit MLX model locally on Apple Silicon.

The server loads `prism-ml/Bonsai-8B-mlx-1bit` with `mlx-lm`, applies the tokenizer chat template, streams generated tokens over a WebSocket, and hides tokens inside explicit `<think>...</think>` blocks when the model emits them.

## Requirements

- macOS on Apple Silicon
- Python 3.11 recommended
- Xcode command line tools and Metal toolchain
- Hugging Face access for downloading the model

This model is MLX-specific. It is not a GGUF model and is not compatible with Ollama or LM Studio.

## Repository Layout

```text
Bonsai_8B_Local_API_Wrapper/
├── server.py
├── requirements.txt
├── readme.md
└── LICENSE
```

Keep commands below inside the cloned repository:

```bash
cd Bonsai_8B_Local_API_Wrapper
```

## Setup

Install and verify the Apple developer tools before installing Python packages. The PrismML MLX fork builds native Metal code, so `pip install` fails if the `metal` compiler cannot run:

```bash
xcode-select --install
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
xcodebuild -downloadComponent MetalToolchain
xcrun metal -v
```

If `xcodebuild` says the Metal Toolchain downloaded but `xcrun metal -v` still fails, mount the downloaded toolchain and put its compiler first in `PATH`:

```bash
hdiutil attach /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/*/AssetData/Restore/*.dmg
export PATH="/Volumes/MetalToolchainCryptex/Metal.xctoolchain/usr/bin:$PATH"
metal -v
```

Keep that terminal open while installing dependencies. If you reboot, you may need to mount the DMG again.

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv bonsai_vnv
source bonsai_vnv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you moved this project folder after creating `bonsai_vnv`, recreate the virtual environment. Virtual environments contain absolute paths, so a moved venv can keep pointing at the old directory:

```bash
deactivate 2>/dev/null || true
hash -r
rm -rf bonsai_vnv
python3.11 -m venv bonsai_vnv
source bonsai_vnv/bin/activate
hash -r
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional Hugging Face login:

```bash
huggingface-cli login
```

## Run The API Server

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

On the first run, the model downloads into the Hugging Face cache and Metal kernels may compile. Later starts are faster.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## WebSocket API

Connect to:

```text
ws://127.0.0.1:8000/ws
```

Send JSON:

```json
{
  "prompt": "Explain transformers simply",
  "max_tokens": 200
}
```

The server streams generated text chunks as plain WebSocket text messages. When generation completes, it sends:

```json
{"event": "end"}
```

Errors are sent as:

```json
{"event": "error", "detail": "message"}
```

## Quick Client Test

Install `websockets` from `requirements.txt`, then run:

```bash
python - <<'PY'
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        await ws.send(json.dumps({
            "prompt": "Explain transformers simply",
            "max_tokens": 120,
        }))

        async for message in ws:
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                print(message, end="", flush=True)
                continue

            if event.get("event") == "end":
                print()
                break
            if event.get("event") == "error":
                raise RuntimeError(event.get("detail"))

asyncio.run(main())
PY
```

## Configuration

The server supports these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BONSAI_MODEL_ID` | `prism-ml/Bonsai-8B-mlx-1bit` | Hugging Face model ID to load |
| `BONSAI_MAX_TOKENS` | `200` | Default generation length when a request omits `max_tokens` |

Example:

```bash
BONSAI_MAX_TOKENS=512 python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## Notes

- The model is loaded when `server.py` is imported by Uvicorn.
- Model weights are cached by Hugging Face, usually under `~/.cache/huggingface/` or `~/Library/Caches/huggingface/`.
- `bonsai_vnv/` is a local virtual environment and should not be committed.
- The PrismML MLX fork is pinned in `requirements.txt`.
- Use Python 3.11 for the venv. A Python 3.13 venv may force source builds and package combinations that are harder to debug.

## Resources

- PrismML: https://prismml.com/
- Bonsai models on Hugging Face: https://huggingface.co/collections/prism-ml/bonsai
- PrismML MLX fork: https://github.com/PrismML-Eng/mlx
