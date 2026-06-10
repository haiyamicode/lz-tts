# LZ-TTS API Routes

## Health & Info

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check with server info, engine status, loaded models, speakers |
| GET | `/llms.txt` | LLM-readable plaintext API documentation |

## PiperTTS (Sparrow/VITS)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/synthesize` | Text/SSML → speech (WAV or MP3). Supports multilingual auto-routing, speaker override, `primary_speaker`, noise/length params. |
| GET  | `/synthesize` | Same as POST via query params (for easy testing). |
| GET  | `/models` | List models enabled for on-demand use. |
| GET  | `/models/{model}` | Model info: name, speakers, BERT status. |
| GET  | `/models/{model}/speakers` | Speaker list for a model. |

## Seed-VC (Voice Conversion)

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/seed-vc/status` | Seed-VC backend status (loaded, device, presets). |
| POST | `/vc` | Single voice conversion: base64 source audio → MP3 via reference voice. |
| POST | `/vc-batch` | Batched voice conversion (all items share same target voice). |
| POST | `/find-voice` | Lookup voice ID from a reference URL. |
| POST | `/enhance` | Download reference audio, normalize, enhance, return MP3. |

## Qwen3 (Falcon)

| Method | Path | Description |
|--------|------|-------------|
| (Qwen3 routes from `qwen3_router`) | See `src/api/qwen3.py` for full route list. |
| GET  | `/qwen3/demo` | Basic-auth protected demo UI (mounted via `_mount_qwen_demo`). |

## RVC (Retrieval-based Voice Conversion)

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/rvc/status` | RVC backend status (loaded, default model, available models). |
| GET  | `/rvc/models` | List available RVC .pth model weights. |
| POST | `/rvc/convert` | Voice conversion: base64 source audio → WAV/MP3 via RVC model. Supports model selection, pitch shift, f0 method, index/rms/protect params. |
