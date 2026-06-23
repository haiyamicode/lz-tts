# Package installation
Use `uv add` to add packages.
Use `uv run` to run python

# Model naming convention
- VITS/Piper based lzspeech model: Sparrow (`lzspeech sparrow`)
- Matcha based lzspeech model: Starling (`lzspeech starling`)
- Qwen3 based model: Falcon, but keep using the Qwen3 name until it is realized

# Long-running tmux jobs
When running long tasks in tmux:
- Start a normal persistent tmux session.
- Run the command directly inside the session.
- Do not wrap the command so tmux exits or terminates when the command finishes.
- Do not use `tee`, manual log redirection, or capture loops unless explicitly requested.
- For debugging, inspect the tmux session directly or use the real framework logs/TensorBoard artifacts.

# Qwen3 TTS precision on V100
- V100 can run Qwen3 TTS inference with `bf16`; it is simulated and slower, but works correctly.
- Do not use `fp16` for Qwen3 TTS inference on V100; it produces broken audio.
- Use the RTX 3090 for Qwen3 TTS training. V100 `bf16` is acceptable for inference checks only.

# Building piper-phonemize C++ extension
After modifying C++ code in `src/phonemizer/src/`:
```bash
cd /mnt/data/lz-tts/src/phonemizer && uv build
cd /mnt/data/lz-tts && uv pip install --reinstall src/phonemizer/dist/piper_phonemize-1.2.0-cp310-cp310-linux_x86_64.whl
```
