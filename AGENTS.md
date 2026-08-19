# Package installation
Use `uv add` to add packages.
Use `uv run` to run python

# Model naming convention
- VITS/Piper based lzspeech model: Sparrow (`lzspeech sparrow`)
- Matcha based lzspeech model: Starling (`lzspeech starling`)

# Long-running tmux jobs
When running long tasks in tmux:
- Start a normal persistent tmux session.
- Run the command directly inside the session.
- Do not wrap the command so tmux exits or terminates when the command finishes.
- Do not use `tee`, manual log redirection, or capture loops unless explicitly requested.
- For debugging, inspect the tmux session directly or use the real framework logs/TensorBoard artifacts.

# Building piper-phonemize C++ extension
After modifying C++ code in `src/phonemizer/src/`:
```bash
cd /mnt/data/lz-tts/src/phonemizer && uv build
cd /mnt/data/lz-tts && uv pip install --reinstall src/phonemizer/dist/piper_phonemize-1.2.0-cp310-cp310-linux_x86_64.whl
```

# NO FAKE TESTS

Do not write tests full of mock code. They are fake and gay and useless, DO NOT!
If something has real logic and is testable, ok, you can write test for that, if not, just don't.