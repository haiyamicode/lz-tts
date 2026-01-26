# Package installation
Use `uv add` to add packages.
Use `uv run` to run python

# Building piper-phonemize C++ extension
After modifying C++ code in `src/phonemizer/src/`:
```bash
cd /mnt/data/lz-tts/src/phonemizer && uv build
cd /mnt/data/lz-tts && uv pip install --reinstall src/phonemizer/dist/piper_phonemize-1.2.0-cp310-cp310-linux_x86_64.whl
```