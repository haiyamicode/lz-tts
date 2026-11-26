# lzai (lz-tts)

Lazybird's speech + media GPU worker. Project notes live in `CLAUDE.md` (local).

## Deploy (home box, prod)

```bash
cd ~/Projects/lz-tts && scripts/deploy_home.sh   # --skip-data: code only; --restart: restart at the same commit
```

Fetches origin/main, pulls S3 `data/runtime-wheels` and refuses before touching anything if a wheel the new
commit installs is missing, pulls (fast-forward only, clean tree), syncs the model data each pm2 app serves,
`uv sync --frozen` (main + media venvs), then restarts `lzai-gpu0` and `lzai-gpu1` one at a time, each
waiting for `/health` 200 and "Taskflow worker ready"; it stops at the first one that does not come back.
Docker images get the same wheels from S3 in `docker/entrypoint.sh` before `uv sync`.
