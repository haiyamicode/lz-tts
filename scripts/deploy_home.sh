#!/usr/bin/env bash
# Deploy lz-tts main to the home box's pm2 workers (prod), one worker at a time.
#
#   cd ~/Projects/lz-tts && scripts/deploy_home.sh [--skip-data] [--restart]
#
# 1. Fetches origin/main; refuses a tree with tracked changes or a non-fast-forward.
# 2. Downloads data/runtime-wheels from S3 and checks every local `path =` wheel the new commit's
#    pyproject/uv.lock (main + src/media_runtime) installs: a missing one stops here, before the
#    tree or a venv is touched.
# 3. git pull --ff-only, then the model data each worker serves (scripts/download_data.py with that
#    pm2 app's LZAI_SERVER_CONFIG / LZAI_MODELS; per-file ETag check, --skip-data skips it).
# 4. uv sync --frozen for the main venv and, when present, the media runtime venv.
# 5. Rolling restart (APPS, default lzai-gpu0 then lzai-gpu1): pm2 restart, then wait for the app's
#    /health 200 on its PORT (default 8000) and "Taskflow worker ready" in its new log lines. One that does not come
#    back stops the script, so the others keep serving the old code (pm2 shows which one).
# 6. Prints the old and new HEAD, the commits and the package changes.
# Nothing new on origin/main: stops after step 1 unless --restart. A run that failed after the pull
# leaves local/.deploy-pending/ (the HEAD and package lists the workers started from); the next run
# sees it and carries on from there (data, sync, restarts), removing it once every app is back.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
APPS=(${APPS:-lzai-gpu0 lzai-gpu1})
READY_TIMEOUT=${READY_TIMEOUT:-900}
SKIP_DATA=0
FORCE_RESTART=0
for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=1 ;;
        --restart) FORCE_RESTART=1 ;;
        *) echo "usage: $0 [--skip-data] [--restart]" >&2; exit 2 ;;
    esac
done
for tool in git uv pm2 curl python3; do
    command -v "$tool" >/dev/null || { echo "deploy: $tool not on PATH" >&2; exit 1; }
done

step() { echo; echo "== $*"; }
fail() { echo "deploy: FAILED: $*" >&2; exit 1; }
dl() { uv run --no-project --with boto3 --with python-dotenv --with pyyaml python scripts/download_data.py "$@"; }
# One field of a pm2 app: its env var (PORT, LZAI_*), or out_log / err_log. Never prints the env.
app_field() {
    pm2 jlist | python3 -c '
import json, sys
app, field = sys.argv[1], sys.argv[2]
for proc in json.load(sys.stdin):
    if proc["name"] == app:
        env = proc["pm2_env"]
        keys = {"out_log": "pm_out_log_path", "err_log": "pm_err_log_path", "pid": "pid"}
        value = proc.get("pid") if field == "pid" else env.get(keys.get(field, field), env.get("env", {}).get(field))
        print("" if value is None else value)
        sys.exit(0)
sys.exit(f"pm2 app {app} not found")' "$1" "$2"
}
# The app's server-config selection (KEY=VALUE, NUL-separated) for download_data.py; unset keys left out
# so .env still applies as it does for the worker.
app_env() {
    pm2 jlist | python3 -c '
import json, sys
for proc in json.load(sys.stdin):
    if proc["name"] == sys.argv[1]:
        env = {**proc["pm2_env"].get("env", {}), **proc["pm2_env"]}
        for key in ("LZAI_SERVER_CONFIG", "LZAI_SERVER_CONFIG_YAML", "LZAI_MODELS"):
            if isinstance(env.get(key), str) and env[key]:
                sys.stdout.write(f"{key}={env[key]}\0")' "$1"
}
venv_packages() { [ -x "$1/bin/python" ] && uv pip freeze --python "$1/bin/python" 2>/dev/null | sort || true; }

# --------------------------------------------------------------------------- 1. what to deploy
step "fetch origin/main"
for app in "${APPS[@]}"; do app_field "$app" PORT >/dev/null || fail "pm2 app $app missing"; done
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || fail "not on main"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { git status --short --untracked-files=no; fail "tracked changes in the tree"; }
git fetch --quiet origin main
PENDING=local/.deploy-pending
NEW=$(git rev-parse origin/main)
git merge-base --is-ancestor HEAD "$NEW" || fail "HEAD $(git rev-parse --short HEAD) is not an ancestor of origin/main (no fast-forward)"
if [ -f "$PENDING/old-head" ]; then
    OLD=$(cat "$PENDING/old-head")
    echo "resuming an unfinished deploy: workers started on $(git log --oneline -1 "$OLD")"
else
    OLD=$(git rev-parse HEAD)
fi
if [ "$OLD" = "$NEW" ] && [ "$FORCE_RESTART" = 0 ] && [ ! -f "$PENDING/old-head" ]; then
    echo "already at origin/main ($(git log --oneline -1)); nothing to deploy (--restart to restart anyway)"
    exit 0
fi
git log --oneline "$OLD..$NEW"

# --------------------------------------------------------------------------- 2. wheels, checked before anything changes
step "runtime wheels from S3"
dl --filter runtime-wheels
missing=0
for project in . src/media_runtime; do
    for file in pyproject.toml uv.lock; do
        git cat-file -e "$NEW:$project/$file" 2>/dev/null || continue
        while read -r wheel; do
            [ -n "$wheel" ] || continue
            if [ ! -f "$project/$wheel" ]; then echo "  missing: $project/$wheel ($file)" >&2; missing=1; fi
        done < <(git show "$NEW:$project/$file" | grep -oE 'path = "[^"]+\.whl"' | sed -E 's/path = "(.*)"/\1/' | sort -u)
    done
done
[ "$missing" = 0 ] || fail "wheels the new commit installs are not in data/runtime-wheels (upload them: scripts/upload_data.py --model runtime-wheels)"
echo "every path wheel of $(git rev-parse --short "$NEW") is present"

# --------------------------------------------------------------------------- 3. code + served model data
if [ ! -f "$PENDING/old-head" ]; then
    mkdir -p "$PENDING"
    venv_packages .venv > "$PENDING/main-packages"
    venv_packages src/media_runtime/.venv > "$PENDING/media-packages"
    echo "$OLD" > "$PENDING/old-head"
fi
step "git pull --ff-only"
git pull --ff-only --quiet origin main
[ "$(git rev-parse HEAD)" = "$NEW" ] || fail "pull did not land on $NEW"
if [ "$SKIP_DATA" = 0 ]; then
    for app in "${APPS[@]}"; do
        step "model data served by $app"
        mapfile -d '' app_env < <(app_env "$app")
        env -u LZAI_SERVER_CONFIG -u LZAI_SERVER_CONFIG_YAML -u LZAI_MODELS -u LZAI_REQUIRED_MODELS "${app_env[@]}" \
            uv run --no-project --with boto3 --with python-dotenv --with pyyaml python scripts/download_data.py --stage all \
            || fail "data download for $app (code is pulled, workers still run the old code; rerun to resume)"
    done
fi

# --------------------------------------------------------------------------- 4. venvs
step "uv sync --frozen"
uv sync --frozen || fail "uv sync (workers still run the old code; fix it and rerun to resume)"
if [ -d src/media_runtime/.venv ]; then
    (cd src/media_runtime && uv sync --frozen) || fail "media runtime uv sync (workers still run the old code; fix it and rerun to resume)"
fi

# --------------------------------------------------------------------------- 5. rolling restart
for app in "${APPS[@]}"; do
    port=$(app_field "$app" PORT)
    port=${port:-8000}  # the worker's own default (src/api/server.py) when its pm2 env sets none
    step "restart $app (port $port)"
    out_log=$(app_field "$app" out_log)
    err_log=$(app_field "$app" err_log)
    out_size=$(stat -c %s "$out_log" 2>/dev/null || echo 0)
    err_size=$(stat -c %s "$err_log" 2>/dev/null || echo 0)
    old_pid=$(app_field "$app" pid)
    pm2 restart "$app" >/dev/null
    started=$(date +%s)
    while :; do
        elapsed=$(( $(date +%s) - started ))
        health=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" || true)
        ready=0
        for pair in "$out_log:$out_size" "$err_log:$err_size"; do
            log=${pair%:*}; size=${pair##*:}
            [ -f "$log" ] || continue
            [ "$(stat -c %s "$log")" -ge "$size" ] || size=0  # rotated
            tail -c "+$((size + 1))" "$log" | grep -q "Taskflow worker ready" && ready=1
        done
        pid=$(app_field "$app" pid)
        if [ "$health" = 200 ] && [ "$ready" = 1 ] && [ "$pid" != "$old_pid" ]; then
            echo "$app ready in ${elapsed}s (pid $pid)"
            break
        fi
        if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
            curl -s -m 5 "http://127.0.0.1:$port/health" || true; echo
            pm2 describe "$app" | grep -E "status|restarts|uptime" || true
            fail "$app not ready after ${READY_TIMEOUT}s (health $health, ready line $ready); later apps not restarted: check pm2 logs $app, then rerun to resume"
        fi
        sleep 5
    done
done

# --------------------------------------------------------------------------- 6. report
step "deployed"
BEFORE_MAIN=$(cat "$PENDING/main-packages")
BEFORE_MEDIA=$(cat "$PENDING/media-packages")
rm -rf "$PENDING"
echo "old HEAD: $OLD"
echo "new HEAD: $NEW"
git log --oneline "$OLD..$NEW"
echo "-- main venv package changes:"
{ diff <(echo "$BEFORE_MAIN") <(venv_packages .venv) || true; } | grep '^[<>]' || echo "   none"
if [ -d src/media_runtime/.venv ]; then
    echo "-- media venv package changes:"
    { diff <(echo "$BEFORE_MEDIA") <(venv_packages src/media_runtime/.venv) || true; } | grep '^[<>]' || echo "   none"
fi
