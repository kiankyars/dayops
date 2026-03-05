#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_PATH="$REPO_DIR/com.kian.dayops.plist"
PLIST_OUT="$HOME/Library/LaunchAgents/com.kian.dayops.plist"
DOTENV_PATH="$REPO_DIR/.env"
LOG_DIR="$REPO_DIR/logs"

read_dotenv_value() {
  python3 - "$DOTENV_PATH" "$1" <<'PY'
import sys
from pathlib import Path

dotenv_path = Path(sys.argv[1])
key = sys.argv[2]
if not dotenv_path.exists():
    raise SystemExit(0)

for raw in dotenv_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    k = k.strip()
    if k != key:
        continue
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    print(v)
    break
PY
}

VOICE_MEMOS_DIR="${VOICE_MEMOS_DIR:-$(read_dotenv_value VOICE_MEMOS_DIR)}"
LABEL="${DAYOPS_LAUNCHD_LABEL:-$(read_dotenv_value DAYOPS_LAUNCHD_LABEL)}"
LABEL="${LABEL:-com.kian.dayops}"

export REPO_DIR
export LABEL
export VOICE_MEMOS_DIR
export DOTENV_PATH

: "${VOICE_MEMOS_DIR:?Set VOICE_MEMOS_DIR in .env}"

mkdir -p "$(dirname "$PLIST_OUT")" "$LOG_DIR"

python3 - "$TEMPLATE_PATH" "$PLIST_OUT" <<'PY'
import html
from pathlib import Path
import os
import sys

def parse_dotenv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        data[key] = value
    return data

template = Path(sys.argv[1]).read_text()
repo = Path(os.environ["REPO_DIR"]).expanduser().resolve()
dotenv_values = parse_dotenv(Path(os.environ["DOTENV_PATH"]))
env_values = dict(dotenv_values)
for key in list(dotenv_values.keys()):
    override = os.environ.get(key)
    if override is not None and override != "":
        env_values[key] = override

env_lines = []
for key in sorted(env_values.keys()):
    value = env_values[key]
    env_lines.append(
        f"        <key>{html.escape(key)}</key>\n"
        f"        <string>{html.escape(value)}</string>"
    )
environment_block = "\n".join(env_lines)

replacements = {
    "__DAYOPS_LABEL__": os.environ["LABEL"],
    "__DAYOPS_RUNNER_SCRIPT__": str((repo / "run_dayops.sh").resolve()),
    "__VOICE_MEMOS_DIR__": str(Path(os.environ["VOICE_MEMOS_DIR"]).expanduser().resolve()),
    "__WORKING_DIR__": str(repo),
    "__STDOUT_LOG__": str((repo / "logs" / "launchd_stdout.log").resolve()),
    "__STDERR_LOG__": str((repo / "logs" / "launchd_stderr.log").resolve()),
    "__ENVIRONMENT_VARIABLES__": environment_block,
}
for key, value in replacements.items():
    template = template.replace(key, value)
Path(sys.argv[2]).write_text(template)
PY

UID_VALUE="$(id -u)"
TARGET="gui/${UID_VALUE}/${LABEL}"

launchctl bootout "gui/${UID_VALUE}" "$PLIST_OUT" >/dev/null 2>&1 || true
launchctl bootout "$TARGET" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${UID_VALUE}" "$PLIST_OUT"
launchctl enable "$TARGET"
launchctl kickstart -k "$TARGET"

echo "Installed and started $LABEL"
echo "plist: $PLIST_OUT"
