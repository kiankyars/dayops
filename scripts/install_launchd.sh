#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_PATH="$REPO_DIR/com.kian.dayops.plist"
PLIST_OUT="$HOME/Library/LaunchAgents/com.kian.dayops.plist"
LABEL="${DAYOPS_LAUNCHD_LABEL:-com.kian.dayops}"
LOG_DIR="$REPO_DIR/logs"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi
export REPO_DIR
export LABEL

: "${VOICE_MEMOS_DIR:?Set VOICE_MEMOS_DIR in .env}"

mkdir -p "$(dirname "$PLIST_OUT")" "$LOG_DIR"

python3 - "$TEMPLATE_PATH" "$PLIST_OUT" <<'PY'
from pathlib import Path
import os
import sys

template = Path(sys.argv[1]).read_text()
repo = Path(os.environ["REPO_DIR"]).expanduser().resolve()
replacements = {
    "__DAYOPS_LABEL__": os.environ["LABEL"],
    "__DAYOPS_RUNNER_SCRIPT__": str((repo / "run_dayops.sh").resolve()),
    "__VOICE_MEMOS_DIR__": str(Path(os.environ["VOICE_MEMOS_DIR"]).expanduser().resolve()),
    "__WORKING_DIR__": str(repo),
    "__STDOUT_LOG__": str((repo / "logs" / "launchd_stdout.log").resolve()),
    "__STDERR_LOG__": str((repo / "logs" / "launchd_stderr.log").resolve()),
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
