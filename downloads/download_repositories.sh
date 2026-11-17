#!/usr/bin/env bash
set -euo pipefail
MTMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MTMO_ROOT"
mkdir -p .upstream-git downloads/logs

while IFS=$'\t' read -r name url revision; do
    metadata="$MTMO_ROOT/.upstream-git/$name.git"
    if [ ! -d "$metadata" ]; then
        git clone --bare "$url" "$metadata" > "downloads/logs/clone-$name.log" 2>&1
    fi
    if ! git --git-dir="$metadata" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git --git-dir="$metadata" fetch origin "$revision" >> "downloads/logs/clone-$name.log" 2>&1
    fi
    mkdir -p "$name"
    git --git-dir="$metadata" archive "$revision" | tar --extract --skip-old-files --directory="$name"
    printf '%s restored missing files from %s\n' "$name" "$revision"
done < <(python - <<'PY'
import json
from pathlib import Path
for repo in json.loads(Path('UPSTREAMS.json').read_text())['repositories']:
    print(repo['directory'], repo['url'], repo['commit'], sep='\t')
PY
)
