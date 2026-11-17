"""Run the complete pipeline; resume only after bounded transient scheduler/network failures."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run_dir', default='runs/mtmo_n100_seed2023')
parser.add_argument('--max_attempts', type=int, default=3)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = (root / args.run_dir).resolve()
run.mkdir(parents=True, exist_ok=True)
transient = ('connection reset by peer', 'connection timed out', 'ncclsystemerror',
             'socket timeout', 'unable to create step', 'temporarily unavailable', 'node failure')
for attempt in range(1, args.max_attempts + 1):
    record = {'pid': os.getpid(), 'attempt': attempt,
              'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    (run / 'supervision.json').write_text(json.dumps(record, indent=2) + '\n')
    with (run / f'pipeline-attempt-{attempt}.log').open('a') as log:
        result = subprocess.run(['bash', str(root / 'scripts/run_full_mtmo.sh'), str(run)],
                                cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        break
    snippets = []
    for path in [run / f'pipeline-attempt-{attempt}.log', run / 'training.log', run / 'calibration.log']:
        if path.exists():
            with path.open('rb') as f:
                f.seek(0, 2); length = f.tell(); f.seek(max(0, length - 24000))
                snippets.append(f.read().decode(errors='replace').lower())
    text = '\n'.join(snippets)
    if attempt == args.max_attempts or not any(x in text for x in transient):
        raise SystemExit(result.returncode)
    # Never retry a numerical, objective, or memory failure as if it were transient.
    if any(x in text for x in ('out of memory', 'floatingpointerror', 'assertionerror', 'non-finite')):
        raise SystemExit(result.returncode)
    time.sleep(30)
else:
    raise SystemExit(1)
