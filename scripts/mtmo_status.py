"""Read the pipeline and training progress without touching a running model."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', nargs='?', default='runs/mtmo_n100_seed2023')
args = parser.parse_args()
root = Path(args.run_dir)
status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {'status': 'not_started'}
log = root / 'training/train.jsonl'
if log.exists():
    last = None
    with log.open('rb') as f:
        f.seek(0, 2)
        length = f.tell()
        f.seek(max(0, length - 131072))
        for line in f.read().splitlines():
            try:
                last = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
    if last:
        status['training'] = last
        status['training']['progress_fraction'] = last.get('total_instances', 0) / 100000000
status['evaluation_cases_completed'] = len(list((root / 'evaluation').glob('*_n*.json')))
print(json.dumps(status, ensure_ascii=False, indent=2))
