#!/usr/bin/env bash
set -euo pipefail
MTMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MTMO_ROOT"
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
RUN_DIR="${1:-$MTMO_ROOT/runs/mtmo_n100_seed2023}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
exec 9>"$RUN_DIR/pipeline.lock"
flock -n 9 || { echo 'This run already has an active pipeline.' >&2; exit 1; }
MTMO_JOB="${MTMO_SLURM_JOB:-1465}"
SRUN=(srun --jobid="$MTMO_JOB" --overlap -N1 -n1 -c4 --gres=gpu:2 --immediate=60)
STAGE=preflight
status() {
    python - "$RUN_DIR" "$1" "$STAGE" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
out=Path(sys.argv[1])/'status.json'
tmp=out.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(status=sys.argv[2],stage=sys.argv[3],updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()),indent=2)+'\n')
tmp.replace(out)
PY
}
trap 'status failed' ERR
status running
"${SRUN[@]}" nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > "$RUN_DIR/gpu_environment.log"
if [ ! -f "$RUN_DIR/source_manifest.json" ]; then
    python - "$MTMO_ROOT" "$RUN_DIR" <<'PY'
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root,run=map(Path,sys.argv[1:])
files={}
for path in sorted((root/'MTMO').rglob('*.py')):
    if '__pycache__' in path.parts: continue
    rel=path.relative_to(root/'MTMO'); out=run/'source/MTMO'/rel
    out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,out)
    files[str(rel)]=hashlib.sha256(out.read_bytes()).hexdigest()
reference={}
for path in sorted((root/'Routing-MVMoE/models').glob('*.py')):
    out=run/'source/Reference-MVMoE/models'/path.name
    out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,out)
    reference[path.name]=hashlib.sha256(out.read_bytes()).hexdigest()
for path in sorted((root/'scripts').glob('*')):
    if path.is_file() and path.suffix in {'.py','.sh'}:
        out=run/'source/scripts'/path.name
        out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,out)
shutil.copy2(root/'Routing-MVMoE/pretrained/mvmoe_4e_n100/epoch-5000.pt',run/'reference_checkpoint.pt')
shutil.copy2(root/'configs/full_n100.json',run/'protocol.json')
record=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_sha256=files,reference_source_sha256=reference)
(run/'source_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
PY
fi
python - "$RUN_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1]); record=json.loads((run/'source_manifest.json').read_text())
for rel,sha in record['source_sha256'].items():
    assert hashlib.sha256((run/'source/MTMO'/rel).read_bytes()).hexdigest()==sha,rel
for rel,sha in record['reference_source_sha256'].items():
    assert hashlib.sha256((run/'source/Reference-MVMoE/models'/rel).read_bytes()).hexdigest()==sha,rel
PY
CODE="$RUN_DIR/source/MTMO"
STAGE=calibration
status running
if [ ! -f "$RUN_DIR/scales.json" ]; then
    "${SRUN[@]}" python "$CODE/calibrate.py" --output "$RUN_DIR/scales.json" \
        --reference_code "$RUN_DIR/source/Reference-MVMoE" --checkpoint "$RUN_DIR/reference_checkpoint.pt" \
        --problem_size 100 --samples_per_task 256 --batch_size 16 --seed 2023 --gpu_id 0 \
        > "$RUN_DIR/calibration.log" 2>&1
fi
python - "$RUN_DIR/scales.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['samples_per_task']==256 and x['problem_size']==100 and x['seed']==2023
assert set(x['per_task'])=={'CVRP','OVRP','VRPB','VRPL','VRPTW','OVRPTW'}
assert not x['unseen_task_calibration'] and all(v>0 for v in x['scales'])
PY
STAGE=training
status running
RESUME=()
if [ -f "$RUN_DIR/training/latest.pt" ]; then RESUME=(--checkpoint "$RUN_DIR/training/latest.pt"); fi
"${SRUN[@]}" python -m torch.distributed.run --standalone --nproc_per_node=2 "$CODE/train.py" \
    --scales "$RUN_DIR/scales.json" --run_dir "$RUN_DIR/training" \
    --problem Train_ALL --problem_size 100 --pomo_size 100 --epochs 5000 --train_episodes 20000 \
    --train_batch_size 128 --seed 2023 --save_interval 100 --step_log_interval 50 \
    "${RESUME[@]}" >> "$RUN_DIR/training.log" 2>&1
python - "$RUN_DIR/training/latest.pt" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert x['epoch']==5000 and x['total_instances']==100000000,(x['epoch'],x['total_instances'])
assert x['args']['max_steps']==0
PY
STAGE=evaluation
status running
mkdir -p "$RUN_DIR/evaluation"
"${SRUN[@]}" bash -c '
set -euo pipefail
code=$1; checkpoint=$2; output=$3; data=$4
python "$code/test.py" --checkpoint "$checkpoint" --output "$output" --data_root "$data" \
    --problem ALL --problem_sizes 50 100 200 --test_episodes 1000 --preferences 101 \
    --aug_factor 8 --test_batch_size 16 --gpu_id 0 --shard_index 0 --num_shards 2 \
    > "$output/shard-0.log" 2>&1 & p0=$!
python "$code/test.py" --checkpoint "$checkpoint" --output "$output" --data_root "$data" \
    --problem ALL --problem_sizes 50 100 200 --test_episodes 1000 --preferences 101 \
    --aug_factor 8 --test_batch_size 16 --gpu_id 1 --shard_index 1 --num_shards 2 \
    > "$output/shard-1.log" 2>&1 & p1=$!
trap '\''kill "$p0" "$p1" 2>/dev/null || true'\'' EXIT
wait "$p0"
wait "$p1"
trap - EXIT
' _ "$CODE" "$RUN_DIR/training/latest.pt" "$RUN_DIR/evaluation" "$MTMO_ROOT/MTMO/data"
STAGE=report
status running
python "$RUN_DIR/source/scripts/report_mtmo.py" --run_dir "$RUN_DIR"
STAGE=complete
status complete
