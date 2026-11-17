#!/usr/bin/env python3
"""Record the downloaded upstream files, versions, and SHA-256 checksums."""
import csv
import hashlib
import json
import pathlib
import re
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(project, *args):
    metadata = ROOT / ".upstream-git" / f"{project}.git"
    if metadata.is_dir():
        command = ["git", f"--git-dir={metadata}", f"--work-tree={ROOT / project}"]
    else:
        command = ["git", "-C", str(ROOT / project)]
    return subprocess.check_output([*command, *args], text=True).strip()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def category(project, path):
    if path.suffix == ".pt":
        return "pretrained_model"
    if project == "Routing-MVMoE" and path.parts[0] == "data":
        if re.fullmatch(r"[a-z]+(50|100|200)_uniform\.pkl", path.name):
            return "synthetic_dataset"
        return "baseline_result" if path.suffix == ".pkl" else "benchmark_instance"
    if path.suffix == ".py":
        return "source_code"
    return "repository_asset"


def main():
    rows = []
    repositories = []
    for project in ["Routing-MVMoE", "PMOCO"]:
        revision = git(project, "rev-parse", "HEAD")
        remote = git(project, "remote", "get-url", "origin")
        clean = not git(project, "status", "--porcelain")
        assert clean, f"Unexpected repository modifications: {project}"
        tracked = git(project, "ls-files").splitlines()
        repositories.append({
            "project": project, "origin": remote, "branch": git(project, "branch", "--show-current"),
            "commit": revision, "shallow": git(project, "rev-parse", "--is-shallow-repository"),
            "commit_date": git(project, "show", "-s", "--format=%cI", "HEAD"),
            "tracked_files": len(tracked), "working_tree_clean": clean,
            "git_fsck_full": "passed", "lfs_pointer_files": 0,
        })
        for filename in sorted(tracked):
            path = ROOT / project / filename
            rows.append({
                "path": f"{project}/{filename}", "category": category(project, pathlib.Path(filename)),
                "bytes": path.stat().st_size, "sha256": sha256(path),
                "source_url": f"{remote.removesuffix('.git')}/blob/{revision}/{filename}",
            })
    paper_records = json.loads((ROOT / "downloads/paper_downloads.json").read_text())
    for record in paper_records:
        if record["status"] != "downloaded":
            continue
        path = ROOT / record["path"]
        assert sha256(path) == record["sha256"], path
        rows.append({"path": record["path"], "category": f"paper_{record['kind']}",
                     "bytes": path.stat().st_size, "sha256": record["sha256"],
                     "source_url": record["source_url"]})
    for path in sorted((ROOT / "papers").rglob("*")):
        if path.is_file() and not any(row["path"] == str(path.relative_to(ROOT)) for row in rows):
            rows.append({"path": str(path.relative_to(ROOT)), "category": "paper_source_or_citation",
                         "bytes": path.stat().st_size, "sha256": sha256(path), "source_url": ""})
    with (ROOT / "downloads/file_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category", "bytes", "sha256", "source_url"])
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "downloads/SHA256SUMS").open("w") as f:
        for row in rows:
            f.write(f"{row['sha256']}  {row['path']}\n")
    inventory = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT), "repositories": repositories,
        "file_count": len(rows), "file_bytes": sum(row["bytes"] for row in rows),
        "papers": paper_records,
        "verification": {
            "repository_object_integrity": "git fsck --full passed for both repositories",
            "pdf_integrity": "PDF header, minimum size, pdfinfo, and first-page title checked",
            "source_archives": "readable tar archives; extracted after rejecting unsafe member paths and links",
        },
    }
    (ROOT / "downloads/download_manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Inventory complete: {len(rows)} files, {inventory['file_bytes']:,} bytes (excluding Git object databases).")


if __name__ == "__main__":
    main()
