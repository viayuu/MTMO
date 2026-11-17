#!/usr/bin/env python3
"""Download the two projects' papers from official publication services."""
import hashlib
import json
import pathlib
import subprocess
import tarfile
import time
from datetime import datetime, timezone

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "downloads/paper_downloads.json"
session = requests.Session()
session.headers["User-Agent"] = "MTMO-research-download/1.0"
records = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []


def save():
    MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def download(path, urls, kind):
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Already exists: {path}", flush=True)
        return
    errors = []
    for url in urls:
        for attempt in range(2):
            part = out.with_name(out.name + ".part")
            try:
                print(f"Downloading {url} -> {path}", flush=True)
                with session.get(url, stream=True, timeout=(20, 75)) as r:
                    r.raise_for_status()
                    final_url = r.url
                    content_type = r.headers.get("Content-Type", "")
                    with part.open("wb") as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                if kind == "pdf":
                    with part.open("rb") as f:
                        assert f.read(5) == b"%PDF-", "not a PDF"
                    assert part.stat().st_size > 10240, "PDF smaller than 10 KiB"
                    subprocess.run(["pdfinfo", str(part)], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                elif kind == "source":
                    assert tarfile.is_tarfile(part), "not a readable source tar archive"
                part.rename(out)
                rec = {
                    "path": path, "status": "downloaded", "kind": kind,
                    "source_url": url, "final_url": final_url,
                    "content_type": content_type, "bytes": out.stat().st_size,
                    "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
                records[:] = [x for x in records if x["path"] != path]
                records.append(rec)
                save()
                print(f"OK: {path} ({rec['bytes']:,} bytes)", flush=True)
                time.sleep(1)
                return
            except Exception as exc:
                error = f"{url}: {type(exc).__name__}: {exc}"
                errors.append(error)
                print(error, flush=True)
                part.unlink(missing_ok=True)
                time.sleep(5 if "429" in str(exc) else 1)
    records[:] = [x for x in records if x["path"] != path]
    records.append({"path": path, "status": "failed", "kind": kind,
                    "attempted_urls": urls, "errors": errors})
    save()


jobs = [
    ("papers/MVMoE/MVMoE_ICML2024.pdf", [
        "https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhou24c/zhou24c.pdf",
        "https://proceedings.mlr.press/v235/zhou24c/zhou24c.pdf",
    ], "pdf"),
    ("papers/MVMoE/MVMoE_arxiv_2405.01029v2.pdf", [
        "https://arxiv.org/pdf/2405.01029v2",
        "https://export.arxiv.org/pdf/2405.01029v2",
    ], "pdf"),
    ("papers/PMOCO/PMOCO_arxiv_2203.15386v2.pdf", [
        "https://arxiv.org/pdf/2203.15386v2",
        "https://export.arxiv.org/pdf/2203.15386v2",
    ], "pdf"),
    ("papers/PMOCO/PMOCO_ICLR2022.pdf", [
        "https://openreview.net/pdf?id=QuObT9BTWo",
    ], "pdf"),
    ("papers/MVMoE/MVMoE_arxiv_2405.01029v2_source.tar.gz", [
        "https://arxiv.org/src/2405.01029v2",
        "https://export.arxiv.org/e-print/2405.01029v2",
    ], "source"),
    ("papers/PMOCO/PMOCO_arxiv_2203.15386v2_source.tar.gz", [
        "https://arxiv.org/src/2203.15386v2",
        "https://export.arxiv.org/e-print/2203.15386v2",
    ], "source"),
]

if __name__ == "__main__":
    (ROOT / "downloads/logs").mkdir(parents=True, exist_ok=True)
    for path, urls, kind in jobs:
        download(path, urls, kind)
    print("Paper download pass complete.", flush=True)
