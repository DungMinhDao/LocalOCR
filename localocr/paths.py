import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LOCALOCR_DATA") or ROOT / "data")  # uploads, jobs, templates
FILES_DIR = DATA / "files"
JOBS_DIR = DATA / "jobs"
STATIC = ROOT / "static"
WORKERS = ROOT / "workers"
BIN = ROOT / "bin"
NATIVE = ROOT / "native"

for _d in (FILES_DIR, JOBS_DIR, BIN):
    _d.mkdir(parents=True, exist_ok=True)
