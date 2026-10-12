"""Check source snapshot and frozen runtime routines against recorded SHA-256 values."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
provenance = json.loads((root / "references/PROVENANCE.json").read_text(encoding="utf-8"))
records = provenance["files"] + provenance["runtime_vendor"]
errors = []
for record in records:
    path = root / record["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        errors.append(record["path"])
if errors:
    raise SystemExit("Snapshot mismatch:\n" + "\n".join(errors))
print(f"Verified {len(records)} reference files, including frozen runtime quota implementation.")
