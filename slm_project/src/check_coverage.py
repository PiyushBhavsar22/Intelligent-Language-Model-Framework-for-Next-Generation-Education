import silence  # noqa: F401
import sqlite3
from collections import defaultdict
from pathlib import Path

from config import CONFIG

SUPPORTED = {".pdf", ".pptx", ".docx", ".txt", ".vtt", ".xlsx"}

# every supported file on disk, grouped by topic folder
on_disk: dict[str, set[str]] = defaultdict(set)
all_disk_files: set[str] = set()
for p in CONFIG.raw_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in SUPPORTED:
        rel = p.relative_to(CONFIG.raw_dir)
        topic = str(rel.parts[0]) + ("/" + rel.parts[1] if len(rel.parts) > 2 else "")
        on_disk[topic].add(p.name)
        all_disk_files.add(p.name)

# every filename actually present in the SQLite index
con = sqlite3.connect(str(CONFIG.db_path))
cur = con.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")]
source_table = source_col = None
for t in tables:
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})")]
    if "source" in cols:
        source_table, source_col = t, "source"
        break
if source_table is None:
    raise RuntimeError(f"No table with a 'source' column found among {tables} "
                       f"- share store.py's schema so this can be fixed.")

indexed_files = {r[0] for r in
                 cur.execute(f"SELECT DISTINCT {source_col} FROM {source_table}")}
con.close()

#3. report
missing = all_disk_files - indexed_files
print(f"Files on disk (supported types): {len(all_disk_files)}")
print(f"Files actually indexed:          {len(indexed_files)}")
print(f"NOT indexed:                     {len(missing)}\n")

if missing:
    print("Missing, by topic folder:")
    for topic, files in sorted(on_disk.items()):
        gap = files & missing
        if gap:
            print(f"  {topic}  ({len(gap)} missing)")
            for f in sorted(gap):
                print(f"    - {f}")
else:
    print("Every supported file on disk is represented in the index.")