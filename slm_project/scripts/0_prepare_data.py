"""
Step 0 (optional helper): Unzip all MOOC_*.zip archives into data/raw/.

Put your MOOC_1.zip, MOOC_2.zip, ... into the project's data/raw/ folder
(or pass a folder of zips), then run:

    python scripts/0_prepare_data.py                # unzip everything in data/raw
    python scripts/0_prepare_data.py /path/to/zips  # unzip from another folder

Images and spreadsheets are left in place; the extractor simply ignores them,
so you do NOT need to delete anything.
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR
    zips = sorted(src.glob("MOOC_*.zip")) + sorted(src.glob("MOOC*.zip"))
    zips = sorted(set(zips))
    if not zips:
        print(f"No MOOC_*.zip files found in {src}")
        print("Place your zip files there (or pass a folder path as an argument).")
        return

    for z in zips:
        print(f"Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
    print(f"Done. Extracted {len(zips)} archive(s) into {RAW_DIR}")
    # show what top-level module folders now exist
    mods = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    print("Module folders present:", mods)


if __name__ == "__main__":
    main()
