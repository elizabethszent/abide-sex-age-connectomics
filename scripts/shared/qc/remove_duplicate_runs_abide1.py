# scripts/shared/qc/remove_duplicate_runs_abide1.py
import re
import argparse
import shutil
from pathlib import Path
from datetime import datetime

RX = re.compile(r"sub-(\d+)_task-rest_run-(\d+)\.npy$", re.IGNORECASE)

def parse_name(p: Path):
    m = RX.search(p.name)
    if not m:
        return None
    sub = int(m.group(1))
    run = int(m.group(2))
    return sub, run

def main():
    ap = argparse.ArgumentParser(description="Remove duplicate ABIDE1 connectomes across runs (keep run-1 if present).")
    ap.add_argument("--conn-dir", type=Path, required=True, help="Directory containing .npy connectomes")
    ap.add_argument("--meta-dir", type=Path, default=None, help="Optional meta dir with matching .json files")
    ap.add_argument("--keep-run", type=int, default=1, help="Preferred run number to keep if present (default: 1)")
    ap.add_argument("--mode", choices=["move", "delete"], default="move",
                    help="What to do with duplicates: move to trash dir (default) or delete")
    ap.add_argument("--trash-dir", type=Path, default=None,
                    help="Where to move duplicates (default: <conn-dir>/.duplicates_<timestamp>)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually perform changes. If not set, does a dry-run.")
    args = ap.parse_args()

    conn_dir: Path = args.conn_dir
    meta_dir: Path | None = args.meta_dir

    if not conn_dir.exists():
        raise FileNotFoundError(f"Connectome dir not found: {conn_dir}")

    files = sorted(conn_dir.glob("sub-*_task-rest_run-*.npy"))
    grouped: dict[int, list[tuple[int, Path]]] = {}

    for f in files:
        parsed = parse_name(f)
        if not parsed:
            continue
        sub, run = parsed
        grouped.setdefault(sub, []).append((run, f))

    # Decide which to remove
    to_remove_npy: list[Path] = []
    keep_map: dict[int, Path] = {}

    for sub, run_list in grouped.items():
        if len(run_list) <= 1:
            keep_map[sub] = run_list[0][1]
            continue

        run_list.sort(key=lambda x: x[0])  # sort by run number
        # keep preferred run if present, else lowest run
        keep = None
        for run, fp in run_list:
            if run == args.keep_run:
                keep = fp
                break
        if keep is None:
            keep = run_list[0][1]

        keep_map[sub] = keep
        for run, fp in run_list:
            if fp != keep:
                to_remove_npy.append(fp)

    # If meta dir supplied, match jsons by stem
    to_remove_json: list[Path] = []
    if meta_dir is not None:
        if not meta_dir.exists():
            raise FileNotFoundError(f"Meta dir not found: {meta_dir}")
        for npy in to_remove_npy:
            js = meta_dir / (npy.stem + ".json")
            if js.exists():
                to_remove_json.append(js)

    print(f"[INFO] Connectome dir: {conn_dir}")
    print(f"[INFO] Total .npy files found: {len(files)}")
    print(f"[INFO] Unique subjects found: {len(grouped)}")
    print(f"[INFO] Subjects with multiple runs: {sum(1 for v in grouped.values() if len(v) > 1)}")
    print(f"[INFO] Duplicate .npy files to remove: {len(to_remove_npy)}")
    if meta_dir is not None:
        print(f"[INFO] Matching .json files to remove: {len(to_remove_json)}")

    # Show a small preview
    preview = to_remove_npy[:20]
    if preview:
        print("\n[PREVIEW] First duplicates:")
        for p in preview:
            print("  ", p.name)
        if len(to_remove_npy) > len(preview):
            print(f"  ... ({len(to_remove_npy) - len(preview)} more)")

    if not args.apply:
        print("\n[DRY-RUN] No files changed. Re-run with --apply to perform the operation.")
        return

    # Perform operation
    if args.mode == "move":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_dir = args.trash_dir or (conn_dir / f".duplicates_{ts}")
        trash_dir.mkdir(parents=True, exist_ok=True)

        # if meta dir present, create a parallel trash folder
        meta_trash = None
        if meta_dir is not None:
            meta_trash = trash_dir / "_meta_json"
            meta_trash.mkdir(parents=True, exist_ok=True)

        for p in to_remove_npy:
            shutil.move(str(p), str(trash_dir / p.name))

        if meta_dir is not None and meta_trash is not None:
            for p in to_remove_json:
                shutil.move(str(p), str(meta_trash / p.name))

        print(f"\n[DONE] Moved duplicates to: {trash_dir}")

    else:  # delete
        for p in to_remove_npy:
            p.unlink(missing_ok=True)
        for p in to_remove_json:
            p.unlink(missing_ok=True)
        print("\n[DONE] Deleted duplicates.")

if __name__ == "__main__":
    main()