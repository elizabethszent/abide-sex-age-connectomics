#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

def to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return None
    return None

def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[WARN] Could not parse JSON: {p} ({e})", file=sys.stderr)
        return None

def write_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)

def fix_obj(obj, *, rt_default=None, is_bold=False):
    if "AcquisitionDuration" in obj:
        val = to_float(obj.get("AcquisitionDuration"))
        if val is None:
            obj.pop("AcquisitionDuration", None)
        else:
            obj["AcquisitionDuration"] = val

    if is_bold:
        rt = obj.get("RepetitionTime", None)
        rt_val = to_float(rt) if rt is not None else None
        if rt_val is None and rt_default is not None:
            obj["RepetitionTime"] = float(rt_default)
        elif rt_val is not None:
            obj["RepetitionTime"] = float(rt_val)

    return obj

def main():
    if len(sys.argv) != 2:
        print("Usage: fix_bids_json.py <BIDS_SITE_DIR>", file=sys.stderr)
        sys.exit(2)

    site_dir = Path(sys.argv[1]).resolve()
    if not site_dir.exists():
        print(f"[ERR] Not found: {site_dir}", file=sys.stderr)
        sys.exit(1)

    template_path = site_dir / "task-rest_bold.json"
    template = load_json(template_path) if template_path.exists() else None

    rt_default = None
    if template is not None:
        template = fix_obj(template, rt_default=None, is_bold=True)
        rt_default = to_float(template.get("RepetitionTime"))
        write_json(template_path, template)

    for jp in site_dir.rglob("*.json"):
        obj = load_json(jp)
        if obj is None:
            continue
        is_bold = (jp.name.endswith("_bold.json") or jp.name == "task-rest_bold.json")
        obj = fix_obj(obj, rt_default=rt_default, is_bold=is_bold)
        write_json(jp, obj)

    if template is not None:
        created = 0
        for bold in site_dir.rglob("*_bold.nii.gz"):
            base = bold.name[:-7]
            sidecar = bold.with_name(base + ".json")
            if not sidecar.exists():
                obj = dict(template)
                obj = fix_obj(obj, rt_default=rt_default, is_bold=True)
                write_json(sidecar, obj)
                created += 1
        print(f"[INFO] Created {created} missing bold sidecars from template.")
    else:
        print("[WARN] No task-rest_bold.json template found; cannot create missing sidecars.", file=sys.stderr)

if __name__ == "__main__":
    main()
