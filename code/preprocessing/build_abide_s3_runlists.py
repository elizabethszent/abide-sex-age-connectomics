#!/usr/bin/env python3
import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "fcp-indi"
BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"

DATASET_PREFIXES = {
    "ABIDE1": "data/Projects/ABIDE/Outputs/fmriprep/",
    "ABIDE2": "data/Projects/ABIDE2/Outputs/fmriprep/",
}

NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix: str):
    token = None
    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if token:
            params["continuation-token"] = token
        url = BASE_URL + "/?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        root = ET.fromstring(data)

        for contents in root.findall("s3:Contents", NS):
            key = contents.findtext("s3:Key", default="", namespaces=NS)
            if key:
                yield key

        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=NS)
        if truncated.lower() != "true":
            break
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=NS)
        if not token:
            break


def key_to_run_prefix(key: str):
    suffixes = [
        "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "_desc-preproc_bold.nii.gz",
    ]
    for sfx in suffixes:
        if key.endswith(sfx):
            return f"s3://{BUCKET}/" + key[: -len(sfx)]
    return None


def build_runlist(dataset: str, out_dir: Path):
    prefix = DATASET_PREFIXES[dataset]
    keys = list(list_keys(prefix))
    bold_keys = [k for k in keys if k.endswith("desc-preproc_bold.nii.gz")]
    run_prefixes = sorted({rp for k in bold_keys if (rp := key_to_run_prefix(k))})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset.lower()}_valid_run_prefixes.txt"
    out_file.write_text("\n".join(run_prefixes) + ("\n" if run_prefixes else ""))

    print(f"{dataset}: total keys scanned = {len(keys)}")
    print(f"{dataset}: bold files found = {len(bold_keys)}")
    print(f"{dataset}: run prefixes written = {len(run_prefixes)}")
    print(f"{dataset}: output = {out_file}")
    for sample in run_prefixes[:5]:
        print(f"  sample: {sample}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", choices=["ABIDE1", "ABIDE2", "both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    datasets = ["ABIDE1", "ABIDE2"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        build_runlist(ds, out_dir)


if __name__ == "__main__":
    main()
