import csv
import os
import requests
from tqdm import tqdm

PHENO_PATH = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\Phenotypic_V1_0b_preprocessed1.csv" 
OUT_DIR    = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\roi_timeseries"
PIPELINE   = "cpac"
STRATEGY   = "nofilt_noglobal" 
ATLAS      = "cc200"             

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"

derivative = f"rois_{ATLAS}"   
suffix     = f"rois_{ATLAS}.1D"  

os.makedirs(os.path.join(OUT_DIR, PIPELINE, STRATEGY, derivative), exist_ok=True)

def download_one(file_id):
    url = f"{BASE}/{PIPELINE}/{STRATEGY}/{derivative}/{file_id}_{suffix}"
    out_path = os.path.join(OUT_DIR, PIPELINE, STRATEGY, derivative, f"{file_id}_{suffix}")
    if os.path.exists(out_path):
        return True, "exists"

    r = requests.get(url, timeout=30)
    if r.status_code == 200 and r.text and not r.text.startswith("<Error>"):
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True, "ok"
    else:
        return False, f"http {r.status_code}"

def load_file_ids(pheno_csv):
    ids = []
    with open(pheno_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row.get("FILE_ID", "").strip()
            sex = row.get("SEX", "").strip()# 1=Male, 2=Female
            if fid:
                ids.append(fid)
    return ids

if __name__ == "__main__":
    file_ids = load_file_ids(PHENO_PATH)
    print(f"Found {len(file_ids)} FILE_IDs")

    ok, miss = 0, 0
    for fid in tqdm(file_ids):
        success, msg = download_one(fid)
        if success:
            ok += 1
        else:
            miss += 1
    print(f"Done. Downloaded {ok}, missing {miss}.")
    print(f"Output in: {os.path.join(OUT_DIR, PIPELINE, STRATEGY, derivative)}")
