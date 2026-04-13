import numpy as np
from pathlib import Path

# updated path to point directly to where your new .npy files are stored
IN_DIR = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\connectomes\CC200\ABIDE1\FDpersubject2")
OUT_DIR = IN_DIR  # This will save the overall average in the same folder

FD_LIST = ["0.2"]

def load_mean(group: str, fd: str) -> np.ndarray:
    p = IN_DIR / f"{group}_fd-{fd}_mean_z.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return np.load(p)

for fd in FD_LIST:
    F_ASD = load_mean("F_ASD", fd)
    F_CTL = load_mean("F_CTL", fd)
    M_ASD = load_mean("M_ASD", fd)
    M_CTL = load_mean("M_CTL", fd)

    overall_z = 0.25 * (F_ASD + F_CTL + M_ASD + M_CTL)
    overall_r = np.tanh(overall_z)

    out_z = OUT_DIR / f"OVERALL_sexbalanced_fd-{fd}_mean_z.npy"
    out_r = OUT_DIR / f"OVERALL_sexbalanced_fd-{fd}_mean_r.npy"

    np.save(out_z, overall_z)
    np.save(out_r, overall_r)

    print(f"[DONE] fd={fd}: saved {out_z.name} and {out_r.name}")