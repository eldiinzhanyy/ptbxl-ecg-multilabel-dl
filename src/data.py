"""PTB-XL v1.0.3 loading: five diagnostic superclasses, official patient-disjoint folds."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(root):
    root = Path(root)
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]

    def to_superclasses(codes):
        found = {scp.loc[c, "diagnostic_class"] for c, lik in codes.items() if lik > 0 and c in scp.index}
        return [c in found for c in CLASSES]

    labels = np.array(df["scp_codes"].apply(to_superclasses).tolist(), dtype=np.float32)
    keep = labels.sum(1) > 0                            # drops the 411 records without a superclass
    return df[keep], labels[keep]


def load_signals(df, root, sr=100):
    col = "filename_lr" if sr == 100 else "filename_hr"
    x = [wfdb.rdsamp(str(Path(root) / f))[0].T for f in df[col]]   # (12, T) per record
    return np.stack(x).astype(np.float32)


def load_split(root, sr=100, cache=None):
    """Returns dict with x/y for train (folds 1-8), val (fold 9) and test (fold 10)."""
    cache = Path(cache) if cache else None
    if cache and (cache / f"x_{sr}.npy").exists():
        x, y, fold = (np.load(cache / f"{n}_{sr}.npy") for n in ("x", "y", "fold"))
    else:
        df, y = load_metadata(root)
        x, fold = load_signals(df, root, sr), df["strat_fold"].to_numpy()
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            for n, a in (("x", x), ("y", y), ("fold", fold)):
                np.save(cache / f"{n}_{sr}.npy", a)
    tr, va, te = fold <= 8, fold == 9, fold == 10
    mu = x[tr].mean(axis=(0, 2), keepdims=True)         # lead-wise statistics from training folds only
    sd = x[tr].std(axis=(0, 2), keepdims=True)
    x = (x - mu) / (sd + 1e-8)
    out = {k: (x[m], y[m]) for k, m in (("train", tr), ("val", va), ("test", te))}
    out["pos_weight"] = (tr.sum() - y[tr].sum(0)) / y[tr].sum(0)   # w_c = (N - N_c+) / N_c+
    return out


if __name__ == "__main__":
    import sys
    d = load_split(sys.argv[1], cache="cache")
    for k in ("train", "val", "test"):
        print(k, d[k][0].shape, d[k][1].sum(0))
    print("positive class weights", dict(zip(CLASSES, d["pos_weight"].round(3))))
