"""Train one model with one seed. Saves validation / test probabilities for src/stats_eval.py.

Example:  python src/train.py --data /path/to/ptb-xl-1.0.3 --model cnn_lstm --seed 42
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from data import load_split
from models import build

SEEDS = [42, 100, 2024, 3407, 7777]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, x, device, bs=256):
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(x[i:i + bs]).to(device))).cpu().numpy()
           for i in range(0, len(x), bs)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["cnn", "cnn_lstm", "transformer"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sr", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_split(a.data, a.sr, cache="cache")
    (xtr, ytr), (xva, yva), (xte, _) = d["train"], d["val"], d["test"]
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                        batch_size=64, shuffle=True)

    model = build(a.model).to(dev)
    lr = 5e-4 if a.model == "transformer" else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    if a.model == "transformer":                          # 5-epoch linear warm-up
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[5])
    else:
        sched = cosine
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(d["pos_weight"], dtype=torch.float32, device=dev))

    best, bad = -1.0, 0
    out = Path(a.out) / f"{a.model}_sr{a.sr}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(dev)), yb.to(dev)).backward()
            opt.step()
        sched.step()
        auc = roc_auc_score(yva, predict(model, xva, dev), average="macro")
        print(f"epoch {ep + 1:02d}  val macro-AUROC {auc:.4f}", flush=True)
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    np.save(out / "val_probs.npy", predict(model, xva, dev))
    np.save(out / "test_probs.npy", predict(model, xte, dev))
    print("done:", out)


if __name__ == "__main__":
    main()
