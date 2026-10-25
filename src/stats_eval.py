"""Metrics, validation-tuned thresholds, bootstrap CIs, DeLong tests, Benjamini-Hochberg correction."""
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def tune_thresholds(y_val, p_val, grid=np.linspace(0.05, 0.95, 91)):
    """Per-class threshold maximising F1 on the validation fold."""
    return np.array([grid[np.argmax([f1_score(y_val[:, c], p_val[:, c] >= t, zero_division=0) for t in grid])]
                     for c in range(y_val.shape[1])])


def class_metrics(y, p, thr):
    rows = {}
    for c, name in enumerate(CLASSES):
        pred = p[:, c] >= thr[c]
        rows[name] = dict(auroc=roc_auc_score(y[:, c], p[:, c]),
                          f1=f1_score(y[:, c], pred, zero_division=0),
                          recall=recall_score(y[:, c], pred, zero_division=0),
                          precision=precision_score(y[:, c], pred, zero_division=0))
    rows["macro"] = {k: float(np.mean([rows[n][k] for n in CLASSES])) for k in rows["NORM"]}
    return rows


def bootstrap_ci(y, p, thr, metric, cls=None, B=2000, seed=42):
    """Percentile bootstrap 95% CI (resampling recordings). cls=None -> macro average."""
    rng, n, vals = np.random.default_rng(seed), len(y), []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            m = class_metrics(y[idx], p[idx], thr)
        except ValueError:                                 # a resample without positives for some class
            continue
        vals.append(m["macro" if cls is None else CLASSES[cls]][metric])
    return np.percentile(vals, [2.5, 97.5])


def _midrank(x):
    order = np.argsort(x)
    z, ranks, i = x[order], np.zeros(len(x)), 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs (fast algorithm of Sun & Xu, 2014)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(p)
        aucs.append(tz[pos].sum() / m / n - (m + 1) / 2 / n)
        v10.append((tz[pos] - tx) / n)
        v01.append(1 - (tz[neg] - ty) / m)
    s = np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0] - aucs[1], z, 2 * stats.norm.sf(abs(z))


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * len(p) / rank)
        adj[i] = prev
    return adj


def pairwise_delong(y, probs, pairs):
    """probs: {model: (N, 5) probabilities}; pairs: [(A, B), ...]. Returns rows incl. BH-adjusted p."""
    rows = [(a, b, CLASSES[c], *delong_test(y[:, c], probs[a][:, c], probs[b][:, c]))
            for a, b in pairs for c in range(len(CLASSES))]
    adj = benjamini_hochberg([r[-1] for r in rows])
    return [dict(a=r[0], b=r[1], cls=r[2], dAUROC=r[3], Z=r[4], p=r[5], p_adj=pa) for r, pa in zip(rows, adj)]


def seed_ensemble(prob_files):
    return np.mean([np.load(f) for f in prob_files], axis=0)
