# ptbxl-ecg-multilabel-dl

Multi-label detection of cardiovascular abnormalities from 10-second 12-lead ECG (PTB-XL v1.0.3):
1D CNN vs CNN+LSTM vs Transformer, with per-class statistical testing and zero-shot external validation
on the Chapman–Shaoxing / Ningbo cohort.

Code for the course article *"Development of a Deep Learning Model for Early Detection of Cardiovascular
Diseases Using 12-Lead ECG Signals"* (Research Methods and Tools, Astana IT University).

## Status

**Expected-results stage.** The multi-seed training runs are not finished. All model metrics, confidence
intervals, test statistics and error counts in the article are *expected* values projected from published
PTB-XL benchmarks and are labelled as such. This repository contains the planned pipeline; it does not yet
contain the final results. Dataset counts and class weights of PTB-XL v1.0.3 can be reproduced with
`python src/data.py <path-to-ptb-xl>`.

## Task

Five diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label, sigmoid outputs, class-weighted
binary cross-entropy. Official patient-disjoint split: folds 1–8 train, fold 9 validation (early stopping,
per-class thresholds), fold 10 test. 411 records without a diagnostic superclass are excluded
(21,799 → 21,388 records).

## Structure

```
src/data.py        PTB-XL loading, superclass labels, lead-wise normalisation, class weights
src/models.py      1D CNN, CNN+LSTM, Transformer encoder (python src/models.py prints parameter counts)
src/train.py       training for one model and one seed; saves validation/test probabilities
src/stats_eval.py  metrics, tuned thresholds, bootstrap CI, DeLong test, Benjamini–Hochberg correction
requirements.txt   planned software versions
```

## Data

Download PTB-XL v1.0.3 from PhysioNet (https://doi.org/10.13026/kfzx-aw45) and pass the folder with
`ptbxl_database.csv` to `--data`. Data and trained weights are not stored in this repository.

## Usage

```bash
pip install -r requirements.txt
python src/data.py /path/to/ptb-xl-1.0.3                       # record counts and class weights
for seed in 42 100 2024 3407 7777; do
  for model in cnn cnn_lstm transformer; do
    python src/train.py --data /path/to/ptb-xl-1.0.3 --model $model --seed $seed
  done
done
```

`src/stats_eval.py` is a library: load the saved `test_probs.npy` files, average over seeds with
`seed_ensemble`, then call `class_metrics`, `bootstrap_ci` (2,000 resamples) and `pairwise_delong`
(Benjamini–Hochberg over 15 tests).

## Planned setup

| Item | Value |
|---|---|
| Seeds | 42, 100, 2024, 3407, 7777 |
| Software | Python 3.11.9, PyTorch 2.3.1 (CUDA 12.1), see `requirements.txt` |
| Hardware | one NVIDIA Tesla T4 (16 GB), Google Colab |
| Expected time per seed at 100 Hz | about 6.5 min (CNN), 11.2 min (CNN+LSTM), 14.8 min (Transformer) |
| Determinism | `torch.backends.cudnn.deterministic = True` |

## License

MIT (see `LICENSE`). PTB-XL and the Chapman–Shaoxing / Ningbo database have their own licenses on PhysioNet.
