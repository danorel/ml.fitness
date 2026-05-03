# Makemore — Experiment Results

Character-level language models trained on `names.txt` (first names).  
Vocabulary: 27 tokens — `a–z` + `.` (separator/boundary token).

---

## 1. Bigram (count-based)

**Approach:** raw co-occurrence counts → row-normalized probability table. No gradient descent.

**Architecture:**

| Component | Detail |
|-----------|--------|
| Matrix | `P[27, 27]` — bigram transition probabilities |
| Parameters | 0 (no learned weights) |

**Evaluation (full dataset, per-character NLL):**

| Metric | Value |
|--------|-------|
| NLL | **2.4541** |

**Sample names** (seed `1776892230`):

```
can, joinune, sheema, ste, aiselyaryded, tell, dheelt, tarensyanel, kaylolsha, aha
```

---

## 2. Bigram FFN Equivalence

**Approach:** single linear layer on one-hot input — mathematically equivalent to the count-based bigram, but trained via gradient descent.

**Architecture:**

| Component | Shape | Detail |
|-----------|-------|--------|
| Input | `(N, 27)` | one-hot encoded character |
| `W1` | `(27, 27)` | weight matrix |
| `b1` | `(27,)` | bias |
| Output | `(N, 27)` | logits → cross-entropy |
| **Total params** | **756** | |

**Training:**

| Hyperparameter | Value |
|----------------|-------|
| Loss | Cross-entropy |
| Optimizer | SGD |
| Learning rate | 0.1 (fixed) |
| Epochs | 5000 |

**Training curve:**

| Epoch | Loss |
|-------|------|
| 0 | 4.2209 |
| 100 | 3.5889 |
| 200 | 3.3313 |
| 300 | 3.1959 |
| 400 | 3.1099 |
| 500 | 3.0496 |
| 600 | 3.0035 |
| 700 | 2.9659 |
| 800 | 2.9341 |
| 900 | 2.9066 |
| 1000 | 2.8823 |
| 1500 | 2.7921 |
| 2000 | 2.7330 |
| 2500 | 2.6912 |
| 3000 | 2.6598 |
| 3500 | 2.6355 |
| 4000 | 2.6161 |
| 4500 | 2.6003 |
| 4900 | 2.5896 |

**Evaluation (per-character NLL on full dataset):**

| Metric | Value |
|--------|-------|
| NLL | **2.5868** |

**Sample names** (continued from seed `1776892230`):

```
keh, tlieudhsoleqa, avs, in, scyo, krirpi, m, lyilriniyriae, amalasrahr, aiar
```

> FFN converges toward the count-based bigram NLL (2.4541) with more epochs; 5000 epochs at LR=0.1 not yet fully converged — loss still declining slowly.

---

## 3. MLP (Bengio et al. 2003)

**Approach:** embedding lookup + one hidden tanh layer + output projection. Uses trigram context (3 previous characters).

**Architecture:**

| Component | Shape | Detail |
|-----------|-------|--------|
| Embedding `C` | `(27, 10)` | character embeddings |
| `W1` | `(30, 200)` | `SEQ_SIZE × EMB_SIZE → HID_SIZE` |
| `b1` | `(200,)` | |
| `W2` | `(200, 27)` | hidden → logits |
| `b2` | `(27,)` | |
| **Total params** | **11,897** | |

**Hyperparameters:**

| Hyperparameter | Value |
|----------------|-------|
| `SEQ_SIZE` (context) | 3 |
| `VOC_SIZE` | 27 |
| `EMB_SIZE` | 10 |
| `HID_SIZE` | 200 |
| `BCH_SIZE` (mini-batch) | 128 |
| Loss | Cross-entropy |
| Optimizer | SGD |
| Learning rate schedule | Linear decay `1.0 → 0.0` over epochs |
| Epochs | 5000 |
| Random seed | `1777213731` |

**Dataset split:**

| Split | Samples | Fraction |
|-------|---------|----------|
| Train | 182,516 | 80% |
| Validation | 22,814 | 10% |
| Test | 22,816 | 10% |
| **Total** | **228,146** | |

**Training curve (train loss, mini-batch):**

| Epoch | Train Loss |
|-------|-----------|
| 0 | 26.7138 |
| 500 | 4.6508 |
| 1000 | 5.3671 |
| 1500 | 3.4488 |
| 2000 | 4.1943 |
| 2500 | 3.0242 |
| 3000 | 2.5494 |
| 3500 | 2.5927 |
| 4000 | 2.0735 |
| 4500 | 2.0485 |

**Final evaluation:**

| Split | Loss (NLL) |
|-------|-----------|
| Test set (held-out, 22,816 samples) | **2.5411** |
| Full dataset (per-character NLL) | **2.4042** |

**Sample names** (seed `1777213731`):

```
kaxlee, kharo, tere, tegh, ever, lole, sainah, raeristephyri, syn, eca
```

---

## 4. MLP + BatchNorm + Kaiming Init

**Approach:** same MLP architecture, with two improvements: (1) batch normalization before `tanh` activation stabilizes hidden pre-activations; (2) Kaiming initialization (`(5/3) / sqrt(fan_in)`) on `W1` controls gradient flow through `tanh`. Bias `b1` removed — batch norm's `bn_bias` subsumes it.

**Architecture:**

| Component | Shape | Init | Detail |
|-----------|-------|------|--------|
| Embedding `C` | `(27, 10)` | `1/sqrt(27)` | character embeddings |
| `W1` | `(30, 200)` | Kaiming `(5/3)/sqrt(30)` | pre-BN projection |
| `bn_gain` | `(1, 200)` | ones | BN scale |
| `bn_bias` | `(1, 200)` | zeros | BN shift (replaces `b1`) |
| `W2` | `(200, 27)` | `1/sqrt(200×27)` | hidden → logits |
| `b2` | `(27,)` | zeros | output bias |
| **Total params** | **12,097** | | `+200` vs baseline (bn_gain, bn_bias replace b1) |

**Hyperparameters:**

| Hyperparameter | Value |
|----------------|-------|
| `SEQ_SIZE` (context) | 3 |
| `VOC_SIZE` | 27 |
| `EMB_SIZE` | 10 |
| `HID_SIZE` | 200 |
| `BCH_SIZE` (mini-batch) | 128 |
| Loss | Cross-entropy |
| Optimizer | SGD |
| Learning rate schedule | Linear decay `1.0 → 0.0` over epochs |
| Epochs | 5,000 |
| Random seed | `1777213731` |
| BN momentum | `1e-3` |
| BN epsilon | `1e-6` |

**Training curve (train loss, mini-batch):**

| Epoch | Train Loss |
|-------|-----------|
| 1 | 3.2743 |
| 500 | 2.3764 |
| 1000 | 2.3867 |
| 1500 | 2.1884 |
| 2000 | 2.1499 |
| 2500 | 2.1358 |
| 3000 | 2.2544 |
| 3500 | 2.1511 |
| 4000 | 2.3286 |
| 4500 | 2.2691 |
| 5000 | 2.3091 |

**Final evaluation:**

| Split | Loss (NLL) |
|-------|-----------|
| Test set (held-out, 22,816 samples) | **2.4036** |
| Full dataset (per-name avg NLL) | **2.1925** |

---

## Summary

| Model | NLL (full eval) | NLL (test) | Params | Notes |
|-------|-----------------|------------|--------|-------|
| Bigram (count-based) | 2.4541 | — | 0 | analytical, no training |
| Bigram FFN | 2.5868 | — | 756 | 5000 epochs, LR=0.1, still converging |
| MLP (trigram context) | 2.4042 | 2.5411 | 11,897 | 5000 epochs, LR 1→0 |
| MLP + BN + Kaiming | **2.1925** | 2.4036 | 12,097 | 5,000 epochs, LR 1→0, BN before tanh |

BatchNorm + Kaiming init improves full-dataset NLL (2.1925 vs 2.4042) at same epoch budget, confirming proper initialization and activation normalization accelerate convergence. Test NLL (2.4036) nearly matches baseline MLP (2.5411) — better generalization too.
