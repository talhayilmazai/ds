# Unified Turkish Sentiment Analysis Notebook — Design Spec

**Date:** 2026-05-09
**Author:** Talha Yılmaz (DS / DSAI 302 Capstone)
**Topic:** Merge the existing BASIC and ELECTRA sentiment-analysis notebooks into a single unified Colab notebook, and add a new from-scratch PyTorch Bi-LSTM model for an apples-to-apples comparison against the TF-IDF + LinearSVC baseline and the two pretrained Turkish transformers.

---

## 1. Goal

Produce one Jupyter notebook (`SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`) that:

1. Reuses the shared data-loading / preprocessing / EDA / split logic from the existing two notebooks **once**.
2. Trains and evaluates **four** models on the same train/val/test split:
   - TF-IDF + LinearSVC (lexical baseline)
   - BERTurk fine-tune (`dbmdz/bert-base-turkish-cased`)
   - ELECTRA fine-tune (`dbmdz/electra-base-turkish-cased-discriminator`)
   - **NEW**: Custom Embedding + Bidirectional LSTM, all weights random-initialised (the from-scratch deep-learning contribution)
3. Ends with a unified evaluation section: a single comparison DataFrame, a 2×2 grid of confusion matrices, and a Macro-F1 bar chart.
4. Runs end-to-end on a Google Colab T4 GPU instance.
5. Provides academic justification (in markdown cells next to the model class) for every non-trivial Bi-LSTM hyperparameter choice, suitable for the IEEE report.

## 2. Scope

**In scope.**
- One unified `.ipynb`.
- Bi-LSTM module: vocab builder, sequence encoder, `Dataset`, `DataLoader`, `nn.Module`, training loop, evaluation.
- Updates to the existing evaluation section to include all four models.
- Markdown cells justifying the Bi-LSTM hyperparameters.

**Out of scope.**
- Any change to the preprocessing pipeline (Zemberek / mojibake handling) beyond consolidating it.
- Pretrained Turkish word embeddings (e.g. fastText, Word2Vec). The user explicitly wants a **from-scratch** architecture; embeddings are random-initialised and learned during training.
- New model architectures beyond Bi-LSTM (no CNN, no GRU, no attention pooling).
- Hyperparameter search for the Bi-LSTM. Defaults are chosen from literature and justified inline; tuning is mentioned only in "Future work".

## 3. Notebook section layout

| § | Section | Source / Action |
|---|---------|-----------------|
| 0 | Environment bootstrap (Colab) | Lifted verbatim from BASIC. Installs Java + zemberek-python + transformers + datasets + accelerate + wordcloud + nltk + snowballstemmer. |
| 1.1 | Imports / seeds / matplotlib config | Lifted verbatim. Add `from torch import nn` and the `Counter` import for vocab. |
| 1.2 | Mojibake-safe CSV loading | Lifted verbatim. |
| 1.3 | Label cleaning & dedup filtering | Lifted verbatim. |
| 1.4 | Class-distribution EDA + plot | Lifted verbatim. |
| 1.5 | Length distribution EDA + plot | Lifted verbatim. |
| 1.6 | Turkish preprocessing (Zemberek + Snowball fallback) | Lifted verbatim. |
| 1.7 | Word clouds per class | Lifted verbatim. |
| 1.8 | Stratified 70/15/15 split | Lifted verbatim. Produces `X_train/X_val/X_test` (cleaned text) + `Xr_train/Xr_val/Xr_test` (raw text) + `y_train/y_val/y_test`. |
| 2.1 | Baseline — TF-IDF + LinearSVC | Lifted verbatim from BASIC. CV-tuned `C ∈ {0.1, 0.5, 1, 2, 5}`. |
| 2.2 | Transformer A — BERTurk fine-tune | Lifted verbatim from BASIC. `MAX_LEN=64`, `BATCH=32`, `EPOCHS=4`, `LR=2e-5`, `WD=0.01`, 10% warmup. |
| 2.3 | Transformer B — ELECTRA fine-tune | Same code path as 2.2 but with `dbmdz/electra-base-turkish-cased-discriminator`. Variables stored under separate names (`electra`, `electra_pred`, etc.) so 2.2's `bert` is not overwritten. |
| 2.4 | **NEW — Custom Bi-LSTM from scratch (PyTorch)** | See §4 below. |
| 3.1 | Unified evaluation function | Lifted from BASIC's `evaluate(name, y_true, y_pred)`. |
| 3.2 | Per-model test inference | Calls `evaluate` for SVM, BERTurk, ELECTRA, Bi-LSTM. Stores `svm_pred`, `bert_pred`, `electra_pred`, `lstm_pred`. |
| 3.3 | Side-by-side metrics DataFrame | 4 rows × 4 cols (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`). Saved as `results_metrics.csv`. |
| 3.4 | 2×2 confusion-matrix grid | Saved as `confusion_matrices.png` at 300 DPI. |
| 3.5 | Macro-F1 bar chart | Saved as `f1_macro_comparison.png` at 300 DPI. Convenient figure for the IEEE Results section. |
| 3.6 | Misclassification samples | Lifted from BASIC; loop over all four models. |
| 4 | Discussion & Conclusion stub | Lifted from BASIC, expanded to mention four-model contrast (lexical → from-scratch DL → MLM-pretrained → discriminative-pretrained). |

## 4. Bi-LSTM module — detailed design

### 4.1 Vocabulary building

A small `build_vocab(texts, min_freq=2)` helper. Inputs are the **already-preprocessed** (Zemberek-stemmed, lowercased, denoised) `X_train` strings — using the same input space as the SVM ensures the apples-to-apples comparison is "what does a deep architecture buy you over a sparse linear model on identical features?"

- Tokenization: whitespace `.split()`. Preprocessing already normalised everything.
- Reserved indices: `<PAD>=0`, `<UNK>=1`. These are added before any data-driven tokens so their indices are deterministic.
- Frequency filter: `min_freq=2` (drop hapaxes). Hapaxes contribute noise without generalising and inflate the embedding matrix.
- Returns `(stoi: dict[str, int], itos: list[str])`. Vocab fit on **train only** to avoid leakage.

### 4.2 Sequence encoding

A small `encode(texts, stoi, max_len=64)` helper:

- Map each token to `stoi.get(tok, 1)` (UNK fallback).
- Truncate to `max_len`.
- Right-pad with `0` (PAD) up to `max_len`.
- Return a `torch.long` tensor of shape `(N, max_len)`.

`MAX_LEN=64` matches the transformer choice (≈ 95th percentile of word counts in the dataset), keeping the comparison fair.

### 4.3 Dataset & DataLoader

```python
class TextSeqDataset(Dataset):
    def __init__(self, seqs: torch.Tensor, labels: np.ndarray):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]
```

DataLoaders: `batch_size=64`, `shuffle=True` for train, `shuffle=False` for val/test. Pinned memory not needed on Colab.

### 4.4 Model architecture

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 2,
                 lstm_dropout: float = 0.3, head_dropout: float = 0.5,
                 num_classes: int = 2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                    # (B, L, E)
        _, (h_n, _) = self.lstm(emb)               # h_n: (num_layers*2, B, H)
        # Last layer: forward = h_n[-2], backward = h_n[-1]; concat → (B, 2H)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(self.dropout(h))
```

### 4.5 Hyperparameter justifications (markdown cell content)

| Knob | Value | Rationale (for IEEE report) |
|------|-------|-----------------------------|
| `embed_dim` | 128 | Standard for vocab in the 5–10k range; 5k × 128 ≈ 640k params — large enough for semantic similarity, small enough to train from scratch on 7.7k examples without severe overfitting. |
| `hidden_size` | 128 | Symmetrical with embedding; bidirectional concat → 256-d sentence vector, comparable in capacity to a small transformer pooler. |
| `num_layers` | 2 | Layer-1 captures local morphology cues, layer-2 abstracts phrase-level sentiment. Three+ stacked LSTMs overfit on this dataset size. |
| `lstm_dropout` | 0.3 | Applied between stacked LSTM layers (PyTorch only applies `dropout` between layers, not within, with `num_layers≥2`). Mid-range value — well-supported by Zaremba et al. (2014) and standard sentiment-classification recipes. |
| `head_dropout` | 0.5 | Heavier dropout immediately before the linear classifier. Standard practice for small-data text classification — Goodfellow et al. (2016) recommend higher dropout near the head. |
| `padding_idx=0` | — | Pad embedding row stays at the zero vector; no gradient flows through pads. |
| Loss | `CrossEntropyLoss` (2-output head) | Functionally equivalent to BCEWithLogitsLoss for binary, but cleaner for the 4-way comparison and easier to extend to multi-class downstream. |
| Optimiser | `AdamW(lr=1e-3, weight_decay=1e-5)` | LR an order of magnitude higher than transformer fine-tunes (`2e-5`) because there are no pretrained weights to preserve; `1e-3` is the canonical "from-scratch text" LR. Light weight decay protects the embedding matrix without throttling capacity. |
| Epochs | 8 | Empirical cap. Best-val-F1 checkpoint is kept in memory, so over-training one epoch is harmless. |
| Batch size | 64 | T4 fits this comfortably for a 64-token sequence and ~640k-param model. |

### 4.6 Training loop

- Move model to `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- Per epoch: train pass (track running loss) → eval pass on val loader → compute Macro-F1 → save `deepcopy(state_dict)` if best so far.
- Print `epoch / train_loss / val_f1` per row.
- After loop: `model.load_state_dict(best_state)`.

### 4.7 Bi-LSTM test evaluation

- Run inference on test loader → `lstm_pred`.
- Pass through the shared `evaluate()` helper to print classification report and append metrics dict.

## 5. Reproducibility

- All seeds set in §1.1 (already in the existing notebooks): `random`, `np.random`, `torch.manual_seed`. Add `torch.cuda.manual_seed_all(RANDOM_STATE)` for the GPU side.
- `RANDOM_STATE = 42` everywhere. Train/val/test splits already use it.
- We do **not** enable `torch.use_deterministic_algorithms(True)` — strict CUDA determinism throttles cuDNN and the resulting metric noise (≈ ±0.005 F1) is well below the inter-model gap.

## 6. Artifacts written to disk

- `results_metrics.csv` — 4-row metrics table (overwrites the BASIC/ELECTRA versions).
- `confusion_matrices.png` — 2×2 grid, 300 DPI.
- `f1_macro_comparison.png` — bar chart, 300 DPI.
- `class_distribution.png`, `length_distribution.png`, `wordcloud_pos.png`, `wordcloud_neg.png` — already produced by EDA.

## 7. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Bi-LSTM overfits the 7.7k training set. | `min_freq=2` vocab cutoff, dropout 0.3 + 0.5, best-val-F1 checkpoint, only 8 epochs. |
| Transformer cells run twice (BERTurk + ELECTRA) → ~13 min total GPU time on Colab T4. | Acceptable for capstone; documented in the cell preamble so the user knows what they're committing to. |
| Variable-name collision between BERTurk and ELECTRA cells. | Distinct names (`bert`, `bert_pred`, `electra`, `electra_pred`). The `eval_loader` helper is shared but stateless. |
| Pad-token leakage if the LSTM uses ordinary mean-pool. | We use the **last hidden state** (`h_n`) of the last LSTM layer — this is computed step-by-step and doesn't average pads. The pad embedding is also zero-initialised and never updated thanks to `padding_idx=0`. |
| User runs the notebook without a GPU. | Three transformers + Bi-LSTM on CPU is impractical; cell preamble flags this and recommends Colab T4. |

## 8. Definition of done

- [ ] `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb` runs end-to-end on Colab T4.
- [ ] Final `results` DataFrame contains exactly four rows in this order: SVM, BERTurk, ELECTRA, Bi-LSTM.
- [ ] Confusion-matrix figure shows all four panels.
- [ ] Bi-LSTM achieves **Val Macro-F1 ≥ 0.80** (sanity bar — well below transformers, well above random; if we drop below this the architecture is broken, not just under-tuned).
- [ ] Every non-trivial Bi-LSTM hyperparameter has a justifying markdown cell.
- [ ] Old `sentiment_analysis-BASIC.ipynb` and `sentiment_analysis-ELECTRA.ipynb` are kept on disk untouched as backup references — the unified notebook is additive.

## 9. Future work (deferred — captured here so we don't get distracted)

- Replace random-init embeddings with pretrained Turkish fastText / Word2Vec.
- Add an attention-pooled variant of the Bi-LSTM (sum-pool over outputs weighted by `softmax(W h_t)`).
- LR finder + grid over `{lr, hidden_size, num_layers, dropout}`.
- Apply `nlpaug` synonym augmentation to enlarge the training set.
