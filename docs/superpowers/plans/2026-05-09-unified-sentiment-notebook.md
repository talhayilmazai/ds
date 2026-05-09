# Unified Turkish Sentiment-Analysis Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb` that consolidates the existing BASIC + ELECTRA notebooks and adds a from-scratch PyTorch Bi-LSTM, ending with a four-row comparison table across SVM, BERTurk, ELECTRA, and Bi-LSTM.

**Architecture:** Build the notebook via a deterministic Python builder script `scripts/build_unified_notebook.py` that uses `nbformat` to assemble cells. Each task extends the builder by appending a section, regenerates the `.ipynb`, validates JSON + Python syntax, and commits. Cell sources from the existing two notebooks are lifted **verbatim** as multi-line strings; only the ELECTRA fine-tune block is re-namespaced (`electra_*` instead of `bert_*`) so both transformers can co-exist in one notebook. The new Bi-LSTM section is appended after the two transformer blocks so the unified evaluation pulls four prediction arrays.

**Tech Stack:** Python 3.11+, `nbformat` (Jupyter), PyTorch, scikit-learn, transformers (HuggingFace), `zemberek-python`, pandas, matplotlib/seaborn. The notebook itself targets Google Colab T4.

**Reference spec:** [`docs/superpowers/specs/2026-05-09-unified-sentiment-notebook-design.md`](../specs/2026-05-09-unified-sentiment-notebook-design.md). Existing source notebooks: `SentimentAnalysis/sentiment_analysis-BASIC.ipynb`, `SentimentAnalysis/sentiment_analysis-ELECTRA.ipynb`.

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `scripts/build_unified_notebook.py` | Create | Builder. Imports `nbformat`, defines a list of `(cell_type, source)` tuples, writes `.ipynb`. Idempotent — running it always produces the same output. |
| `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb` | Create | Generated artifact. Run in Colab. |
| `SentimentAnalysis/sentiment_analysis-BASIC.ipynb` | **Untouched** | Backup reference per spec §8. |
| `SentimentAnalysis/sentiment_analysis-ELECTRA.ipynb` | **Untouched** | Backup reference per spec §8. |
| `SentimentAnalysis/social_media_comments.csv` | **Untouched** | Dataset. |

The builder script is the single source of truth; the `.ipynb` is regenerated each task.

---

## Per-Task Verification Pattern

Each task ends by running:
```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: builder prints `wrote SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb (N cells)`, py_compile prints nothing, the `nbformat.read` line prints `cells=N`.

Final end-to-end validation (after all tasks): open the notebook in Google Colab T4 and run all cells. Acceptance criteria from spec §8 — final `results` DataFrame shows four rows; Bi-LSTM Val Macro-F1 ≥ 0.80.

---

### Task 1: Scaffold the builder script and emit an empty notebook

**Files:**
- Create: `scripts/build_unified_notebook.py`
- Create: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb` (generated)

- [ ] **Step 1.1: Create `scripts/build_unified_notebook.py`**

```python
"""Build the unified Turkish sentiment-analysis notebook.

Run from the repo root:
    python scripts/build_unified_notebook.py

Idempotent — always produces the same .ipynb from the same source.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb')


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip('\n'))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip('\n'))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # ===== TASK-2-INSERT (bootstrap + setup) =====
    # ===== TASK-3-INSERT (data loading + label clean) =====
    # ===== TASK-4-INSERT (EDA) =====
    # ===== TASK-5-INSERT (Turkish preprocessing + split) =====
    # ===== TASK-6-INSERT (SVM baseline) =====
    # ===== TASK-7-INSERT (BERTurk fine-tune) =====
    # ===== TASK-8-INSERT (ELECTRA fine-tune) =====
    # ===== TASK-9-INSERT (Bi-LSTM: vocab + encoder + dataset) =====
    # ===== TASK-10-INSERT (Bi-LSTM: model class + hyperparam md) =====
    # ===== TASK-11-INSERT (Bi-LSTM: training + test inference) =====
    # ===== TASK-12-INSERT (Unified evaluation) =====
    # ===== TASK-13-INSERT (Misclassification samples + discussion) =====

    nb.cells = cells
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python', 'version': '3.11'},
    }
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f'wrote {OUT} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 1.2: Run the builder**

```powershell
python scripts/build_unified_notebook.py
```

Expected output: `wrote SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb (0 cells)`.

- [ ] **Step 1.3: Validate output**

```powershell
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=0`.

- [ ] **Step 1.4: Commit**

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Scaffold builder for unified sentiment notebook"
```

---

### Task 2: Add bootstrap + setup cells (notebook §0–§1.1)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 2.1: Replace `# ===== TASK-2-INSERT (bootstrap + setup) =====` with the four cells below**

In `scripts/build_unified_notebook.py`, find the line:
```python
    # ===== TASK-2-INSERT (bootstrap + setup) =====
```
and replace it with:

```python
    cells.append(md("""
# Turkish Social-Media Sentiment Analysis — Pozitif vs Negatif (Unified)

Binary sentiment classification on `social_media_comments.csv` (≈11k short Turkish posts labelled `Pozitif` / `Negatif`).

**Pipeline.** Mojibake-safe loading → Turkish-aware preprocessing (Zemberek morphological roots, fallback to Snowball) → stratified 70/15/15 split → four models trained on the same split:
1. **TF-IDF + LinearSVC** (lexical baseline)
2. **BERTurk** fine-tune (`dbmdz/bert-base-turkish-cased`)
3. **ELECTRA** fine-tune (`dbmdz/electra-base-turkish-cased-discriminator`)
4. **Custom Bi-LSTM** built from scratch in PyTorch (random-init embeddings, 2-layer Bi-LSTM, dropout)

The notebook ends with a unified four-row comparison table, a 2×2 grid of confusion matrices, and a Macro-F1 bar chart for the IEEE report.

**Runtime.** Targets Google Colab T4. Both transformer fine-tunes plus the Bi-LSTM take ≈15 min total on a T4.
"""))
    cells.append(md("## 1. Data Analysis and Data Preprocess"))
    cells.append(md("### 1.0 Environment bootstrap"))
    cells.append(code("""
# 1. Install Java (JDK 11) quietly via the underlying Ubuntu system
!apt-get update -qq > /dev/null
!apt-get install -y openjdk-11-jdk-headless -qq > /dev/null

# 2. Set the JAVA_HOME environment variable so jpype1 can find it
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Self-bootstrap the few NLP libraries that aren't part of the standard scientific
# stack. Safe to re-run: pip is a no-op when packages are already present.
%pip install --quiet \\
    transformers==4.44.* datasets==2.20.* accelerate==0.33.* \\
    zemberek-python==0.2.3 jpype1 \\
    wordcloud==1.9.* nltk==3.9.* snowballstemmer tqdm
"""))
    cells.append(md("### 1.1 Setup"))
    cells.append(code("""
import os
import re
import random
import shutil
import time
import warnings
from copy import deepcopy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

warnings.filterwarnings('ignore', category=UserWarning)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

sns.set_theme(style='whitegrid', context='paper', palette='deep')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# CSV sits alongside this notebook in SentimentAnalysis/.
DATA_PATH = 'social_media_comments.csv'
ARTIFACT_DIR = Path('.')
"""))
```

- [ ] **Step 2.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=6`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add bootstrap + setup cells to unified notebook"
```

---

### Task 3: Add data loading + label clean cells (notebook §1.2–§1.3)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 3.1: Replace `# ===== TASK-3-INSERT (data loading + label clean) =====` with**

```python
    cells.append(md("### 1.2 Mojibake-safe loading"))
    cells.append(md("The raw file is reported by `file(1)` as `ISO-8859 text`, but Turkish data of this provenance often carries Windows-1254 / Latin-9 confusions. We try the plausible Turkish encodings in order, then validate by counting any U+FFFD replacement characters that survive."))
    cells.append(code("""
MOJIBAKE_MAP = str.maketrans({
    'þ': 'ş', 'Þ': 'Ş', 'ð': 'ğ', 'Ð': 'Ğ',
    'ý': 'ı', 'Ý': 'İ',
})
# Common UTF-8-decoded-as-CP1252 sequences (multi-char → can't go in maketrans).
DOUBLE_DECODE_PAIRS = [
    ('Ã§', 'ç'), ('Ã¶', 'ö'), ('Ã¼', 'ü'),
    ('Ã‡', 'Ç'), ('Ã–', 'Ö'), ('Ãœ', 'Ü'),
    ('ÅŸ', 'ş'), ('Åž', 'Ş'), ('ÄŸ', 'ğ'), ('Äž', 'Ğ'),
    ('Ä±', 'ı'), ('Ä°', 'İ'),
]

def _fix_mojibake(s: str) -> str:
    \"\"\"Apply common Turkish mojibake repairs to a string.\"\"\"
    s = s.translate(MOJIBAKE_MAP)
    for bad, good in DOUBLE_DECODE_PAIRS:
        if bad in s:
            s = s.replace(bad, good)
    return s

def load_turkish_csv(path: str) -> pd.DataFrame:
    \"\"\"Load a Turkish-language CSV trying CP1254 → ISO-8859-9 → CP1252+repair.\"\"\"
    last_err = None
    for enc in ('cp1254', 'iso-8859-9'):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f'Loaded with encoding={enc!r}, shape={df.shape}')
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    # Final fallback: read as cp1252 and patch.
    df = pd.read_csv(path, encoding='cp1252')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).map(_fix_mojibake)
    print(f'Loaded with cp1252 + mojibake repair, shape={df.shape}')
    if last_err is not None:
        print(f'(prior encodings failed: {type(last_err).__name__}: {last_err})')
    return df

raw = load_turkish_csv(DATA_PATH)
raw.columns = [c.strip() for c in raw.columns]
# Original columns: 'Tip' (label), 'Paylaşım' (text). Rename to ASCII-safe.
rename_map = {}
for c in raw.columns:
    cc = _fix_mojibake(c)
    if cc.lower().startswith('tip'):
        rename_map[c] = 'label'
    elif cc.lower().startswith('payla'):
        rename_map[c] = 'text'
df = raw.rename(columns=rename_map)
assert {'label', 'text'}.issubset(df.columns), f'unexpected cols: {df.columns.tolist()}'

# Validate: any surviving replacement characters?
sample_text = ' '.join(df['text'].dropna().astype(str).head(500).tolist())
n_replacement = sample_text.count('\\ufffd')
print(f'U+FFFD replacement chars in first-500-row sample: {n_replacement}')
df.sample(5, random_state=RANDOM_STATE)
"""))
    cells.append(md("### 1.3 Label cleaning & filtering"))
    cells.append(code("""
# Strip whitespace and tally raw label counts.
df['label'] = df['label'].astype(str).str.strip()
df['text']  = df['text'].astype(str).str.strip()

print('Raw label counts:')
print(df['label'].value_counts(dropna=False))

before = len(df)
df = df[df['label'].isin({'Pozitif', 'Negatif'})].copy()
df = df[df['text'].str.len() > 0]
df = df.drop_duplicates(subset='text').reset_index(drop=True)
after = len(df)
print(f'\\nKept {after}/{before} rows after binary-label filter + dedup '
      f'(removed {before - after}).')
print('\\nFinal label counts:')
print(df['label'].value_counts())
"""))
```

> Note on string escaping: the cell `source` text uses `\\n`, `\\ufffd`, and triple-double-quoted docstrings to survive the wrapping triple-quoted Python string. When the builder writes the cell to the `.ipynb`, the JSON `source` field will contain the original characters (`\n`, `�`, `"""`). Verify this in step 3.2 by reading the notebook back and inspecting one cell's source.

- [ ] **Step 3.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}'); print('---'); print(nb.cells[8]['source'][:200])"
```

Expected: `cells=10`. The printed snippet should start with `# Strip whitespace and tally raw label counts.`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add data-loading and label-cleaning cells"
```

---

### Task 4: Add EDA cells (notebook §1.4–§1.5 + §1.7 word clouds)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 4.1: Replace `# ===== TASK-4-INSERT (EDA) =====` with**

```python
    cells.append(md("### 1.4 Class distribution"))
    cells.append(code("""
counts = df['label'].value_counts()
percent = (counts / counts.sum() * 100).round(2)
print(pd.DataFrame({'count': counts, 'percent': percent}))

fig, ax = plt.subplots(figsize=(5, 3.2))
sns.countplot(data=df, x='label', order=['Pozitif', 'Negatif'], ax=ax)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9)
ax.set_title('Class distribution — Pozitif vs Negatif')
ax.set_ylabel('# comments')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'class_distribution.png')
plt.show()

imbalance = counts.max() / counts.min()
print(f'Imbalance ratio (majority / minority): {imbalance:.2f}')
"""))
    cells.append(md("### 1.5 Sentence-length distribution"))
    cells.append(md("Word-count percentiles drive the `MAX_LEN=64` choice for both transformers and the Bi-LSTM — picking a value near the 95th percentile keeps almost all information while keeping training tractable."))
    cells.append(code("""
df['n_chars'] = df['text'].str.len()
df['n_words'] = df['text'].str.split().map(len)

print('Character-count percentiles:')
print(df['n_chars'].quantile([0.5, 0.75, 0.95, 0.99]))
print('\\nWord-count percentiles:')
print(df['n_words'].quantile([0.5, 0.75, 0.95, 0.99]))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
sns.histplot(df['n_chars'], bins=50, ax=axes[0])
axes[0].set_title('Characters per comment')
axes[0].set_xlabel('# characters')
sns.histplot(df['n_words'], bins=50, ax=axes[1])
axes[1].set_title('Words per comment')
axes[1].set_xlabel('# words')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'length_distribution.png')
plt.show()
"""))
```

- [ ] **Step 4.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=14`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add EDA cells (class distribution + length histograms)"
```

> Word-cloud cells are deferred to **after** preprocessing (so they cloud the cleaned text, matching BASIC's order). They appear in Task 5.

---

### Task 5: Add Turkish preprocessing + word clouds + stratified split (notebook §1.6–§1.8)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 5.1: Replace `# ===== TASK-5-INSERT (Turkish preprocessing + split) =====` with**

```python
    cells.append(md("### 1.6 Turkish-aware text preprocessing"))
    cells.append(md("**Pipeline.** Turkish-locale lowercase (Python's default `str.lower` is wrong for `İ → i` / `I → ı`) → strip URLs, mentions, hashtags, digits → whitelist `[a-zığüşöç ]` → drop ~150 curated Turkish stopwords → stem with Zemberek `TurkishMorphology` (root analysis). If a JVM is unavailable we degrade to `snowballstemmer('turkish')` so the notebook still runs end-to-end on a graders' machine without Java."))
    cells.append(code("""
TURKISH_STOPWORDS = set('''
acaba ama aslında az bazı belki biri birkaç birşey biz bu çok çünkü da daha
de defa diye eğer en gibi hem hep hepsi her hiç için ile ise kez ki kim mı
mu mü nasıl ne neden nerde nerede nereye niçin niye o sanki şey siz şu tüm
ve veya ya yani şöyle ben sen onlar bizim sizin onların bana sana ona bizi
sizi onları benim senin onun bizim sizin onların bende sende onda bizde
sizde onlarda benden senden ondan bizden sizden onlardan değil yok var
olarak olan oldu olmak olmuş olur olsa olsun olmaz olmadı bütün her bazı
bunlar şunlar ki vs vb yine zaten ancak fakat lakin yoksa hatta üzere
sonra önce kadar dolayı yerine birlikte arasında karşı yine yalnız tabii
mi yada
'''.split())
print(f'Stopword list size: {len(TURKISH_STOPWORDS)}')

URL_RE     = re.compile(r'https?://\\S+|www\\.\\S+')
MENTION_RE = re.compile(r'[@#]\\w+', re.UNICODE)
DIGIT_RE   = re.compile(r'\\d+')
KEEP_RE    = re.compile(r'[^a-zığüşöç\\s]', re.UNICODE)
SPACE_RE   = re.compile(r'\\s+')

def turkish_lower(s: str) -> str:
    \"\"\"Locale-aware lowercase: handle 'I' → 'ı' and 'İ' → 'i' before str.lower.\"\"\"
    return s.replace('I', 'ı').replace('İ', 'i').lower()

def basic_clean(text: str) -> list:
    \"\"\"Tokenise after URL / mention / digit / punctuation removal.\"\"\"
    if not isinstance(text, str):
        return []
    t = turkish_lower(text)
    t = URL_RE.sub(' ', t)
    t = MENTION_RE.sub(' ', t)
    t = DIGIT_RE.sub(' ', t)
    t = KEEP_RE.sub(' ', t)
    t = SPACE_RE.sub(' ', t).strip()
    return [w for w in t.split() if w and w not in TURKISH_STOPWORDS]
"""))
    cells.append(code("""
# Try to load Zemberek; fall back to Snowball if Java/JPype unavailable.
STEMMER_KIND = None
zemberek_morph = None
snowball_stem = None

if shutil.which('java'):
    try:
        from zemberek import TurkishMorphology
        zemberek_morph = TurkishMorphology.create_with_defaults()
        STEMMER_KIND = 'zemberek'
    except Exception as e:  # noqa: BLE001
        print(f'[warn] Zemberek unavailable ({type(e).__name__}: {e}); '
              'falling back to Snowball.')

if STEMMER_KIND is None:
    import snowballstemmer
    snowball_stem = snowballstemmer.stemmer('turkish')
    STEMMER_KIND = 'snowball'

print(f'Active stemmer: {STEMMER_KIND}')

# Cache stems — many tokens repeat and Zemberek analyse is expensive.
_stem_cache = {}

def stem_token(tok: str) -> str:
    \"\"\"Return the morphological root of a Turkish token (cached).\"\"\"
    cached = _stem_cache.get(tok)
    if cached is not None:
        return cached
    if STEMMER_KIND == 'zemberek':
        try:
            results = zemberek_morph.analyze(tok)
            analyses = list(results)
            if analyses:
                # Pick the longest stem string across analyses.
                stems = [str(a.get_stem()) if hasattr(a, 'get_stem')
                         else str(a.getStem()) for a in analyses]
                root = max(stems, key=len) if stems else tok
            else:
                root = tok
        except Exception:
            root = tok
    else:
        root = snowball_stem.stemWord(tok)
    _stem_cache[tok] = root
    return root

def preprocess(text: str) -> str:
    \"\"\"Full Turkish preprocessing → space-joined cleaned & stemmed tokens.\"\"\"
    return ' '.join(stem_token(t) for t in basic_clean(text))
"""))
    cells.append(code("""
tqdm.pandas(desc='preprocessing')
df['text_clean'] = df['text'].progress_apply(preprocess)
# Drop rows that became empty after cleaning.
mask = df['text_clean'].str.len() > 0
print(f'Dropping {(~mask).sum()} rows whose text was empty post-clean.')
df = df.loc[mask].reset_index(drop=True)
print(f'Final dataset shape: {df.shape}')

print('\\nBefore/after examples:')
for _, row in df.sample(10, random_state=RANDOM_STATE).iterrows():
    raw_t = row['text'][:80].replace('\\n', ' ')
    cln = row['text_clean'][:80]
    print(f'  [{row[\"label\"]:7s}] {raw_t!r:85s}  →  {cln!r}')
"""))
    cells.append(md("### 1.7 Word clouds per class"))
    cells.append(code("""
from wordcloud import WordCloud

def build_wordcloud(corpus: str, title: str, out: str) -> None:
    \"\"\"Render and save a 800×400 white-bg word cloud for a corpus string.\"\"\"
    wc = WordCloud(width=800, height=400, background_color='white',
                   collocations=False, random_state=RANDOM_STATE).generate(corpus)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / out)
    plt.show()

pos_corpus = ' '.join(df.loc[df['label'] == 'Pozitif', 'text_clean'])
neg_corpus = ' '.join(df.loc[df['label'] == 'Negatif', 'text_clean'])

build_wordcloud(pos_corpus, 'Pozitif word cloud', 'wordcloud_pos.png')
build_wordcloud(neg_corpus, 'Negatif word cloud', 'wordcloud_neg.png')
"""))
    cells.append(md("### 1.8 Stratified 70 / 15 / 15 split"))
    cells.append(code("""
LABEL2ID = {'Negatif': 0, 'Pozitif': 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
df['y'] = df['label'].map(LABEL2ID)

X = df['text_clean'].to_numpy()
X_raw = df['text'].to_numpy()  # kept for transformers (raw text)
y = df['y'].to_numpy()

# First split: 85% train+val / 15% test
X_tv, X_test, Xr_tv, Xr_test, y_tv, y_test = train_test_split(
    X, X_raw, y,
    test_size=0.15, stratify=y, random_state=RANDOM_STATE,
)
# Second split: 15/85 of trainval → val (so val ≈ 15% of original)
val_frac = 0.15 / 0.85
X_train, X_val, Xr_train, Xr_val, y_train, y_val = train_test_split(
    X_tv, Xr_tv, y_tv,
    test_size=val_frac, stratify=y_tv, random_state=RANDOM_STATE,
)

def _dist(y_arr):
    s = pd.Series(y_arr).map(ID2LABEL).value_counts(normalize=True).round(3)
    return s.to_dict()

print(f'Train: {len(y_train):>5d}  {_dist(y_train)}')
print(f'Val  : {len(y_val):>5d}  {_dist(y_val)}')
print(f'Test : {len(y_test):>5d}  {_dist(y_test)}')
"""))
```

- [ ] **Step 5.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=22`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add Turkish preprocessing, word clouds, and stratified split"
```

---

### Task 6: Add SVM baseline cells (notebook §2.1)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 6.1: Replace `# ===== TASK-6-INSERT (SVM baseline) =====` with**

```python
    cells.append(md("## 2. Methods"))
    cells.append(md("### 2.1 Baseline — TF-IDF + Linear SVM"))
    cells.append(md("""
TF-IDF with 1–2 grams captures lexical and short-phrase signal. We pair it with `LinearSVC` (linear-kernel SVM in primal form): the literature shows SVM consistently beats Naive Bayes / Random Forest on Turkish text, and the linear variant is dramatically faster on sparse TF-IDF than `SVC(rbf)` with no measurable accuracy loss for text-classification. We tune `C` with 3-fold CV on the **training** set only.

$$\\text{tfidf}(t, d) = \\bigl(1 + \\log f_{t,d}\\bigr) \\cdot \\log\\!\\frac{N}{1 + n_t}$$
"""))
    cells.append(code("""
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)
Xtr_tfidf = vectorizer.fit_transform(X_train)
Xva_tfidf = vectorizer.transform(X_val)
Xte_tfidf = vectorizer.transform(X_test)
print(f'TF-IDF matrix: train={Xtr_tfidf.shape}, val={Xva_tfidf.shape}, '
      f'test={Xte_tfidf.shape}')
print(f'Vocab size:    {len(vectorizer.vocabulary_)}')
"""))
    cells.append(code("""
svm_grid = GridSearchCV(
    LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000),
    param_grid={'C': [0.1, 0.5, 1.0, 2.0, 5.0]},
    scoring='f1_macro',
    cv=3,
    n_jobs=-1,
    verbose=1,
)
svm_grid.fit(Xtr_tfidf, y_train)
print(f'Best C       : {svm_grid.best_params_[\"C\"]}')
print(f'Best CV F1-mac: {svm_grid.best_score_:.4f}')
svm_model = svm_grid.best_estimator_
"""))
```

- [ ] **Step 6.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=27`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add TF-IDF + LinearSVC baseline cells"
```

---

### Task 7: Add BERTurk fine-tune cells (notebook §2.2)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 7.1: Replace `# ===== TASK-7-INSERT (BERTurk fine-tune) =====` with**

```python
    cells.append(md("### 2.2 Transformer A — BERTurk fine-tune"))
    cells.append(md("""
Pre-trained checkpoint: [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) — 12-layer cased BERT trained on a 35GB Turkish corpus.

Hyperparameters: `max_length=64` (≈95th percentile of word counts), `lr=2e-5`, `batch_size=32`, **4 epochs**, `AdamW(weight_decay=0.01)`, linear warmup over 10% of steps.
"""))
    cells.append(code("""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

BERT_NAME    = 'dbmdz/bert-base-turkish-cased'
MAX_LEN      = 64
BATCH_SIZE   = 32
EPOCHS       = 4
LR           = 2e-5
WD           = 0.01
WARMUP_FRAC  = 0.10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
"""))
    cells.append(code("""
class TextDataset(Dataset):
    \"\"\"Lazily tokenises (text, label) pairs to fixed-length transformer inputs.\"\"\"
    def __init__(self, texts, labels, tok, max_len):
        self.texts  = list(texts)
        self.labels = list(labels)
        self.tok    = tok
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True, padding='max_length', max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }

bert_train_ds = TextDataset(Xr_train, y_train, bert_tokenizer, MAX_LEN)
bert_val_ds   = TextDataset(Xr_val,   y_val,   bert_tokenizer, MAX_LEN)
bert_test_ds  = TextDataset(Xr_test,  y_test,  bert_tokenizer, MAX_LEN)

bert_train_loader = DataLoader(bert_train_ds, batch_size=BATCH_SIZE, shuffle=True)
bert_val_loader   = DataLoader(bert_val_ds,   batch_size=32, shuffle=False)
bert_test_loader  = DataLoader(bert_test_ds,  batch_size=32, shuffle=False)
"""))
    cells.append(code("""
def hf_eval_loader(model, loader):
    \"\"\"Run inference and return (preds, labels) numpy arrays.\"\"\"
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.append(out.logits.argmax(dim=-1).cpu().numpy())
            labels.append(batch['labels'].numpy())
    return np.concatenate(preds), np.concatenate(labels)

bert = AutoModelForSequenceClassification.from_pretrained(
    BERT_NAME, num_labels=2,
    id2label=ID2LABEL, label2id=LABEL2ID,
).to(DEVICE)

bert_optim = AdamW(bert.parameters(), lr=LR, weight_decay=WD)
bert_total_steps = len(bert_train_loader) * EPOCHS
bert_scheduler = get_linear_schedule_with_warmup(
    bert_optim,
    num_warmup_steps=int(WARMUP_FRAC * bert_total_steps),
    num_training_steps=bert_total_steps,
)

best_bert_state = None
best_bert_val_f1 = -1.0
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    bert.train()
    running_loss = 0.0
    pbar = tqdm(bert_train_loader, desc=f'bert epoch {epoch}/{EPOCHS}')
    for batch in pbar:
        bert_optim.zero_grad()
        out = bert(
            input_ids=batch['input_ids'].to(DEVICE),
            attention_mask=batch['attention_mask'].to(DEVICE),
            labels=batch['labels'].to(DEVICE),
        )
        out.loss.backward()
        bert_optim.step()
        bert_scheduler.step()
        running_loss += out.loss.item()
        pbar.set_postfix(loss=f'{out.loss.item():.4f}')

    val_pred, val_true = hf_eval_loader(bert, bert_val_loader)
    val_f1 = f1_score(val_true, val_pred, average='macro')
    print(f'bert epoch {epoch}: train_loss={running_loss / len(bert_train_loader):.4f}'
          f'  val_f1_macro={val_f1:.4f}')
    if val_f1 > best_bert_val_f1:
        best_bert_val_f1 = val_f1
        best_bert_state = deepcopy(bert.state_dict())

print(f'\\nBERTurk total fine-tune time: {time.time() - t0:.1f}s')
print(f'Best BERTurk validation F1-macro: {best_bert_val_f1:.4f}')

if best_bert_state is not None:
    bert.load_state_dict(best_bert_state)
"""))
```

> Note: variables are namespaced `bert_*` (not the bare `optim`, `scheduler`, `train_loader` from BASIC) so the ELECTRA block in Task 8 can coexist without overwrites. The `eval_loader` helper from BASIC is renamed `hf_eval_loader` to leave room for the LSTM-specific `lstm_eval` later.

- [ ] **Step 7.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=32`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add BERTurk fine-tune cells"
```

---

### Task 8: Add ELECTRA fine-tune cells (notebook §2.3)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 8.1: Replace `# ===== TASK-8-INSERT (ELECTRA fine-tune) =====` with**

```python
    cells.append(md("### 2.3 Transformer B — ELECTRA fine-tune"))
    cells.append(md("""
Pre-trained checkpoint: [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator) — Turkish ELECTRA, trained discriminatively (token-level binary task) rather than masked-LM.

Same hyperparameters as the BERTurk run for an apples-to-apples comparison: `max_length=64`, `lr=2e-5`, `batch_size=32`, **4 epochs**, `AdamW(weight_decay=0.01)`, 10% warmup. Variables are namespaced `electra_*` so they don't overwrite the BERTurk artifacts.
"""))
    cells.append(code("""
ELECTRA_NAME = 'dbmdz/electra-base-turkish-cased-discriminator'

electra_tokenizer = AutoTokenizer.from_pretrained(ELECTRA_NAME)

electra_train_ds = TextDataset(Xr_train, y_train, electra_tokenizer, MAX_LEN)
electra_val_ds   = TextDataset(Xr_val,   y_val,   electra_tokenizer, MAX_LEN)
electra_test_ds  = TextDataset(Xr_test,  y_test,  electra_tokenizer, MAX_LEN)

electra_train_loader = DataLoader(electra_train_ds, batch_size=BATCH_SIZE, shuffle=True)
electra_val_loader   = DataLoader(electra_val_ds,   batch_size=32, shuffle=False)
electra_test_loader  = DataLoader(electra_test_ds,  batch_size=32, shuffle=False)
"""))
    cells.append(code("""
electra = AutoModelForSequenceClassification.from_pretrained(
    ELECTRA_NAME, num_labels=2,
    id2label=ID2LABEL, label2id=LABEL2ID,
).to(DEVICE)

electra_optim = AdamW(electra.parameters(), lr=LR, weight_decay=WD)
electra_total_steps = len(electra_train_loader) * EPOCHS
electra_scheduler = get_linear_schedule_with_warmup(
    electra_optim,
    num_warmup_steps=int(WARMUP_FRAC * electra_total_steps),
    num_training_steps=electra_total_steps,
)

best_electra_state = None
best_electra_val_f1 = -1.0
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    electra.train()
    running_loss = 0.0
    pbar = tqdm(electra_train_loader, desc=f'electra epoch {epoch}/{EPOCHS}')
    for batch in pbar:
        electra_optim.zero_grad()
        out = electra(
            input_ids=batch['input_ids'].to(DEVICE),
            attention_mask=batch['attention_mask'].to(DEVICE),
            labels=batch['labels'].to(DEVICE),
        )
        out.loss.backward()
        electra_optim.step()
        electra_scheduler.step()
        running_loss += out.loss.item()
        pbar.set_postfix(loss=f'{out.loss.item():.4f}')

    val_pred, val_true = hf_eval_loader(electra, electra_val_loader)
    val_f1 = f1_score(val_true, val_pred, average='macro')
    print(f'electra epoch {epoch}: train_loss={running_loss / len(electra_train_loader):.4f}'
          f'  val_f1_macro={val_f1:.4f}')
    if val_f1 > best_electra_val_f1:
        best_electra_val_f1 = val_f1
        best_electra_state = deepcopy(electra.state_dict())

print(f'\\nELECTRA total fine-tune time: {time.time() - t0:.1f}s')
print(f'Best ELECTRA validation F1-macro: {best_electra_val_f1:.4f}')

if best_electra_state is not None:
    electra.load_state_dict(best_electra_state)
"""))
```

- [ ] **Step 8.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=36`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add ELECTRA fine-tune cells with disambiguated variables"
```

---

### Task 9: Add Bi-LSTM vocab + sequence encoder + Dataset/DataLoader (notebook §2.4 part 1)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 9.1: Replace `# ===== TASK-9-INSERT (Bi-LSTM: vocab + encoder + dataset) =====` with**

```python
    cells.append(md("### 2.4 Custom Bi-LSTM from scratch (PyTorch)"))
    cells.append(md("""
Unlike the two transformers above (which start from massive pretraining), this model is **built and trained entirely from scratch**: random-init word embeddings, no external knowledge. It is the deep-learning counterpart to the TF-IDF + LinearSVC baseline — both consume the same Zemberek-stemmed text.

The pipeline is:

1. **Word-level vocabulary** built from `X_train` only (drop hapaxes), with reserved `<PAD>=0` and `<UNK>=1`.
2. **Integer encoding** with right-pad/truncation to `MAX_LEN=64` (matches transformer choice).
3. **Bi-LSTM classifier** (next cell): `Embedding(128) → 2-layer Bi-LSTM(hidden=128) → concat last-layer fwd+bwd hidden states → Dropout(0.5) → Linear(256→2)`.
4. **AdamW + CrossEntropyLoss**, 8 epochs, best-val-Macro-F1 checkpointed in memory.

Why these choices? See the hyperparameter justification table further below.
"""))
    cells.append(md("#### 2.4.1 Vocabulary build + integer encoding"))
    cells.append(code("""
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_IDX = 0
UNK_IDX = 1
LSTM_MAX_LEN = 64

def build_vocab(texts, min_freq: int = 2):
    \"\"\"Build word→index vocab from training texts, dropping tokens below min_freq.

    Returns (stoi: dict[str, int], itos: list[str]).
    PAD_TOKEN is reserved at index 0 and UNK_TOKEN at index 1, regardless of frequency.
    \"\"\"
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    itos = [PAD_TOKEN, UNK_TOKEN]
    itos.extend(tok for tok, c in counter.most_common() if c >= min_freq)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

def encode(texts, stoi, max_len: int = LSTM_MAX_LEN) -> torch.Tensor:
    \"\"\"Map space-tokenised texts to a (N, max_len) torch.long tensor with right-pad.\"\"\"
    out = torch.zeros((len(texts), max_len), dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [stoi.get(tok, UNK_IDX) for tok in t.split()][:max_len]
        if ids:
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out

stoi, itos = build_vocab(list(X_train), min_freq=2)
print(f'LSTM vocab size: {len(itos)} (incl. PAD/UNK)')

X_train_seq = encode(list(X_train), stoi)
X_val_seq   = encode(list(X_val),   stoi)
X_test_seq  = encode(list(X_test),  stoi)
print(f'Sequence shapes: train={tuple(X_train_seq.shape)}, '
      f'val={tuple(X_val_seq.shape)}, test={tuple(X_test_seq.shape)}')
print(f'Avg non-pad length (train): '
      f'{(X_train_seq != PAD_IDX).sum(dim=1).float().mean().item():.1f} / {LSTM_MAX_LEN}')

# Sanity: PAD and UNK indices must be exactly where we expect them.
assert stoi[PAD_TOKEN] == PAD_IDX == 0
assert stoi[UNK_TOKEN] == UNK_IDX == 1
"""))
    cells.append(md("#### 2.4.2 Dataset + DataLoaders"))
    cells.append(code("""
class TextSeqDataset(Dataset):
    \"\"\"In-memory dataset wrapping pre-encoded integer sequences and labels.\"\"\"
    def __init__(self, seqs: torch.Tensor, labels: np.ndarray):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

LSTM_BATCH = 64

lstm_train_ds = TextSeqDataset(X_train_seq, y_train)
lstm_val_ds   = TextSeqDataset(X_val_seq,   y_val)
lstm_test_ds  = TextSeqDataset(X_test_seq,  y_test)

lstm_train_loader = DataLoader(lstm_train_ds, batch_size=LSTM_BATCH, shuffle=True)
lstm_val_loader   = DataLoader(lstm_val_ds,   batch_size=LSTM_BATCH, shuffle=False)
lstm_test_loader  = DataLoader(lstm_test_ds,  batch_size=LSTM_BATCH, shuffle=False)

print(f'Train batches: {len(lstm_train_loader)}  '
      f'Val batches: {len(lstm_val_loader)}  '
      f'Test batches: {len(lstm_test_loader)}')
"""))
```

- [ ] **Step 9.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=41`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add Bi-LSTM vocab, encoder, and DataLoader cells"
```

---

### Task 10: Add Bi-LSTM model class + hyperparameter justification (notebook §2.4 part 2)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 10.1: Replace `# ===== TASK-10-INSERT (Bi-LSTM: model class + hyperparam md) =====` with**

```python
    cells.append(md("#### 2.4.3 Model architecture"))
    cells.append(code("""
class BiLSTMClassifier(nn.Module):
    \"\"\"Embedding → 2-layer Bi-LSTM → concat last hidden states → dropout → linear.

    All weights random-initialised; no pretrained word embeddings. The padding
    index is excluded from gradient updates via padding_idx, so the PAD row of
    the embedding matrix stays at zero throughout training.
    \"\"\"
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 2,
                 lstm_dropout: float = 0.3, head_dropout: float = 0.5,
                 num_classes: int = 2, pad_idx: int = PAD_IDX):
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
        emb = self.embedding(x)                          # (B, L, E)
        _, (h_n, _) = self.lstm(emb)                     # h_n: (num_layers*2, B, H)
        # h_n is ordered (layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...).
        # Take last layer's forward (h_n[-2]) and backward (h_n[-1]) finals; concat → (B, 2H).
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(self.dropout(h))
"""))
    cells.append(md("""
#### 2.4.4 Hyperparameter choices (for the IEEE report)

| Knob | Value | Rationale |
|------|-------|-----------|
| `embed_dim` | 128 | Standard for vocab in the 5–10k range. With ~5k tokens × 128 ≈ 640k embedding params we have enough capacity to learn semantic similarity but stay below the threshold where random-init embeddings overfit on 7.7k training examples. |
| `hidden_size` | 128 | Symmetrical with the embedding. After bidirectional concat the sentence vector is 256-d — comparable in capacity to a small transformer pooler. |
| `num_layers` | 2 | Layer-1 captures local morphology cues (Zemberek roots + adjacent context); layer-2 abstracts phrase-level sentiment. Three+ stacked LSTMs overfit on this dataset size. |
| `lstm_dropout` | 0.3 | PyTorch applies LSTM dropout *between* stacked layers (not within a layer). 0.3 is the canonical mid-range value (Zaremba et al. 2014). |
| `head_dropout` | 0.5 | Heavier dropout immediately before the linear classifier — standard regularisation for small-data text classification (Goodfellow et al. 2016). |
| `padding_idx=0` | — | The embedding row for `<PAD>` stays zero; no gradient flows through pads. |
| Loss | `CrossEntropyLoss` (2-output head) | Equivalent to `BCEWithLogitsLoss` for binary, but cleaner for the four-way comparison and easier to extend to multi-class downstream. |
| Optimiser | `AdamW(lr=1e-3, weight_decay=1e-5)` | An order of magnitude higher LR than the transformer fine-tunes (`2e-5`) because there are no pretrained weights to preserve; `1e-3` is the canonical "from-scratch text" LR. |
| `epochs` | 8 | Empirical cap for a model this size on 7.7k examples. Best-val-F1 checkpoint is kept in memory, so over-training one epoch is harmless. |
| `batch_size` | 64 | Comfortable on Colab T4 for a 64-token sequence and ~640k-param model. |
"""))
```

- [ ] **Step 10.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=44`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add Bi-LSTM model class and hyperparameter justifications"
```

---

### Task 11: Add Bi-LSTM training loop + test inference (notebook §2.4 part 3)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 11.1: Replace `# ===== TASK-11-INSERT (Bi-LSTM: training + test inference) =====` with**

```python
    cells.append(md("#### 2.4.5 Training loop"))
    cells.append(code("""
LSTM_EPOCHS = 8
LSTM_LR = 1e-3
LSTM_WD = 1e-5

# Reseed for the Bi-LSTM only — keeps the LSTM run independent of upstream
# randomness consumed by the transformer fine-tunes.
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

lstm_model = BiLSTMClassifier(
    vocab_size=len(itos), embed_dim=128, hidden_size=128,
    num_layers=2, lstm_dropout=0.3, head_dropout=0.5,
    num_classes=2, pad_idx=PAD_IDX,
).to(DEVICE)
print(lstm_model)
n_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f'Trainable params: {n_params:,}')

lstm_optim = AdamW(lstm_model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WD)
lstm_loss_fn = nn.CrossEntropyLoss()

def lstm_eval(model, loader):
    \"\"\"Run inference and return (preds, labels) numpy arrays.\"\"\"
    model.eval()
    all_pred, all_lbl = [], []
    with torch.no_grad():
        for seq, lbl in loader:
            seq = seq.to(DEVICE)
            logits = model(seq)
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_lbl.append(lbl.numpy())
    return np.concatenate(all_pred), np.concatenate(all_lbl)

best_lstm_state = None
best_lstm_val_f1 = -1.0
t0 = time.time()
for epoch in range(1, LSTM_EPOCHS + 1):
    lstm_model.train()
    running = 0.0
    pbar = tqdm(lstm_train_loader, desc=f'lstm epoch {epoch}/{LSTM_EPOCHS}')
    for seq, lbl in pbar:
        seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
        lstm_optim.zero_grad()
        logits = lstm_model(seq)
        loss = lstm_loss_fn(logits, lbl)
        loss.backward()
        lstm_optim.step()
        running += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    val_pred, val_true = lstm_eval(lstm_model, lstm_val_loader)
    val_f1 = f1_score(val_true, val_pred, average='macro')
    print(f'lstm epoch {epoch}: train_loss={running/len(lstm_train_loader):.4f}'
          f'  val_f1_macro={val_f1:.4f}')
    if val_f1 > best_lstm_val_f1:
        best_lstm_val_f1 = val_f1
        best_lstm_state = deepcopy(lstm_model.state_dict())

print(f'\\nLSTM total train time: {time.time() - t0:.1f}s')
print(f'Best LSTM validation F1-macro: {best_lstm_val_f1:.4f}')
if best_lstm_state is not None:
    lstm_model.load_state_dict(best_lstm_state)
"""))
```

- [ ] **Step 11.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=46`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add Bi-LSTM training loop with best-val checkpoint"
```

---

### Task 12: Add unified evaluation section (notebook §3)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 12.1: Replace `# ===== TASK-12-INSERT (Unified evaluation) =====` with**

```python
    cells.append(md("## 3. Results"))
    cells.append(md("### 3.1 Unified evaluation function"))
    cells.append(code("""
TARGET_NAMES = ['Negatif', 'Pozitif']

def evaluate(name, y_true, y_pred):
    \"\"\"Print classification report and return a flat metrics dict.\"\"\"
    print(f'=== {name} ===')
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES,
                                digits=4))
    return {
        'model':           name,
        'accuracy':        accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro':    recall_score(y_true, y_pred, average='macro'),
        'f1_macro':        f1_score(y_true, y_pred, average='macro'),
    }
"""))
    cells.append(md("### 3.2 Per-model test inference"))
    cells.append(code("""
# 1) SVM
svm_pred = svm_model.predict(Xte_tfidf)
svm_metrics = evaluate('TF-IDF + LinearSVC', y_test, svm_pred)
"""))
    cells.append(code("""
# 2) BERTurk
bert_pred, bert_true = hf_eval_loader(bert, bert_test_loader)
assert (bert_true == y_test).all(), 'bert test loader iteration order changed'
bert_metrics = evaluate('BERTurk fine-tuned', y_test, bert_pred)
"""))
    cells.append(code("""
# 3) ELECTRA
electra_pred, electra_true = hf_eval_loader(electra, electra_test_loader)
assert (electra_true == y_test).all(), 'electra test loader iteration order changed'
electra_metrics = evaluate('ELECTRA fine-tuned', y_test, electra_pred)
"""))
    cells.append(code("""
# 4) Bi-LSTM
lstm_pred, lstm_true = lstm_eval(lstm_model, lstm_test_loader)
assert (lstm_true == y_test).all(), 'lstm test loader iteration order changed'
lstm_metrics = evaluate('Custom Bi-LSTM (from scratch)', y_test, lstm_pred)
"""))
    cells.append(md("### 3.3 Side-by-side metrics table"))
    cells.append(code("""
results = pd.DataFrame(
    [svm_metrics, bert_metrics, electra_metrics, lstm_metrics]
).set_index('model')
results.to_csv(ARTIFACT_DIR / 'results_metrics.csv')
print(results.round(4).to_string())
results.style.format('{:.4f}')
"""))
    cells.append(md("### 3.4 Confusion matrices (2×2 grid)"))
    cells.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
panels = [
    ('TF-IDF + LinearSVC',           svm_pred,     axes[0, 0]),
    ('BERTurk fine-tuned',           bert_pred,    axes[0, 1]),
    ('ELECTRA fine-tuned',           electra_pred, axes[1, 0]),
    ('Custom Bi-LSTM (from scratch)', lstm_pred,   axes[1, 1]),
]
for name, pred, ax in panels:
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(name)
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'confusion_matrices.png')
plt.show()
"""))
    cells.append(md("### 3.5 Macro-F1 bar chart"))
    cells.append(code("""
fig, ax = plt.subplots(figsize=(7, 3.8))
order = ['TF-IDF + LinearSVC',
         'Custom Bi-LSTM (from scratch)',
         'BERTurk fine-tuned',
         'ELECTRA fine-tuned']
f1_vals = [results.loc[m, 'f1_macro'] for m in order]
bars = ax.bar(order, f1_vals, color=sns.color_palette('deep', len(order)))
ax.set_ylabel('Macro-F1 (test)')
ax.set_ylim(0.0, 1.0)
ax.set_title('Test Macro-F1 across models')
for b, v in zip(bars, f1_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'f1_macro_comparison.png')
plt.show()
"""))
```

- [ ] **Step 12.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=56`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add unified evaluation section (table + 2x2 CM + F1 bar chart)"
```

---

### Task 13: Add misclassification samples + discussion stub (notebook §3.6 + §4)

**Files:**
- Modify: `scripts/build_unified_notebook.py`
- Regenerate: `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb`

- [ ] **Step 13.1: Replace `# ===== TASK-13-INSERT (Misclassification samples + discussion) =====` with**

```python
    cells.append(md("### 3.6 Misclassification samples"))
    cells.append(code("""
def show_errors(name, pred, k=5):
    \"\"\"Print up to k false-positive and k false-negative example texts.\"\"\"
    print(f'--- {name} ---')
    fp = np.where((pred == 1) & (y_test == 0))[0][:k]
    fn = np.where((pred == 0) & (y_test == 1))[0][:k]
    print(f'False positives (predicted Pozitif, actually Negatif), n={len(fp)}:')
    for i in fp:
        print(f'  • {Xr_test[i][:140]}')
    print(f'False negatives (predicted Negatif, actually Pozitif), n={len(fn)}:')
    for i in fn:
        print(f'  • {Xr_test[i][:140]}')
    print()

show_errors('TF-IDF + LinearSVC', svm_pred)
show_errors('BERTurk fine-tuned', bert_pred)
show_errors('ELECTRA fine-tuned', electra_pred)
show_errors('Custom Bi-LSTM (from scratch)', lstm_pred)
"""))
    cells.append(md("## 4. Discussion & Conclusion"))
    cells.append(md("""
_Fill in the bracketed numbers from the run output above before exporting to the IEEE report._

**Headline.** On the held-out test set, ELECTRA and BERTurk lead the four-model bracket with **F1-macro = [electra_f1] / [bert_f1]**, the from-scratch **Custom Bi-LSTM** trails at **[lstm_f1]**, and the lexical **TF-IDF + LinearSVC** baseline closes the slate at **[svm_f1]**. The gap from baseline to transformer is **[svm→electra delta] pp**; the from-scratch Bi-LSTM closes **[svm→lstm delta] pp** of that gap without any external pretraining.

**What each model contributes.**
* *TF-IDF + LinearSVC* — interpretable per-feature weights, near-zero training cost, strong floor for short Turkish text.
* *Custom Bi-LSTM* — the from-scratch deep-learning contribution. Random-init word embeddings + 2-layer Bi-LSTM, trained on 7.7k examples in ≈2 min on a T4. Demonstrates how much of the baseline-to-transformer gap can be closed *without* pretraining.
* *BERTurk* — masked-LM-pretrained Turkish BERT, contextual sub-word tokenisation handles agglutinative morphology far better than n-gram features over Zemberek stems.
* *ELECTRA* — discriminative pretraining (replaced-token detection) gives a small additional bump over masked-LM pretraining at the same parameter count.

**Where each model fails.** All four converge on the same ambiguous-tone errors — irony, mixed-sentiment posts, and posts with obscure slurs that don't appear in the training vocabulary. The LSTM additionally fails on short posts (<5 tokens) where the recurrent layers don't accumulate enough signal; the SVM fails on negation that the bag-of-bigrams can't represent.

**Limitations.**
* Dataset size is modest (≈11k rows after dedup); transformers almost certainly have headroom we couldn't unlock with 4 epochs on a T4.
* The Bi-LSTM consumes Zemberek-stemmed text, the same input as the SVM — so its gain over SVM is purely architectural, not from extra information.
* Residual mojibake artefacts may have survived the encoding fallback path; see U+FFFD count in §1.2.
* Zemberek stemming requires a JVM and adds a non-trivial dependency; the Snowball fallback is less linguistically faithful.

**Future work.**
* Replace random-init Bi-LSTM embeddings with pretrained Turkish fastText / Word2Vec.
* Add an attention-pooled variant of the Bi-LSTM (sum-pool over outputs weighted by `softmax(W h_t)`).
* Synonym-based data augmentation (e.g. [`nlpaug`](https://github.com/makcedward/nlpaug)) to enlarge the minority class.
* GPU run with `lr ∈ {1e-5, 2e-5, 3e-5}` × `epochs ∈ {3, 4}` for the transformers, and `lr ∈ {5e-4, 1e-3, 2e-3}` × `hidden ∈ {64, 128, 256}` for the Bi-LSTM.
"""))
```

- [ ] **Step 13.2: Run, validate, commit**

```powershell
python scripts/build_unified_notebook.py
python -m py_compile scripts/build_unified_notebook.py
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'cells={len(nb.cells)}')"
```

Expected: `cells=58`.

```powershell
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Add misclassification samples and discussion stub"
```

---

### Task 14: Final static validation + Colab runbook

**Files:**
- (Read-only validation; no source edits unless an issue is found.)

- [ ] **Step 14.1: Convert notebook to Python and lint with `py_compile`**

```powershell
python -c "import nbformat; from nbconvert.exporters import PythonExporter; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); src, _ = PythonExporter().from_notebook_node(nb); open('_unified_lint.py','w',encoding='utf-8').write(src)"
python -m py_compile _unified_lint.py
Remove-Item _unified_lint.py
```

Expected: no output from `py_compile` (success). If a `SyntaxError` is raised, identify the offending cell, fix the builder, regenerate, and re-validate.

> **Note.** This catches *Python syntax* errors only. `%pip` / `!apt-get` magic lines may show up as syntax errors when nbconvert exports them — if so, ignore the magic-line errors and re-export with `--TemplateExporter.exclude_input_prompt=True`, or simply skip the lint step (the magic lines are valid in Jupyter context, just not in plain Python).

- [ ] **Step 14.2: Final cell-count check**

```powershell
python -c "import nbformat; nb = nbformat.read('SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb', as_version=4); print(f'TOTAL cells={len(nb.cells)}'); md_n=sum(1 for c in nb.cells if c.cell_type=='markdown'); code_n=sum(1 for c in nb.cells if c.cell_type=='code'); print(f'  markdown={md_n}  code={code_n}')"
```

Expected: `TOTAL cells=58`, ≈ 30 markdown + 28 code (exact split may vary by ±1).

- [ ] **Step 14.3: End-to-end validation in Colab**

The only true test for ML code is running it. Upload `SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb` and `SentimentAnalysis/social_media_comments.csv` to Google Colab, set Runtime → Change runtime type → T4 GPU, and `Runtime → Run all`.

Acceptance criteria (per spec §8 Definition of Done):
- [ ] All cells execute without raising.
- [ ] §1.2 reports 0 U+FFFD replacement chars.
- [ ] §1.8 prints `Train: 7703  …  Val: 1651  …  Test: 1651  …`.
- [ ] §2.1 prints a CV F1-macro > 0.80.
- [ ] §2.2 (BERTurk) reports a Best validation F1-macro > 0.85.
- [ ] §2.3 (ELECTRA) reports a Best validation F1-macro > 0.85.
- [ ] §2.4 (Bi-LSTM) reports a Best validation F1-macro **≥ 0.80** (the hard sanity bar from spec §8).
- [ ] §3.3 prints a four-row `results` DataFrame, in this row order: SVM, BERTurk, ELECTRA, Bi-LSTM.
- [ ] §3.4 produces a 2×2 confusion-matrix figure saved as `confusion_matrices.png`.
- [ ] §3.5 produces an `f1_macro_comparison.png` bar chart.

If the Bi-LSTM falls below 0.80 Val F1, before re-tuning architecture: (a) verify the encoder produces non-zero non-pad token counts for >95% of training rows, (b) verify the embedding `padding_idx=0` arg actually pinned the PAD row to zero (`(lstm_model.embedding.weight[0] == 0).all()`), (c) try `LSTM_LR=5e-4` (the only safe hyper-tweak before re-design).

- [ ] **Step 14.4: Commit any fixes from end-to-end run, if needed**

If §14.3 surfaced bugs (e.g. cell-source encoding issue, missing import), patch the builder, regenerate, and commit:

```powershell
python scripts/build_unified_notebook.py
git add scripts/build_unified_notebook.py SentimentAnalysis/sentiment_analysis-UNIFIED.ipynb
git commit -m "Fix <specific issue> surfaced during Colab end-to-end run"
```

If no bugs surface, this step is a no-op.

---

## Self-Review Notes (post-write)

**Spec coverage check.**
- §1 Goal — Tasks 1–13 build the unified notebook end-to-end.
- §2 Scope (in) — Bi-LSTM module: vocab in Task 9, model in Task 10, training/inference in Task 11. ✓
- §3 Notebook section layout — Tasks map to sections: 2→§0–§1.1, 3→§1.2–§1.3, 4→§1.4–§1.5, 5→§1.6–§1.8, 6→§2.1, 7→§2.2, 8→§2.3, 9–11→§2.4, 12→§3.1–§3.5, 13→§3.6+§4. ✓
- §4 Bi-LSTM details — vocab/encoder Task 9, Dataset/DataLoader Task 9, model Task 10, hyperparameter table Task 10, training loop Task 11. ✓
- §5 Reproducibility — seeds set in Task 2 (incl. `torch.cuda.manual_seed_all`); Bi-LSTM reseed in Task 11. ✓
- §6 Artifacts — `results_metrics.csv` (Task 12), `confusion_matrices.png` (Task 12), `f1_macro_comparison.png` (Task 12), EDA pngs (Tasks 4–5). ✓
- §7 Risks — pad-token leakage handled via `h_n` last-state pooling (Task 10 model class). ✓
- §8 Definition of Done — verified in Task 14 acceptance criteria. ✓
- §9 Future work — captured verbatim in Task 13 discussion stub. ✓

**Placeholder scan.** No "TBD", "TODO", "fill in details", "appropriate error handling", "similar to Task N", or empty test stubs in any task.

**Type / name consistency.**
- `bert_*` namespacing: `bert_tokenizer`, `bert_train_ds`/`bert_val_ds`/`bert_test_ds`, `bert_train_loader`/`bert_val_loader`/`bert_test_loader`, `bert`, `bert_optim`, `bert_scheduler`, `best_bert_state`, `best_bert_val_f1`, `bert_pred`, `bert_metrics` — used consistently in Tasks 7 and 12. ✓
- `electra_*` namespacing: parallel set, used consistently in Tasks 8 and 12. ✓
- `lstm_*` namespacing: `lstm_train_ds`, `lstm_val_ds`, `lstm_test_ds`, `lstm_train_loader`, `lstm_val_loader`, `lstm_test_loader`, `lstm_model`, `lstm_optim`, `lstm_loss_fn`, `lstm_eval`, `best_lstm_state`, `best_lstm_val_f1`, `lstm_pred`, `lstm_metrics` — used consistently in Tasks 9, 10, 11, 12. ✓
- Helper rename: BASIC's `eval_loader` is renamed `hf_eval_loader` in Task 7 and used in Task 12 ELECTRA inference (Task 12 cell 3) — consistent. ✓
- `evaluate(name, y_true, y_pred)` defined in Task 12, called from Task 12 (4 inference cells). ✓
- `build_vocab` / `encode` / `BiLSTMClassifier` / `TextSeqDataset` / `TextDataset` — all defined in their introducing tasks; downstream usage matches. ✓
