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
