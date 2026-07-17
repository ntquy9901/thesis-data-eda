"""PhoBERT (frozen) embedding extraction for news text.

Ported from D:\\bmad-projects\\stock_vol_prediction01\\baselines\\2026-07-07_embedding_baseline
\\code\\extract_embeddings.py. Requires ``transformers<5`` + ``sentencepiece``.
"""

from __future__ import annotations

import numpy as np


def extract_phobert_embeddings(
    texts: list[str],
    model: str = "vinai/phobert-base",
    max_len: int = 64,
    batch_size: int = 32,
) -> np.ndarray:
    """Frozen PhoBERT ``[CLS]`` embeddings for a list of texts. Returns (n, hidden_size)."""
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    raw_dim = net.config.hidden_size
    embs = np.zeros((len(texts), raw_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
            cls = net(**enc).last_hidden_state[:, 0, :]
            embs[i : i + len(batch)] = cls.cpu().numpy()

    if not np.isfinite(embs).all():
        raise ValueError(f"PhoBERT produced non-finite embeddings ({int(np.isnan(embs).sum())} NaN)")
    return embs
