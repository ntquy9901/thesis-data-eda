"""Tests for src.nlp.embeddings + src.features.news_embeddings (Story 11-1)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.nlp import embeddings as E


class _FakeEnc(dict):
    def __init__(self, batch_size):
        super().__init__()
        self["input_ids"] = torch.zeros((batch_size, 4), dtype=torch.long)

    def to(self, device):
        return self


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, batch, return_tensors="pt", truncation=True, padding=True, max_length=64):
        return _FakeEnc(len(batch))


class _FakeModel:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return SimpleNamespace(last_hidden_state=torch.randn(batch, 4, 768))

    @property
    def config(self):
        return SimpleNamespace(hidden_size=768)


def _mock_transformers(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", _FakeModel)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """A mocked-model test run must never write into the real data/features/ cache."""
    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "FEATURES_DIR", tmp_path / "features")


def test_extract_phobert_embeddings_shape(monkeypatch):
    _mock_transformers(monkeypatch)
    embs = E.extract_phobert_embeddings(["tin tức VCB tăng", "tin tức FPT giảm"], batch_size=1)
    assert embs.shape == (2, 768)
    assert np.isfinite(embs).all()


def test_extract_phobert_embeddings_empty():
    assert E.extract_phobert_embeddings([]).shape == (0, 768)


def test_extract_phobert_embeddings_nonfinite_raises(monkeypatch):
    """Finiteness guard raises ValueError (not a bare assert, which -O would strip)."""
    _mock_transformers(monkeypatch)
    import transformers

    class _NanModel(_FakeModel):
        def __call__(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            out = torch.randn(batch, 4, 768)
            out[0, 0, 0] = float("nan")
            return SimpleNamespace(last_hidden_state=out)

    monkeypatch.setattr(transformers, "AutoModel", _NanModel)
    with pytest.raises(ValueError, match="non-finite"):
        E.extract_phobert_embeddings(["tin tức"], batch_size=1)


def test_load_group_unknown_group_raises():
    from src.features.news_embeddings import _load_group

    with pytest.raises(ValueError):
        _load_group("bad_group")


def test_ticker_pattern_case_insensitive():
    from src.features.news_embeddings import TICKER_PATTERN

    assert TICKER_PATTERN.findall("cổ phiếu vcb tăng mạnh") == ["vcb"]


def test_build_group_embeddings_tong_hop_smoke(monkeypatch):
    """Real news data + mocked PhoBERT: no crash, sane schema, own-PCA basis."""
    _mock_transformers(monkeypatch)
    from src.features.news_embeddings import build_group_embeddings

    df = build_group_embeddings("tong_hop")
    if df.empty:
        pytest.skip("no news data")
    assert {"date", "ticker", "source"} <= set(df.columns)
    emb_cols = [c for c in df.columns if c.startswith(("emb_", "raw_"))]
    assert len(emb_cols) >= 1
    assert np.isfinite(df[emb_cols].to_numpy()).all()


def test_build_group_embeddings_khach_quan_smoke(monkeypatch):
    """Same smoke check for the 'khach_quan' (per-source) group."""
    _mock_transformers(monkeypatch)
    from src.features.news_embeddings import build_group_embeddings

    df = build_group_embeddings("khach_quan")
    if df.empty:
        pytest.skip("no news data")
    assert {"date", "ticker", "source"} <= set(df.columns)


def test_build_comparable_group_embeddings_shared_dim(monkeypatch):
    """Both groups reduced with the SAME PCA basis -> identical emb_* column count."""
    _mock_transformers(monkeypatch)
    from src.features.news_embeddings import build_comparable_group_embeddings

    out = build_comparable_group_embeddings()
    assert set(out.keys()) == {"khach_quan", "tong_hop"}
    dims = {
        len([c for c in df.columns if c.startswith("emb_")])
        for df in out.values() if not df.empty
    }
    assert len(dims) <= 1  # all non-empty groups share one dimensionality


def test_reduce_small_train_falls_back_to_raw_but_keeps_emb_naming(monkeypatch):
    """With <2 train-period rows, _reduce keeps the full RAW embedding (honest fallback, not a
    mislabeled 1-dim 'PCA') but STILL names columns emb_* (so downstream consumers work
    uniformly) and sets pca_applied=False so the fallback stays inspectable."""
    from src.features.news_embeddings import RAW_DIM, _reduce

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),  # both after TRAIN_CUTOFF
        "ticker": ["VCB", "FPT"],
        "source": ["cafef", "cafef"],
        **{f"raw_{i}": [0.1 * i, 0.2 * i] for i in range(RAW_DIM)},
    })
    out = _reduce(df)
    emb_cols = [c for c in out.columns if c.startswith("emb_")]
    assert len(emb_cols) == RAW_DIM
    assert (out["pca_applied"] == False).all()  # noqa: E712


def test_reduce_normal_train_applies_pca():
    from src.features.news_embeddings import RAW_DIM, _reduce

    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01"] * n),  # before TRAIN_CUTOFF
        "ticker": ["VCB"] * n,
        "source": ["cafef"] * n,
        **{f"raw_{i}": rng.normal(size=n) for i in range(RAW_DIM)},
    })
    out = _reduce(df)
    emb_cols = [c for c in out.columns if c.startswith("emb_")]
    assert 0 < len(emb_cols) <= 32
    assert (out["pca_applied"] == True).all()  # noqa: E712


def test_get_article_embeddings_incremental_only_encodes_new(monkeypatch):
    """Second call with one new url must NOT re-encode already-cached urls."""
    _mock_transformers(monkeypatch)
    from src.features.news_embeddings import _get_article_embeddings

    news1 = pd.DataFrame({"url": ["a", "b"], "_text": ["tin A", "tin B"]})
    first = _get_article_embeddings("cafef", news1)
    assert set(first["url"]) == {"a", "b"}

    calls = []
    import src.features.news_embeddings as ne

    orig = ne.extract_phobert_embeddings
    monkeypatch.setattr(ne, "extract_phobert_embeddings", lambda texts, **kw: (calls.append(texts), orig(texts, **kw))[1])

    news2 = pd.DataFrame({"url": ["a", "b", "c"], "_text": ["tin A", "tin B", "tin C moi"]})
    second = _get_article_embeddings("cafef", news2)
    assert set(second["url"]) == {"a", "b", "c"}
    assert calls == [["tin C moi"]]  # only the NEW article was encoded


def test_get_article_embeddings_corrupted_cache_self_heals(monkeypatch, tmp_path):
    """A truncated/corrupt cache parquet must not crash every future run — treated as empty."""
    _mock_transformers(monkeypatch)
    import src.features.news_embeddings as ne

    ne.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    bad_path = ne._article_cache_path("cafef")
    bad_path.write_bytes(b"not a real parquet file")

    news = pd.DataFrame({"url": ["a"], "_text": ["tin A"]})
    out = ne._get_article_embeddings("cafef", news)
    assert set(out["url"]) == {"a"}


def test_get_article_embeddings_dimension_mismatch_rebuilds(monkeypatch):
    """If a cached embedding has a different dim than RAW_DIM (e.g. model changed), that
    source's cache is rebuilt instead of silently NaN-padding mismatched columns."""
    _mock_transformers(monkeypatch)
    import src.features.news_embeddings as ne

    cache_path = ne._article_cache_path("cafef")
    ne.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    stale = pd.DataFrame({"url": ["old"], "raw_0": [0.1], "raw_1": [0.2]})  # only 2 dims, not 768
    stale.to_parquet(cache_path, index=False)

    news = pd.DataFrame({"url": ["a"], "_text": ["tin A"]})
    out = ne._get_article_embeddings("cafef", news)
    assert "old" not in set(out["url"])  # stale cache discarded, not merged with mismatched dims
    assert set(out["url"]) == {"a"}


def test_get_article_embeddings_no_new_articles_skips_encode(monkeypatch):
    """If all urls are already cached, extract_phobert_embeddings must not be called at all."""
    _mock_transformers(monkeypatch)
    from src.features.news_embeddings import _get_article_embeddings

    news = pd.DataFrame({"url": ["a"], "_text": ["tin A"]})
    _get_article_embeddings("cafef", news)

    import src.features.news_embeddings as ne

    def _boom(texts, **kw):
        raise AssertionError("should not re-encode cached articles")

    monkeypatch.setattr(ne, "extract_phobert_embeddings", _boom)
    again = _get_article_embeddings("cafef", news)
    assert set(again["url"]) == {"a"}


def test_get_article_embeddings_isolated_per_source(monkeypatch, tmp_path):
    """Regression test for per-source cache isolation: two different sources' caches must be
    separate files, so a corrupted/incompatible cache for one source never affects another (this
    is why per-source, not per-group or one shared file, was chosen)."""
    _mock_transformers(monkeypatch)
    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "FEATURES_DIR", tmp_path)
    news_a = pd.DataFrame({"url": ["a1"], "_text": ["tin nguon A"]})
    news_b = pd.DataFrame({"url": ["b1"], "_text": ["tin nguon B"]})
    ne._get_article_embeddings("source_a", news_a)
    ne._get_article_embeddings("source_b", news_b)

    assert ne._article_cache_path("source_a") != ne._article_cache_path("source_b")
    assert ne._article_cache_path("source_a").exists()
    assert ne._article_cache_path("source_b").exists()
    cache_a = pd.read_parquet(ne._article_cache_path("source_a"))
    assert set(cache_a["url"]) == {"a1"}  # source_b's article never leaked into source_a's cache


# ---------- _load_group branch coverage ----------
def test_load_group_khach_quan_skips_missing_source(monkeypatch):
    from pathlib import Path

    import src.features.news_embeddings as ne

    def _fake_discover():
        return {"cafef": Path("cafef_articles.csv"), "hsc": Path("hsc_articles.csv")}

    def _fake_load_source(source, path):
        if source != "cafef":
            raise FileNotFoundError(source)
        return pd.DataFrame({
            "title": ["VCB tăng"], "lead": ["giá lên"], "pub_date": pd.to_datetime(["2020-01-01"]),
            "url": ["u1"], "source": ["cafef"],
        })

    monkeypatch.setattr(ne, "discover_source_files", _fake_discover)
    monkeypatch.setattr(ne, "load_source", _fake_load_source)
    out = ne._load_group("khach_quan")
    assert set(out["source"]) == {"cafef"}


def test_load_group_tong_hop_missing_file_returns_empty(monkeypatch):
    from pathlib import Path

    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "discover_source_files", lambda: {"ssi": Path("ssi_articles.csv")})

    def _fake_load_source(source, path):
        raise FileNotFoundError("no file")

    monkeypatch.setattr(ne, "load_source", _fake_load_source)
    assert ne._load_group("tong_hop").empty


def test_load_group_missing_url_column_raises(monkeypatch):
    from pathlib import Path

    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "discover_source_files", lambda: {"ssi": Path("ssi_articles.csv")})

    def _fake_load_source(source, path):
        return pd.DataFrame({
            "title": ["VCB tăng"], "lead": ["x"], "pub_date": pd.to_datetime(["2020-01-01"]),
            "source": ["ssi"],
        })

    monkeypatch.setattr(ne, "load_source", _fake_load_source)
    with pytest.raises(ValueError, match="url"):
        ne._load_group("tong_hop")


def test_unclassified_sources(monkeypatch):
    from pathlib import Path

    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "discover_source_files", lambda: {
        "cafef": Path("a.csv"), "ssi": Path("b.csv"), "newsource": Path("c.csv"),
    })
    assert ne.unclassified_sources() == {"newsource"}


# ---------- _build_raw branch coverage ----------
def test_build_raw_no_news_returns_empty(monkeypatch):
    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "_load_group", lambda group: pd.DataFrame())
    assert ne._build_raw("tong_hop").empty


def test_build_raw_no_article_embeddings_returns_empty(monkeypatch):
    import src.features.news_embeddings as ne

    news = pd.DataFrame({"url": ["a"], "_text": ["tin A"], "pub_date": pd.to_datetime(["2020-01-01"]), "source": ["x"]})
    monkeypatch.setattr(ne, "_load_group", lambda group: news)
    monkeypatch.setattr(ne, "_get_article_embeddings", lambda s, n: pd.DataFrame())
    assert ne._build_raw("tong_hop").empty


def test_build_raw_no_merge_match_returns_empty(monkeypatch):
    import src.features.news_embeddings as ne

    news = pd.DataFrame({"url": ["a"], "_text": ["tin A"], "pub_date": pd.to_datetime(["2020-01-01"]), "source": ["x"]})
    other_embs = pd.DataFrame({"url": ["different-url"], "raw_0": [0.1]})
    monkeypatch.setattr(ne, "_load_group", lambda group: news)
    monkeypatch.setattr(ne, "_get_article_embeddings", lambda s, n: other_embs)
    assert ne._build_raw("tong_hop").empty


def test_build_raw_no_ticker_match_returns_empty(monkeypatch):
    import src.features.news_embeddings as ne

    news = pd.DataFrame({
        "url": ["a"], "_text": ["tin không nhắc mã nào"],
        "pub_date": pd.to_datetime(["2020-01-01"]), "source": ["x"],
    })
    embs = pd.DataFrame({"url": ["a"], "raw_0": [0.1]})
    monkeypatch.setattr(ne, "_load_group", lambda group: news)
    monkeypatch.setattr(ne, "_get_article_embeddings", lambda s, n: embs)
    assert ne._build_raw("tong_hop").empty


# ---------- build_group_embeddings / build_comparable_group_embeddings branch coverage ----------
def test_build_group_embeddings_empty_raw_returns_empty(monkeypatch):
    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: pd.DataFrame())
    assert ne.build_group_embeddings("tong_hop").empty


def test_build_comparable_one_group_empty(monkeypatch):
    """One group has data (enough for a real PCA fit), the other is empty -> skipped, not crashed."""
    import src.features.news_embeddings as ne

    rng = np.random.default_rng(0)
    n = 20
    non_empty = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01"] * n), "ticker": ["VCB"] * n, "source": ["cafef"] * n,
        **{f"raw_{i}": rng.normal(size=n) for i in range(ne.RAW_DIM)},
    })

    def _fake_raw(group):
        return non_empty if group == "tong_hop" else pd.DataFrame()

    monkeypatch.setattr(ne, "load_or_build_raw", _fake_raw)
    out = ne.build_comparable_group_embeddings()
    assert out["khach_quan"].empty
    assert not out["tong_hop"].empty


def test_build_comparable_degenerate_small_pool(monkeypatch):
    """Fewer than 2 pooled train-period rows -> falls back to per-group _reduce (no shared PCA)."""
    import src.features.news_embeddings as ne

    small = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]), "ticker": ["VCB"], "source": ["cafef"],  # after TRAIN_CUTOFF
        **{f"raw_{i}": [0.1] for i in range(ne.RAW_DIM)},
    })
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: small)
    out = ne.build_comparable_group_embeddings()
    assert (out["tong_hop"]["pca_applied"] == False).all()  # noqa: E712


# ---------- run() ----------
def test_run_populates_cache_for_both_groups(monkeypatch, tmp_path):
    import src.features.news_embeddings as ne

    monkeypatch.setattr(ne, "FEATURES_DIR", tmp_path)

    def _fake_load_or_build_raw(group):
        # simulate a real call actually writing a per-source cache file to disk
        (tmp_path / f"news_emb_articles_{group}_src.parquet").write_bytes(b"x")
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(ne, "load_or_build_raw", _fake_load_or_build_raw)
    # run() also discovers+logs source files for traceability (Story: processing log) — mock
    # the source module (not `ne`, since run() does a fresh local import) to avoid a real,
    # potentially slow directory scan + writing to the real reports/news_processing_log.md.
    monkeypatch.setattr("src.data.discover_news.discover_source_files", lambda: {})
    monkeypatch.setattr("src.data.discover_news.PROCESSING_LOG_PATH", tmp_path / "log.md")
    written = ne.run()
    assert len(written) == 2


# ---------- _raw_cols (embedding-column detection) ----------
def test_raw_cols_excludes_raw_text_and_raw_path():
    """Regression test: 'raw_text'/'raw_path' are genuine data columns in the objective/
    tier-schema crawl files — a loose 'startswith(\"raw_\")' check wrongly swept them up as
    embedding dimensions, crashing PCA with a string-to-float error. Only raw_<digits> counts."""
    import src.features.news_embeddings as ne

    df = pd.DataFrame({
        "raw_text": ["noi dung bai bao"], "raw_path": ["/some/path"],
        "raw_0": [0.1], "raw_1": [0.2], "raw_10": [0.3], "title": ["t"],
    })
    assert ne._raw_cols(df) == ["raw_0", "raw_1", "raw_10"]
