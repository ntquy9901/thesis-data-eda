"""Streamlit dashboard — interactive visualization of all EDA/modeling artifacts.

Run: ``streamlit run src/dashboard/app.py``. Reads ``eda_output/`` (all pages) plus
``crawl_data`` directly for the "Đọc tin tức" raw-article list-view page.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when launched via ``streamlit run`` (no pytest rootdir)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard import data as D  # noqa: E402
from src.eda.common import EDA_OUTPUT_DIR  # noqa: E402

st.set_page_config(page_title="VN Stock News × Volatility", layout="wide")


@st.cache_data(ttl=300)
def _cached(_fn, *args, **kwargs):
    return _fn(*args, **kwargs)


def page_overview() -> None:
    st.header("Overview")
    n_tickers = len(D.available_tickers())
    st.caption(f"{n_tickers} tickers • artifacts from `eda_output/`")

    hm = D.headline_metrics()
    if hm:
        st.subheader("Headline metrics (Ridge, price-only)")
        st.dataframe(pd.DataFrame(hm).T[["r2_price", "rmse_price", "dir_acc", "dm_pvalue"]])

    report = D.load_text("report/eda_final_report.md")
    if report:
        st.subheader("Thesis conclusion (from final report)")
        if "## Thesis Conclusion" in report:
            concl = report.split("## Thesis Conclusion")[1].split("## ")[0]
            st.markdown(concl)
        else:
            st.markdown(report[:2000])


def page_price() -> None:
    st.header("Price EDA")
    tickers = D.available_tickers()
    if not tickers:
        st.warning("No price_metrics artifacts. Run `python -m src.eda.phase03_price_eda`.")
        return
    ticker = st.selectbox("Ticker", tickers, index=0)
    df = D.load_price_metrics(ticker)
    if df.empty:
        st.warning(f"No data for {ticker}")
        return
    df["date"] = pd.to_datetime(df["date"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Close + Parkinson vol")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="close", yaxis="y"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["parkinson_vol"], name="parkinson vol", yaxis="y2"))
        fig.update_layout(yaxis2={"overlaying": "y", "side": "right"}, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Realized vol (20d)")
        import plotly.express as px

        st.plotly_chart(px.line(df, x="date", y="realized_vol_20d", height=380), use_container_width=True)

    st.subheader("Returns + targets")
    cols = [c for c in ["log_returns", "rv_t+1", "rv_t+5", "rv_t+10", "pk_t+1", "pk_t+5", "pk_t+10"] if c in df.columns]
    st.dataframe(df[["date"] + cols].tail(500), height=300)


def page_news() -> None:
    st.header("News EDA")
    sent = D.load_json("news/sentiment_summary.json")
    c1, c2, c3 = st.columns(3)
    if sent:
        c1.metric("Mean sentiment", sent.get("mean"))
        c2.metric("Positive ratio", sent.get("positive_ratio"))
        c3.metric("Negative ratio", sent.get("negative_ratio"))

    sp = D.load_sparse_news()
    if not sp.empty:
        sp["trading_date"] = pd.to_datetime(sp["trading_date"])
        import plotly.express as px

        st.subheader("Daily mean sentiment (across tickers)")
        daily = sp.dropna(subset=["sentiment_mean"]).groupby("trading_date")["sentiment_mean"].mean()
        st.plotly_chart(px.line(daily, height=350), use_container_width=True)

        st.subheader("Mean sentiment by ticker")
        by_t = sp.dropna(subset=["sentiment_mean"]).groupby("ticker")["sentiment_mean"].mean().sort_values()
        st.plotly_chart(px.bar(by_t, height=350), use_container_width=True)

    topics = D.load_json("news/topics.json")
    if topics:
        st.subheader("Topics (top terms)")
        for k, v in list(topics.items())[:7]:
            if isinstance(v, dict):
                st.write(f"**{k}** ({v.get('category', '—')}): {', '.join(v.get('top_terms', [])[:6])}")


def page_news_embedding() -> None:
    st.header("News Embedding (PhoBERT, khách quan vs tổng hợp)")

    stats = D.load_news_embedding_source_stats()
    if stats.empty:
        st.warning("No source_stats.csv. Run `python -m src.eda.phase11_news_embedding_eda`.")
        return
    st.subheader("Source stats")
    st.dataframe(stats)

    coverage = D.load_news_embedding_coverage()
    if not coverage.empty:
        st.subheader("Embedding coverage (per group x source)")
        st.dataframe(coverage)

    sim = D.load_json("news_embedding/group_similarity.json")
    if sim:
        st.subheader("Cosine similarity (within vs across groups)")
        st.json(sim)

    scatter_path = EDA_OUTPUT_DIR / "news_embedding" / "group_scatter.png"
    if scatter_path.exists():
        st.subheader("PCA scatter (shared basis, khách quan vs tổng hợp)")
        st.image(str(scatter_path))


def page_embedding_correlation() -> None:
    st.header("Embedding × Price Correlation (linear vs non-linear)")

    corr = D.load_embedding_price_corr()
    if corr.empty or not {"feature", "pearson_r"} <= set(corr.columns):
        st.warning("No (valid) embedding_price_corr.csv. Run `python -m src.eda.phase12_embedding_price_correlation`.")
        return

    summary = D.load_json("news_embedding/embedding_price_corr_summary.json")
    if summary and "note" not in summary:
        c1, c2 = st.columns(2)
        c1.metric("Linear-significant dims", summary.get("linear_significant_count"))
        c2.metric("Non-linear-only-significant dims", summary.get("nonlinear_only_significant_count"))

    st.subheader("Top |Pearson r| (emb_i x target)")
    st.dataframe(corr[corr["feature"] != "emb_norm"].sort_values("pearson_r", key=abs, ascending=False).head(50))

    ext = D.load_extended_horizon_corr()
    if not ext.empty:
        st.subheader("Extended horizons T+15 / T+20 (does news matter further out?)")
        st.caption("emb_norm = L2 norm of the day's mean-pooled embedding (\"news intensity\" proxy).")
        for tgt in sorted(ext["target"].unique()):
            sub = ext[ext["target"] == tgt].reindex(
                ext[ext["target"] == tgt]["pearson_r"].abs().sort_values(ascending=False).index
            ).head(3)
            st.write(f"**{tgt}**")
            st.dataframe(sub[["feature", "pearson_r", "pearson_p", "fdr_pearson", "spearman_r", "fdr_spearman"]])


def page_news_articles() -> None:
    from src.features.news_embeddings import GROUP_SOURCES

    st.header("Đọc tin tức gốc (khách quan vs tổng hợp)")
    st.caption(
        "List view để đọc trực tiếp và kiểm chứng nội dung bài báo, không phải artifact EDA. "
        "Khách quan = cafef, hsc (tin tường thuật). Tổng hợp = ssi, vndirect, vnstock "
        "(công ty chứng khoán tự phân tích/tổng hợp). Hai nhóm loại trừ lẫn nhau."
    )

    group_label = st.radio(
        "Nhóm", ["Khách quan (cafef, hsc)", "Tổng hợp (ssi, vndirect, vnstock)"],
        horizontal=True, key="news_group",
    )
    group = "khach_quan" if group_label.startswith("Khách quan") else "tong_hop"
    source = st.selectbox(
        "Nguồn (hoặc tất cả nguồn trong nhóm)",
        ["(tất cả)"] + sorted(GROUP_SOURCES[group]),
        key="news_source",
    )
    source = None if source == "(tất cả)" else source

    limit = st.slider("Số bài hiển thị", 20, 300, 100, step=20, key="news_limit")
    search = st.text_input("Tìm trong tiêu đề/lead (tuỳ chọn)", key="news_search")

    df = _cached(D.load_articles_list, group, source, limit if not search else None)
    if df.empty:
        st.warning("Không có dữ liệu cho nhóm/nguồn này.")
        return

    if search:
        title_match = df["title"].fillna("").str.contains(search, case=False, na=False)
        lead_match = df["lead"].fillna("").str.contains(search, case=False, na=False) if "lead" in df.columns else False
        df = df[title_match | lead_match].head(limit)
        if df.empty:
            st.warning(f"Không tìm thấy bài viết nào chứa '{search}'.")
            return

    st.caption(f"{len(df)} bài viết")
    for _, row in df.iterrows():
        label = f"[{row.get('source', '')}] {row.get('pub_date', '')} — {row.get('title', '(không có tiêu đề)')}"
        with st.expander(label):
            st.write(row.get("lead", "") or "(không có lead)")
            if row.get("url"):
                st.markdown(f"[Xem bài gốc]({row['url']})")


def page_modeling() -> None:
    st.header("Modeling")
    m = D.load_metrics()
    if m.empty:
        st.warning("No metrics.csv. Run `python -m src.modeling.baseline`.")
        return
    st.subheader("Metrics (model × feature_set × target)")
    st.dataframe(m[["target", "model", "feature_set", "rmse", "r2", "qlike", "dir_acc"]])

    import plotly.express as px

    st.subheader("R² by target × feature_set (Ridge vs GBM)")
    st.plotly_chart(px.bar(m, x="target", y="r2", color="feature_set", barmode="group",
                           facet_row="model", height=450, title="R² by target"),
                    use_container_width=True)


def page_significance() -> None:
    st.header("Statistical Significance")
    sig = D.load_significance()
    if not sig:
        st.warning("No significance.json. Run `python -m src.modeling.significance`.")
        return

    st.subheader("Diebold-Mariano (price vs +news_adv)")
    rows = []
    for target, v in sig.get("per_target", {}).items():
        dm = v.get("dm", {})
        boot = v.get("bootstrap", {})
        rows.append({"target": target, "dm_pvalue": dm.get("dm_pvalue"),
                     "ΔR² CI low": boot.get("delta_r2_ci", [None])[0],
                     "ΔR² CI high": boot.get("delta_r2_ci", [None, None])[1]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)
        import plotly.express as px

        df["significant"] = df["dm_pvalue"] < 0.05
        st.plotly_chart(px.bar(df, x="target", y="dm_pvalue", color="significant",
                               height=350, title="DM p-value (log not needed; <0.05 = significant)"),
                        use_container_width=True)

    st.subheader("Per-ticker ΔR² heterogeneity (news helps if >0)")
    het_rows = []
    for target, d in sig.get("per_ticker_delta_r2", {}).items():
        for ticker, delta in d.items():
            het_rows.append({"target": target, "ticker": ticker, "delta_r2": delta})
    if het_rows:
        hdf = pd.DataFrame(het_rows)
        import plotly.express as px

        st.plotly_chart(px.bar(hdf, x="ticker", y="delta_r2", color="target", height=400,
                               title="ΔR² per ticker (price+news_adv − price)"),
                        use_container_width=True)


def page_novelty_correlation() -> None:
    st.header("Phase 13: Novelty-based Correlation")
    st.caption("How does article novelty (never-seen-before topics) correlate with future volatility?")

    nov_corr = D.load_novelty_price_corr()
    if nov_corr.empty:
        st.warning("No novelty_price_corr.csv. Run `python -m src.eda.phase13_novelty_correlation`.")
        return

    st.subheader("Correlation summary")
    avg_r = nov_corr["pearson_r"].mean()
    st.metric("Mean Pearson r", f"{avg_r:.4f}")

    st.subheader("Top correlations by target")
    for tgt in sorted(nov_corr["target"].unique()):
        sub = nov_corr[nov_corr["target"] == tgt].sort_values("pearson_r", key=abs, ascending=False)
        st.write(f"**{tgt}** ({len(sub)} features)")
        st.dataframe(sub[["feature", "pearson_r", "pearson_p", "fdr_pearson"]].head(10))


def page_uncertainty_index() -> None:
    st.header("Phase 14: Uncertainty Index")
    st.caption("Articles containing uncertain language patterns (conditional, hedged, speculative).")

    unc_corr = D.load_uncertainty_price_corr()
    if unc_corr.empty:
        st.warning("No uncertainty_price_corr.csv. Run `python -m src.eda.phase14_uncertainty_index`.")
        return

    unc_summary = D.load_json("uncertainty/uncertainty_price_corr_summary.json")
    if unc_summary:
        c1, c2 = st.columns(2)
        c1.metric("Uncertain articles", unc_summary.get("n_uncertain_articles"))
        c2.metric("Mean r (unc×vol)", f"{unc_summary.get('mean_correlation', 0):.4f}")

    st.subheader("Correlation: Uncertainty content → Future volatility")
    st.dataframe(unc_corr[["target", "pearson_r", "pearson_p", "fdr_pearson"]].drop_duplicates("target"))

    st.subheader("By target (detailed)")
    for tgt in sorted(unc_corr["target"].unique()):
        sub = unc_corr[unc_corr["target"] == tgt].sort_values("pearson_r", ascending=False)
        st.write(f"**{tgt}**")
        st.dataframe(sub[["pearson_r", "pearson_p", "fdr_pearson"]].head(5))


def page_temporal_decay() -> None:
    st.header("Phase 15: Temporal Decay of Embedding Signal")
    st.caption("How does the news embedding signal decay over time? Fit: exponential halflife model.")

    decay_corr = D.load_decay_price_corr()
    if decay_corr.empty:
        st.warning("No decay_price_corr.csv. Run `python -m src.eda.phase15_temporal_decay_correlation`.")
        return

    st.subheader("Correlation strength by time lag")
    import plotly.express as px

    st.dataframe(decay_corr[["target", "lag_days", "pearson_r", "fdr_pearson"]])

    # Visualize decay by target
    try:
        st.plotly_chart(
            px.line(decay_corr, x="lag_days", y="pearson_r", color="target", height=400,
                   title="Embedding signal decay over days (halflife model)"),
            use_container_width=True
        )
    except Exception:
        st.warning("Could not plot decay curve; check data format.")


PAGES = {
    "Overview": page_overview,
    "Price EDA": page_price,
    "News EDA": page_news,
    "News Embedding": page_news_embedding,
    "Embedding Correlation": page_embedding_correlation,
    "Novelty Correlation": page_novelty_correlation,
    "Uncertainty Index": page_uncertainty_index,
    "Temporal Decay": page_temporal_decay,
    "Đọc tin tức": page_news_articles,
    "Modeling": page_modeling,
    "Significance": page_significance,
}


def main() -> None:
    st.sidebar.title("VN News × Volatility")
    choice = st.sidebar.radio("Page", list(PAGES.keys()), key="page")
    PAGES[choice]()


if __name__ == "__main__":
    main()
