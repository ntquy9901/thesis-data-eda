"""Streamlit dashboard — interactive visualization of all EDA/modeling artifacts.

Run: ``streamlit run src/dashboard/app.py``. Reads ``eda_output/`` only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import data as D

st.set_page_config(page_title="VN Stock News × Volatility", layout="wide")


@st.cache_data(ttl=300)
def _cached(fn, *args, **kwargs):
    return fn(*args, **kwargs)


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


PAGES = {
    "Overview": page_overview,
    "Price EDA": page_price,
    "News EDA": page_news,
    "Modeling": page_modeling,
    "Significance": page_significance,
}


def main() -> None:
    st.sidebar.title("VN News × Volatility")
    choice = st.sidebar.radio("Page", list(PAGES.keys()), key="page")
    PAGES[choice]()


if __name__ == "__main__":
    main()
