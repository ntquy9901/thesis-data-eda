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


def _insight(text: str) -> None:
    """Consistent 💡 explanation/suggestion box placed under every chart/table."""
    st.info(f"💡 {text}")


def page_overview() -> None:
    st.header("Overview")
    n_tickers = len(D.available_tickers())
    st.caption(f"{n_tickers} tickers • artifacts from `eda_output/`")

    hm = D.headline_metrics()
    if hm:
        st.subheader("Headline metrics (Ridge, price-only)")
        st.dataframe(pd.DataFrame(hm).T[["r2_price", "rmse_price", "dir_acc", "dm_pvalue"]])
        best_t = max(hm, key=lambda t: hm[t].get("r2_price") or -1)
        best_r2 = hm[best_t].get("r2_price")
        best_dir = hm[best_t].get("dir_acc")
        dir_txt = f"{best_dir:.0%}" if best_dir is not None else "?"
        _insight(
            f"Đây là mô hình **chỉ dùng giá** (chưa có tin tức) — baseline để so sánh. "
            f"**R²** = mô hình giải thích được bao nhiêu % biến động thật (càng gần 1 càng tốt, "
            f"gần 0 = gần như đoán mò). **dir_acc** = tỷ lệ đoán đúng chiều tăng/giảm. "
            f"Hiện tốt nhất là **{best_t}** (R²={best_r2}, đoán đúng chiều {dir_txt} số lần). "
            f"Gợi ý: nếu R² của mọi horizon đều thấp (<0.2), giá quá khứ tự nó khó dự báo — "
            f"cần xem thêm tin tức có giúp được không (trang Modeling/Significance)."
        )

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
        _insight(
            "**Parkinson vol** (đường cam) đo độ biến động trong ngày từ khoảng cách giá cao/thấp "
            "(không cần biết giá đóng cửa hôm trước) — đây chính là **biến số mà mô hình cố dự "
            "báo (target)**. Khi đường cam nhảy vọt = ngày đó cổ phiếu biến động mạnh (thường "
            "quanh tin tức lớn hoặc biến cố thị trường). Gợi ý: đối chiếu các đỉnh vol với ngày "
            "có tin tức ở trang 'Đọc tin tức' để kiểm tra bằng mắt xem tin có thực sự gây biến động."
        )
    with c2:
        st.subheader("Realized vol (20d)")
        import plotly.express as px

        st.plotly_chart(px.line(df, x="date", y="realized_vol_20d", height=380), use_container_width=True)
        _insight(
            "Biến động trung bình 20 ngày gần nhất — mượt hơn Parkinson vol hàng ngày, dùng để "
            "thấy xu hướng \"chế độ thị trường\" (biến động cao kéo dài vs. yên ắng kéo dài) thay "
            "vì nhiễu từng ngày. Vol có tính \"cụm\" (volatility clustering): giai đoạn biến động "
            "cao thường kéo dài liên tục — đây là lý do các đặc trưng HAR (trung bình trượt) "
            "hoạt động tốt làm baseline."
        )

    st.subheader("Returns + targets")
    cols = [c for c in ["log_returns", "rv_t+1", "rv_t+5", "rv_t+10", "rv_t+22", "pk_t+1", "pk_t+5", "pk_t+10", "pk_t+22"] if c in df.columns]
    st.dataframe(df[["date"] + cols].tail(500), height=300)
    _insight(
        "`log_returns` = lợi suất log hàng ngày. `pk_t+1/5/10/22` = Parkinson vol **1/5/10/22 ngày sau** "
        "(mục tiêu dự báo chính — mô hình không được nhìn thấy các cột này khi dự đoán, chỉ dùng "
        "để chấm điểm). `rv_t+*` = biến động thực hiện (realized vol) tương lai, một cách đo thay "
        "thế. 500 dòng gần nhất được hiển thị; cuộn bảng để xem chi tiết."
    )


def page_news() -> None:
    st.header("News EDA")
    sent = D.load_json("news/sentiment_summary.json")
    c1, c2, c3 = st.columns(3)
    if sent:
        c1.metric("Mean sentiment", sent.get("mean"))
        c2.metric("Positive ratio", sent.get("positive_ratio"))
        c3.metric("Negative ratio", sent.get("negative_ratio"))
        _insight(
            "Điểm sentiment trung bình (thang -1 = tiêu cực .. +1 = tích cực) trên toàn bộ tin đã "
            "cào. Số gần 0 là bình thường (đa số tin tường thuật trung tính); tỷ lệ positive/"
            "negative cho biết thị trường tin tức đang nghiêng chiều nào tổng thể."
        )

    sp = D.load_sparse_news()
    if not sp.empty:
        sp["trading_date"] = pd.to_datetime(sp["trading_date"])
        import plotly.express as px

        st.subheader("Daily mean sentiment (across tickers)")
        daily = sp.dropna(subset=["sentiment_mean"]).groupby("trading_date")["sentiment_mean"].mean()
        st.plotly_chart(px.line(daily, height=350), use_container_width=True)
        _insight(
            "Sentiment trung bình mỗi ngày, gộp tất cả ticker. Đường dao động quanh 0 = tin tức "
            "đa số trung tính theo ngày; các đỉnh/đáy nhọn thường trùng biến cố lớn (KQKD, chính "
            "sách vĩ mô...). Gợi ý: so sánh hình dạng đường này với biểu đồ vol ở trang 'Price EDA' "
            "để có cảm quan ban đầu — quan hệ định lượng thật sự nằm ở trang 'Significance'."
        )

        st.subheader("Mean sentiment by ticker")
        by_t = sp.dropna(subset=["sentiment_mean"]).groupby("ticker")["sentiment_mean"].mean().sort_values()
        st.plotly_chart(px.bar(by_t, height=350), use_container_width=True)
        if not by_t.empty:
            most_pos, most_neg = by_t.index[-1], by_t.index[0]
            _insight(
                f"Ticker có tin tức tích cực nhất (trung bình): **{most_pos}** "
                f"({by_t.iloc[-1]:.3f}). Tiêu cực nhất: **{most_neg}** ({by_t.iloc[0]:.3f}). "
                f"Đây chỉ là mô tả (descriptive), chưa chứng minh sentiment ảnh hưởng tới giá — "
                f"xem trang 'Significance' cho kiểm định thống kê."
            )

    topics = D.load_json("news/topics.json")
    if topics:
        st.subheader("Topics (top terms)")
        for k, v in list(topics.items())[:7]:
            if isinstance(v, dict):
                st.write(f"**{k}** ({v.get('category', '—')}): {', '.join(v.get('top_terms', [])[:6])}")
        _insight(
            "Chủ đề được trích tự động (NMF topic modeling) từ tiêu đề/lead bài báo — mỗi chủ đề "
            "là 1 nhóm từ hay xuất hiện cùng nhau. Dùng để kiểm tra chất lượng dữ liệu (chủ đề có "
            "hợp lý về mặt tài chính không) hơn là để suy luận thống kê."
        )


def page_news_embedding() -> None:
    st.header("News Embedding (PhoBERT, khách quan vs tổng hợp)")

    stats = D.load_news_embedding_source_stats()
    if stats.empty:
        st.warning("No source_stats.csv. Run `python -m src.eda.phase11_news_embedding_eda`.")
        return
    st.subheader("Source stats")
    st.dataframe(stats)
    _insight(
        "Số bài + tỷ lệ thiếu title/lead theo từng nguồn báo — bảng kiểm tra chất lượng dữ liệu "
        "thô trước khi encode. `group=unclassified` nghĩa là nguồn mới cào chưa được gán vào "
        "khách_quan/tổng_hợp (cần bổ sung vào danh sách phân loại nếu xuất hiện)."
    )

    coverage = D.load_news_embedding_coverage()
    if not coverage.empty:
        st.subheader("Embedding coverage (per group x source)")
        st.dataframe(coverage)
        _insight(
            "`n_embedded_rows` = số dòng (bài báo × ticker được nhắc tới) đã có PhoBERT embedding "
            "và dùng được cho phân tích — luôn NHỎ HƠN nhiều tổng số bài ở bảng trên, vì phần lớn "
            "bài báo không nhắc trực tiếp mã cổ phiếu nào trong VN30. `mean_emb_norm` là độ lớn "
            "trung bình của vector embedding — chỉ số kiểm tra chất lượng encode (giá trị bất "
            "thường/= 0 cảnh báo lỗi encode)."
        )

    sim = D.load_json("news_embedding/group_similarity.json")
    if sim:
        st.subheader("Cosine similarity (within vs across groups)")
        st.json(sim)
        _insight(
            "Cosine similarity đo mức độ giống nhau về ngữ nghĩa (0 = hoàn toàn khác, 1 = giống "
            "hệt). `within_*` càng cao = các bài trong nhóm đó càng viết giống nhau (có thể do tin "
            "ngắn/khuôn mẫu). `across_groups` thấp hơn cả hai `within_*` nghĩa là 2 nhóm nguồn "
            "thực sự viết khác nhau về nội dung — hợp lý khi so sánh 'khách quan' (tường thuật) "
            "với 'tổng hợp' (phân tích của công ty chứng khoán)."
        )

    scatter_path = EDA_OUTPUT_DIR / "news_embedding" / "group_scatter.png"
    if scatter_path.exists():
        col_chart, col_note = st.columns([3, 2])
        with col_chart:
            st.subheader("PCA scatter (shared basis, khách quan vs tổng hợp)")
            st.image(str(scatter_path))
        with col_note:
            st.subheader("Giải thích & nhận xét")
            # Use embedding_coverage (rows actually plotted: ticker-matched + embedded),
            # NOT raw source_stats article counts — those two can diverge a lot (e.g. most
            # khach_quan articles never mention a VN30 ticker by name, so very few make it
            # into the scatter even though the raw article count looks balanced).
            n_kq = int(coverage.loc[coverage["group"] == "khach_quan", "n_embedded_rows"].sum()) if not coverage.empty and "group" in coverage.columns else None
            n_th = int(coverage.loc[coverage["group"] == "tong_hop", "n_embedded_rows"].sum()) if not coverage.empty and "group" in coverage.columns else None
            st.markdown(
                "**Cách tính:** mỗi chấm = 1 bài báo. Text (title+lead) → PhoBERT "
                "[CLS] embedding (768-dim) → PCA lần 1 giảm còn 32-dim (basis dùng chung "
                "cho cả 2 nhóm, fit trên dữ liệu train) → PCA lần 2 giảm còn 2-dim "
                "**chỉ để vẽ** (không dùng cho modeling). Trục PC1/PC2 là hướng phương sai "
                "lớn nhất, không mang ý nghĩa chủ đề trực tiếp."
            )
            if isinstance(sim, dict) and sim:
                w_kq = sim.get("within_khach_quan")
                w_th = sim.get("within_tong_hop")
                w_ac = sim.get("across_groups")
                bullets = []
                if w_kq is not None:
                    tight = "rất chặt (gần trùng nhau)" if w_kq > 0.5 else ("khá tản mạn" if w_kq < 0.2 else "vừa phải")
                    bullets.append(f"- **khach_quan** (similarity nội nhóm = {w_kq}): các bài {tight}.")
                if w_th is not None:
                    tight = "rất chặt (gần trùng nhau)" if w_th > 0.5 else ("khá tản mạn / đa dạng nội dung" if w_th < 0.2 else "vừa phải")
                    bullets.append(f"- **tong_hop** (similarity nội nhóm = {w_th}): các bài {tight}.")
                if w_ac is not None:
                    rel = "khác biệt rõ với" if (w_kq is not None and w_ac < w_kq * 0.5) else "tương đối gần với"
                    bullets.append(f"- Similarity **giữa 2 nhóm** = {w_ac} → nội dung 2 nhóm {rel} similarity nội nhóm khach_quan.")
                st.markdown("\n".join(bullets))
            if n_kq is not None and n_th is not None:
                if min(n_kq, n_th) > 0 and max(n_kq, n_th) / max(1, min(n_kq, n_th)) > 20:
                    st.warning(
                        f"⚠️ Lệch mẫu mạnh trên biểu đồ: khach_quan={n_kq} chấm vs tong_hop={n_th} chấm "
                        "(số dòng đã match ticker + có embedding, khác với tổng số bài báo thô ở bảng trên). "
                        "Nhóm ít điểm (khach_quan) dễ tạo cụm ảo do quá ít mẫu, "
                        "không nên coi là khác biệt ngữ nghĩa đáng tin cậy về mặt thống kê."
                    )


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
        _insight(
            "Mỗi `emb_i` là 1 trong 32 chiều PCA của embedding tin tức (không có ý nghĩa chủ đề "
            "riêng lẻ). 'Linear-significant' = số chiều có tương quan tuyến tính (Pearson) với "
            "target sau khi hiệu chỉnh đa kiểm định (FDR). Nếu 2 con số này nhỏ so với 32 chiều "
            "→ hầu hết chiều embedding KHÔNG liên hệ tuyến tính với biến động giá — không có nghĩa "
            "là vô dụng hoàn toàn (xem thêm Level-1 Significance để có Kendall/MI/dcor)."
        )

    st.subheader("Top |Pearson r| (emb_i x target)")
    st.dataframe(corr[corr["feature"] != "emb_norm"].sort_values("pearson_r", key=abs, ascending=False).head(50))
    top_row = corr[corr["feature"] != "emb_norm"].reindex(
        corr[corr["feature"] != "emb_norm"]["pearson_r"].abs().sort_values(ascending=False).index
    ).head(1)
    if not top_row.empty:
        r0 = top_row.iloc[0]
        _insight(
            f"Tương quan mạnh nhất: **{r0['feature']}** với **{r0['target']}** (r={r0['pearson_r']}). "
            f"|r| thường rất nhỏ (<0.1) trong dữ liệu tài chính hàng ngày — bình thường, không phải "
            f"lỗi. So sánh |r| của các dòng đầu bảng để xem có chiều embedding nào nổi bật hẳn "
            f"không, hay tất cả đều yếu như nhau (= tín hiệu tin tức yếu tổng thể)."
        )

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
        _insight(
            "Kiểm tra xem tín hiệu tin tức có 'ngấm' vào giá chậm hơn T+10 hay không (horizon dài "
            "hơn — 15/20 ngày). Nếu |r| ở đây KHÔNG lớn hơn các horizon ngắn, tín hiệu không mạnh "
            "lên theo thời gian — ủng hộ kết luận 'tin tức là tín hiệu yếu, không phải bị trễ'."
        )


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
    _insight(
        "So sánh nhiều tổ hợp: **feature_set** (price = chỉ giá; +news_basic/+news_adv/"
        "+sentiment5/+event_type = thêm từng loại tin tức) × **model** (ridge = tuyến tính, "
        "gbm = cây quyết định, bắt được quan hệ phi tuyến). **rmse** càng thấp càng tốt; **r2** "
        "càng cao càng tốt. Cách đọc nhanh: so `r2` của `price` với các dòng `price+...` cùng "
        "target/model — nếu `price+X` cao hơn `price` rõ rệt, feature X có đóng góp. Một vài "
        "dòng `qlike` cực lớn (>>1) là dấu hiệu dự đoán gần 0 gây lỗi số học, bỏ qua khi so sánh."
    )

    import plotly.express as px

    st.subheader("R² by target × feature_set (Ridge vs GBM)")
    st.plotly_chart(px.bar(m, x="target", y="r2", color="feature_set", barmode="group",
                           facet_row="model", height=450, title="R² by target"),
                    use_container_width=True)
    _insight(
        "So sánh trực quan R² giữa các feature_set. Nếu các cột `price+...` gần như bằng chiều "
        "cao cột `price` (không cao hơn rõ rệt) ở cả 2 hàng ridge/gbm → thêm tin tức không cải "
        "thiện đáng kể. Kết luận CHẮC CHẮN cần xem thêm kiểm định thống kê (DM test) ở trang "
        "'Significance' — chênh lệch nhỏ trên biểu đồ có thể không có ý nghĩa thống kê."
    )


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
        n_sig = int(df["significant"].sum())
        _insight(
            "**Diebold-Mariano test**: kiểm định xem sai số dự báo của mô hình +news_adv có "
            "THỰC SỰ nhỏ hơn mô hình price-only hay không (không phải chỉ do may rủi). "
            f"p<0.05 (cột xanh trong biểu đồ) = có ý nghĩa thống kê. Hiện tại: **{n_sig}/{len(df)} "
            "horizon** có ý nghĩa. **ΔR² 95% CI** chứa 0 = không chắc chắn tin tức giúp ích; "
            "CI hoàn toàn dương = tin tức giúp ích thật (dù có thể rất nhỏ)."
        )

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
        n_help = int((hdf["delta_r2"] > 0).sum())
        _insight(
            f"Mỗi cột = 1 (ticker, horizon): cột **dương** (trên vạch 0) = tin tức giúp cải "
            f"thiện R² cho riêng mã đó; cột **âm** = tin tức làm mô hình tệ hơn (nhiễu). "
            f"Hiện tin tức giúp ích ở **{n_help}/{len(hdf)}** tổ hợp (ticker×horizon). Nếu số "
            f"này chỉ khoảng 20-30%, kết luận hợp lý là 'tin tức chỉ hữu ích cho một số ít mã "
            f"cổ phiếu cụ thể', không phải hiệu ứng chung cho toàn thị trường."
        )

    st.subheader("Per-family ablation (Story 14-1: sentiment5 / event_type, isolated from news_adv)")
    fam_rows = []
    for fset, per_target in sig.get("per_family", {}).items():
        for target, v in per_target.items():
            dm = v.get("dm", {})
            boot = v.get("bootstrap", {})
            fam_rows.append({
                "feature_set": fset, "target": target, "dm_pvalue": dm.get("dm_pvalue"),
                "ΔR² CI low": boot.get("delta_r2_ci", [None])[0],
                "ΔR² CI high": boot.get("delta_r2_ci", [None, None])[1],
                "significant": (dm.get("dm_pvalue") or 1) < 0.05,
            })
    if fam_rows:
        fam_df = pd.DataFrame(fam_rows)
        st.dataframe(fam_df)
        import plotly.express as px

        st.plotly_chart(
            px.bar(fam_df, x="target", y="dm_pvalue", color="feature_set", barmode="group",
                  height=350, title="DM p-value theo feature family (<0.05 = có ý nghĩa)"),
            use_container_width=True,
        )
        sig_fam = fam_df[fam_df["significant"]]
        if not sig_fam.empty:
            lines = [f"- **{r.feature_set}** tại **{r.target}** (p={r.dm_pvalue})" for r in sig_fam.itertuples()]
            _insight(
                "So sánh RIÊNG từng nhóm feature (tách khỏi embedding gộp `news_adv`) để biết "
                "chính xác nhóm nào đóng góp. Có ý nghĩa thống kê (Ridge):\n" + "\n".join(lines) +
                "\n\nXem bảng GBM ngay bên dưới để kiểm chứng giả thuyết đa cộng tuyến."
            )
        else:
            _insight(
                "Không có family nào đạt ý nghĩa thống kê (p<0.05) ở bảng này — nhất quán với "
                "kết luận tổng thể: tin tức là tín hiệu yếu, không đủ mạnh để cải thiện dự báo "
                "biến động một cách chắc chắn."
            )

    st.subheader("Per-family ablation — GBM (kiểm tra giả thuyết đa cộng tuyến)")
    fam_gbm_rows = []
    for fset, per_target in sig.get("per_family_gbm", {}).items():
        for target, v in per_target.items():
            dm = v.get("dm", {})
            boot = v.get("bootstrap", {})
            fam_gbm_rows.append({
                "feature_set": fset, "target": target, "dm_pvalue": dm.get("dm_pvalue"),
                "ΔR² CI low": boot.get("delta_r2_ci", [None])[0],
                "ΔR² CI high": boot.get("delta_r2_ci", [None, None])[1],
                "significant": (dm.get("dm_pvalue") or 1) < 0.05,
            })
    if fam_gbm_rows:
        fam_gbm_df = pd.DataFrame(fam_gbm_rows)
        st.dataframe(fam_gbm_df)
        all_p1 = (fam_gbm_df["dm_pvalue"].fillna(0) == 1.0).all()
        if all_p1:
            _insight(
                "**Kết luận:** DM p=1.0 tuyệt đối ở MỌI dòng — nghĩa là dự đoán của GBM "
                "**giống hệt nhau về mặt số học** dù có hay không có sentiment5/event_type "
                "(kể cả dùng RIÊNG LẺ, không chỉ khi gộp chung). Điều này **bác bỏ giả thuyết "
                "đa cộng tuyến**: GBM không hề bị ảnh hưởng bởi tương quan giữa các feature "
                "(cây quyết định tách từng feature độc lập), vậy mà vẫn phớt lờ hoàn toàn cả "
                "sentiment5 lẫn event_type — cho thấy tín hiệu Ridge tìm được ở T+1 (p=0.033) "
                "là tín hiệu **rất nhỏ và có thể mong manh** (chỉ mô hình tuyến tính đơn giản "
                "mới 'nhìn thấy', mô hình cây coi là nhiễu không đáng tách nhánh)."
            )
        else:
            _insight(
                "So với bảng Ridge ở trên: nếu GBM cũng có dòng significant → tín hiệu đủ mạnh "
                "để cả 2 loại mô hình đều phát hiện (đáng tin hơn). Nếu GBM toàn bộ p=1.0 trong "
                "khi Ridge có significant → tín hiệu chỉ là tuyến tính rất nhỏ, không phải quan "
                "hệ mạnh/phi tuyến."
            )


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
    _insight(
        "**Novelty** = 1 trừ độ giống (cosine similarity) với các bài gần nhất cùng ticker trong "
        "5 ngày trước — bài càng 'mới lạ' (không phải tin nhắc lại) thì novelty càng gần 1. "
        "Giả thuyết: tin THỰC SỰ MỚI mới gây biến động, tin lặp lại (rehash) thì không. Mean r "
        f"gần 0 ({avg_r:.4f}) → chưa thấy quan hệ tuyến tính rõ ràng ở mức tổng thể."
    )

    st.subheader("Top correlations by target")
    for tgt in sorted(nov_corr["target"].unique()):
        sub = nov_corr[nov_corr["target"] == tgt].sort_values("pearson_r", key=abs, ascending=False)
        st.write(f"**{tgt}** ({len(sub)} features)")
        st.dataframe(sub[["feature", "pearson_r", "pearson_p", "fdr_pearson"]].head(10))
    _insight(
        "`fdr_pearson=True` = tương quan sống sót sau hiệu chỉnh đa kiểm định (đáng tin hơn p "
        "thô). Lưu ý: novelty của các ngày liên tiếp bị tự tương quan (cùng cửa sổ 5 ngày lặp lại) "
        "nên p-value ở đây nên hiểu là 'khám phá ban đầu' (exploratory), không phải bằng chứng "
        "thống kê chặt chẽ như ở trang Significance."
    )


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
        _insight(
            "Chỉ số kiểu **Baker-Bloom-Davis (EPU)**: 1 bài được tính 'bất định' khi chứa CẢ 3 "
            "nhóm từ khóa (Kinh tế + Chính sách + Bất định/rủi ro) cùng lúc — chỉ đếm keyword, "
            "không dùng AI. Đây là chỉ số 'toàn thị trường theo ngày' (không tách theo từng "
            "ticker), nên so sánh với biến động vol trung bình toàn VN30."
        )

    st.subheader("Correlation: Uncertainty content → Future volatility")
    st.dataframe(unc_corr[["target", "pearson_r", "pearson_p", "fdr_pearson"]].drop_duplicates("target"))

    st.subheader("By target (detailed)")
    for tgt in sorted(unc_corr["target"].unique()):
        sub = unc_corr[unc_corr["target"] == tgt].sort_values("pearson_r", ascending=False)
        st.write(f"**{tgt}**")
        st.dataframe(sub[["pearson_r", "pearson_p", "fdr_pearson"]].head(5))
    _insight(
        "Nghiên cứu gốc BBD (Mỹ) tìm r≈0.73 với VIX — nếu r ở đây thấp hơn nhiều, khả năng: (1) "
        "chỉ số EPU cần lượng tin đủ lớn/nhiều năm mới ổn định, hoặc (2) tin tức tiếng Việt hiện "
        "có chưa đủ bao phủ để phản ánh đúng mức độ bất định thị trường. Không nên kết luận "
        "'phương pháp BBD sai' chỉ từ 1 lần chạy trên tập dữ liệu này."
    )


def page_temporal_decay() -> None:
    st.header("Phase 15: Temporal Decay of Embedding Signal")
    st.caption(
        "Đặc trưng gộp tin N ngày gần nhất với trọng số giảm dần theo thời gian (halflife cố "
        "định = 5 ngày giao dịch — tin hôm qua có trọng số gấp đôi tin 5 ngày trước), sau đó "
        "PCA-reduce còn các chiều `emb_decay_0..31`. KHÔNG có lưới nhiều halflife/lag để so "
        "sánh trong lần chạy này — chỉ 1 halflife cố định."
    )

    decay_corr = D.load_decay_price_corr()
    if decay_corr.empty:
        st.warning("No decay_price_corr.csv. Run `python -m src.eda.phase15_temporal_decay_correlation`.")
        return

    st.subheader("Top correlations by target (emb_decay_i x target)")
    import plotly.express as px

    for tgt in sorted(decay_corr["target"].unique()):
        sub = decay_corr[decay_corr["target"] == tgt].reindex(
            decay_corr[decay_corr["target"] == tgt]["pearson_r"].abs().sort_values(ascending=False).index
        ).head(5)
        st.write(f"**{tgt}**")
        st.dataframe(sub[["feature", "pearson_r", "pearson_p", "fdr_pearson", "spearman_r", "mi"]])
    _insight(
        "Mỗi `emb_decay_i` là 1 chiều PCA của embedding-đã-suy-giảm-theo-thời-gian (không mang "
        "ý nghĩa chủ đề riêng lẻ). So sánh |pearson_r| ở đây với bảng 'Embedding × Price "
        "Correlation' (không suy giảm theo thời gian) — nếu |r| ở đây KHÔNG cao hơn, việc gộp "
        "nhiều ngày tin tức không giúp tăng tín hiệu so với chỉ dùng tin trong ngày."
    )

    st.subheader("Phân phối |Pearson r| theo target")
    try:
        plot_df = decay_corr.copy()
        plot_df["abs_pearson_r"] = plot_df["pearson_r"].abs()
        st.plotly_chart(
            px.box(plot_df, x="target", y="abs_pearson_r", height=400,
                   title="|Pearson r| của 32 chiều emb_decay theo target (halflife=5 ngày)"),
            use_container_width=True,
        )
        _insight(
            "Mỗi box tổng hợp |r| của cả 32 chiều emb_decay cho 1 target. Box càng thấp/hẹp gần "
            "0 = tín hiệu embedding-suy-giảm-theo-thời-gian nhìn chung yếu ở target đó, kể cả sau "
            "khi đã thử gộp nhiều ngày tin tức lại với nhau."
        )
    except Exception:
        st.warning("Could not plot decay distribution; check data format.")


def page_level1_significance() -> None:
    st.header("Phase 17: Level-1 Statistical Significance (Sentiment/Event)")
    st.caption(
        "Positive/Negative/Fear/Optimism/Uncertainty score + Event-type counts, đánh giá bằng "
        "Pearson, Spearman, Kendall Tau, Mutual Information, và Distance Correlation "
        "(theo docs/gpt-guide/news_feature_evaluation_guideline.md)."
    )

    corr = D.load_level1_corr()
    if corr.empty:
        st.warning("No level1_corr.csv. Run `python -m src.eda.phase17_level1_significance`.")
        return

    summary = D.load_json("level1_significance/level1_summary.json")
    if summary and "note" not in summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cặp (feature, target)", summary.get("n_feature_target_pairs"))
        c2.metric("Linear-significant (Pearson)", summary.get("linear_significant_count"))
        c3.metric("MI≈0 → likely useless", summary.get("likely_useless_mi_near_zero_count"))
        st.caption(summary.get("interpretation", ""))
        candidates = summary.get("nonlinear_candidates", [])
        if candidates:
            st.subheader("Nonlinear candidates (Pearson ≈ 0 nhưng MI > 0)")
            st.dataframe(pd.DataFrame(candidates))

    st.subheader("Toàn bộ kết quả (5 phép đo) theo feature")
    feature_choice = st.selectbox("Feature", sorted(corr["feature"].unique()))
    sub = corr[corr["feature"] == feature_choice]
    st.dataframe(sub[[
        "target", "n", "pearson_r", "pearson_p", "fdr_pearson",
        "spearman_r", "fdr_spearman", "kendall_tau", "fdr_kendall", "mi", "dcor",
    ]])
    _insight(
        "5 phép đo, mỗi cái bắt 1 kiểu quan hệ khác nhau: **Pearson** = tuyến tính; **Spearman/"
        "Kendall** = đơn điệu (tăng/giảm đều, không cần thẳng hàng); **MI/dcor** = BẤT KỲ quan hệ "
        "nào kể cả phi tuyến phức tạp (hình chữ U, sóng...). Đọc theo guideline: nếu MI/dcor > 0 "
        "rõ rệt nhưng Pearson ≈ 0 → có quan hệ thật nhưng Pearson 'nhìn không thấy' vì nó không "
        "thẳng hàng. Nếu CẢ 5 đều ≈ 0 → feature này thực sự không liên quan tới target."
    )

    import plotly.express as px

    st.subheader("So sánh MI vs |Pearson r| — phát hiện tín hiệu phi tuyến")
    plot_df = corr.dropna(subset=["mi", "pearson_r"]).copy()
    if not plot_df.empty:
        plot_df["abs_pearson_r"] = plot_df["pearson_r"].abs()
        st.plotly_chart(
            px.scatter(plot_df, x="abs_pearson_r", y="mi", color="feature", hover_data=["target"],
                      height=450, title="MI > 0 & |Pearson r| ≈ 0 → khả năng tín hiệu phi tuyến"),
            use_container_width=True,
        )
        _insight(
            "Mỗi chấm = 1 cặp (feature, target). Chấm nằm **góc trên-trái** (MI cao, |Pearson r| "
            "thấp) là ứng viên tín hiệu phi tuyến đáng điều tra thêm (thử feature đó trong GBM "
            "thay vì Ridge). Chấm nằm **gần góc dưới-trái** (cả 2 đều thấp) = ứng viên loại bỏ, "
            "gần như chắc chắn không mang thông tin dự báo."
        )


def page_event_study_by_type() -> None:
    st.header("Phase 18: Event Study by Type (Level-2)")
    st.caption(
        "Mỗi loại sự kiện (earnings/dividend/M&A/management/regulation/macro/sector) phân tích "
        "riêng biệt: T-5..T0..T+10, abnormal volatility, và Cumulative Abnormal Return (CAR) so "
        "với benchmark thị trường (bình quân log-return toàn bộ VN30)."
    )

    events = D.load_event_study_by_type()
    if events.empty:
        st.warning("No event_study_by_type.csv. Run `python -m src.eda.phase18_event_study_by_type`.")
        return

    summary = D.load_json("event_study_by_type/event_study_by_type_summary.json")

    event_type = st.selectbox("Loại sự kiện", sorted(events["event_type"].unique()))
    horizon = st.selectbox("Horizon (ngày)", sorted(events["horizon"].unique()))
    key = f"{event_type}_h{horizon}"
    if summary and key in summary:
        info = summary[key]
        c1, c2, c3 = st.columns(3)
        c1.metric("Số sự kiện", info.get("n_events"))
        av = info.get("abnormal_vol_ttest", {})
        c2.metric("Mean abnormal vol", av.get("mean"), delta=f"p={av.get('pvalue')}")
        car = info.get("post_car_ttest", {})
        c3.metric("Mean post-CAR", car.get("mean"), delta=f"p={car.get('pvalue')}")
        if av.get("significant") or car.get("significant"):
            st.success("Có ý nghĩa thống kê (p<0.05) — sự kiện này thực sự thay đổi vol/return trung bình.")
        else:
            st.info("Không có ý nghĩa thống kê ở horizon này (p≥0.05).")

    sub = events[(events["event_type"] == event_type) & (events["horizon"] == horizon)]
    st.subheader(f"Chi tiết ({len(sub)} sự kiện)")
    st.dataframe(sub[["ticker", "event_date", "pre_vol", "post_vol", "abnormal_vol", "pre_car", "post_car"]])
    _insight(
        "`abnormal_vol` = vol sau sự kiện trừ vol trước sự kiện (dương = biến động tăng lên sau "
        "tin). `post_car` = Cumulative Abnormal Return — lợi suất cộng dồn SAU khi trừ đi lợi "
        "suất bình quân thị trường (dương = cổ phiếu đó tăng vượt trội so với thị trường chung "
        "sau sự kiện, âm = giảm kém hơn thị trường)."
    )

    import plotly.express as px

    if not sub.empty:
        st.plotly_chart(
            px.histogram(sub, x="abnormal_vol", height=350, title=f"Phân phối abnormal vol — {event_type} (T+{horizon})"),
            use_container_width=True,
        )
        _insight(
            "Nếu phân phối lệch rõ về phía dương (đa số sự kiện làm vol tăng) → loại sự kiện này "
            "có khả năng gây biến động thật. Nếu phân phối đối xứng quanh 0 → hiệu ứng trung bình "
            "triệt tiêu (một số ticker tăng vol, số khác giảm) dù t-test trung bình có thể vẫn "
            "'không ý nghĩa' — nhìn hình dạng phân phối giúp tránh hiểu lầm chỉ từ 1 con số trung "
            "bình."
        )


PAGES = {
    "Overview": page_overview,
    "Price EDA": page_price,
    "News EDA": page_news,
    "News Embedding": page_news_embedding,
    "Embedding Correlation": page_embedding_correlation,
    "Novelty Correlation": page_novelty_correlation,
    "Uncertainty Index": page_uncertainty_index,
    "Temporal Decay": page_temporal_decay,
    "Level-1 Significance": page_level1_significance,
    "Event Study by Type": page_event_study_by_type,
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
