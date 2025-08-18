# streamlit_app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf

# Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Bayesian
import pymc as pm
import arviz as az

# ====================== Page / Styles ======================
st.set_page_config(page_title="Stock News Sentiment & Bayesian Forecast", layout="wide")

st.markdown("""
<style>
:root{ --accent:#ff4d4f; --text:#1f2937; --muted:#6b7280; --card:#ffffff; --rule:rgba(0,0,0,.06); }
@media (prefers-color-scheme: dark){
  :root{ --text:#e5e7eb; --muted:#9ca3af; --card:#111418; --rule:rgba(255,255,255,.08); }
}
html, body, [class*="css"]{ font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
.header { background: linear-gradient(90deg, #2E86C1 0%, #1B4F72 100%); color: white; padding: 18px 22px; border-radius: 14px; margin-bottom: 14px; box-shadow: 0 6px 20px rgba(30, 64, 175, 0.25); }
.header .title { font-size: 28px; font-weight: 800; margin: 0; }
.header .subtitle { font-size: 14px; opacity: .95; margin: 6px 0 0; }
.kpi-row { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:10px 0 4px; }
.kpi { background:var(--card); border:1px solid var(--rule); border-radius:14px; padding:14px 16px; box-shadow:0 2px 8px rgba(0,0,0,.05); }
.kpi .label { font-size:12px; color:var(--muted); margin-bottom:6px; }
.kpi .value { font-size:20px; font-weight:700; color:var(--text); }
.kpi .delta { font-size:12px; color:var(--muted); margin-top:2px; }
.news-item { margin:6px 0; }
.chip { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; border:1px solid var(--rule); margin-right:8px; }
.positive{ background:#eafaf1; color:#16a34a; } .neutral{ background:#f4f6f7; color:#9ca3af; } .negative{ background:#fdecea; color:#ef4444; }
.news-link a { color:#1b4f72; text-decoration:none; font-weight:600; }
.news-link a:hover { text-decoration:underline; }
.section-head{ margin:12px 0 8px; }
.section-head .label{ font-weight:800; font-size:18px; color:var(--text); }
.section-head .rule{ height:4px; background:var(--accent); border-radius:3px; margin-top:6px; }
.divider{ height:1px; background:var(--rule); margin:14px 0; }
thead tr th { font-weight:700!important; }
.stSlider [role="slider"]{ background:var(--accent)!important; border:2px solid rgba(255,255,255,.5); }
.stSlider [data-baseweb="slider"] > div:nth-child(2) > div{ background:var(--accent)!important; }
div[data-baseweb="input"] svg{ color:var(--accent)!important; }
@media (max-width:1100px){ .kpi-row{ grid-template-columns:1fr; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <div class="title">Stock News Sentiment & Bayesian Forecast</div>
  <div class="subtitle">Latest headlines → sentiment → Bayesian regression (Student-t) → next-day return & multi-day price forecast with uncertainty.</div>
</div>
""", unsafe_allow_html=True)

# ---------- Matplotlib theme ----------
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": "#9ca3af",
    "grid.color": "#9ca3af",
    "grid.linestyle": "-",
    "grid.alpha": 0.2,
    "axes.labelcolor": "#6b7280",
    "axes.titlecolor": "#e5e7eb",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "legend.frameon": True,
    "legend.fancybox": True,
})

def style_legend(ax):
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    if leg:
        fr = leg.get_frame()
        fr.set_facecolor("#f8fafc")
        fr.set_edgecolor("#e5e7eb")
        fr.set_alpha(0.9)
        for t in leg.get_texts():
            t.set_color("#111827")

def section(label: str):
    st.markdown(f"""
    <div class="section-head">
      <div class="label">{label}</div>
      <div class="rule"></div>
    </div>
    """, unsafe_allow_html=True)

# ====================== Resources ======================
@st.cache_resource
def _get_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()
sia = _get_vader()

# ====================== Utilities ======================
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c) for c in df.columns]
    return df

def _to_naive_day(s):
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(None).dt.floor("D")
    return ts.tz_convert(None).floor("D")

@st.cache_data(ttl=3600)
def resolve_to_ticker(user_text: str) -> tuple[str, str]:
    q = (user_text or "").strip()
    if not q:
        return "", ""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        r = requests.get(url, params={"q": q, "quotesCount": 6, "newsCount": 0, "lang": "en-US"}, timeout=10)
        js = r.json()
        quotes = js.get("quotes", []) if isinstance(js, dict) else []
        for item in quotes:
            sym = (item.get("symbol") or "").strip()
            name = (item.get("shortname") or item.get("longname") or "").strip()
            if sym and name:
                return sym.upper(), f"{name} ({sym.upper()})"
        if quotes:
            sym = (quotes[0].get("symbol") or "").strip()
            if sym:
                return sym.upper(), sym.upper()
    except Exception:
        pass
    if len(q) <= 6 and all(c.isalnum() or c in ".-" for c in q):
        return q.upper(), q.upper()
    return q.upper(), q.upper()

# ====================== Company profile ======================
@st.cache_data(ttl=6*3600, show_spinner=False)
def get_company_blurb(ticker: str) -> dict:
    name = ticker
    sector = industry = ""
    summary = ""
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
        if isinstance(info, dict):
            name = info.get("longName") or info.get("shortName") or name
            sector = info.get("sector") or ""
            industry = info.get("industry") or ""
            summary = (info.get("longBusinessSummary") or info.get("description") or "").strip()
    except Exception:
        pass
    return {"name": name, "sector": sector, "industry": industry, "summary": summary}

def render_company_blurb(symbol: str, display_name: str):
    import re
    section("About the company")
    prof = get_company_blurb(symbol)
    name = prof.get("name") or display_name
    meta = " · ".join(x for x in (prof.get("sector"), prof.get("industry")) if x)
    st.markdown(f"**{name}**" + (f" — {meta}" if meta else ""))

    raw = prof.get("summary") or ""
    text = raw.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    bullets = [s.strip() for s in sentences if s.strip()][:3]
    if bullets:
        st.markdown("\n".join(f"- {b}" for b in bullets))
    else:
        st.caption("No profile text available from Yahoo Finance.")

# ====================== Prices (LOG RETURNS) ======================
@st.cache_data(ttl=3600)
def get_prices(ticker: str, lookback_days: int = 180) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=max(lookback_days, 30))
    df = yf.download(
        ticker, start=start.date(), end=end.date(),
        progress=False, auto_adjust=True, group_by="column", interval="1d"
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","Close","return"])
    df = _flatten_columns(df).reset_index().rename(columns={"Date":"date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.floor("D")
    close_col = "Close" if "Close" in df.columns else next((c for c in df.columns if "Close" in c), None)
    if not close_col:
        return pd.DataFrame(columns=["date","Close","return"])
    # LOG return
    df["log_close"] = np.log(df[close_col])
    df["return"] = df["log_close"].diff()
    out = df[["date", close_col, "return"]].dropna().reset_index(drop=True)
    out = out.rename(columns={close_col: "Close"})
    return out

# ====================== News ======================
def _empty_news_df():
    return pd.DataFrame(columns=["source","title","link","published","date"])

@st.cache_data(ttl=1800)
def fetch_google_news(ticker: str, max_articles=80) -> pd.DataFrame:
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    rows=[]
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.content, features="xml")
        for it in soup.find_all("item")[:max_articles]:
            title = (it.title.text or "").strip() if it.title else ""
            link  = (it.link.text or "").strip() if it.link else ""
            pub   = pd.to_datetime((it.pubDate.text if it.pubDate else ""), errors="coerce", utc=True)
            if title:
                rows.append({"source":"google","title":title,"link":link,"published":pub})
    except Exception as e:
        st.warning(f"Google News error: {e}")
    df = pd.DataFrame(rows) if rows else _empty_news_df()
    if not df.empty:
        df["date"] = _to_naive_day(df["published"])
    return df

@st.cache_data(ttl=1800)
def fetch_yahoo_rss(ticker: str, max_articles: int = 60) -> pd.DataFrame:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    rows=[]
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.content, features="xml")
        for it in soup.find_all("item")[:max_articles]:
            title = (it.title.text or "").strip() if it.title else ""
            link  = (it.link.text or "").strip() if it.link else ""
            pub   = pd.to_datetime((it.pubDate.text if it.pubDate else ""), errors="coerce", utc=True)
            if title:
                rows.append({"source":"yahoo","title":title,"link":link,"published":pub})
    except Exception as e:
        st.warning(f"Yahoo RSS error: {e}")
    df = pd.DataFrame(rows) if rows else _empty_news_df()
    if not df.empty:
        df["date"] = _to_naive_day(df["published"])
    return df

@st.cache_data(ttl=1800)
def fetch_gdelt_news(query: str, days: int = 7, max_articles: int = 80) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=max(days,1))
    q = requests.utils.quote(query)
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={q}&mode=artlist&maxrecords={max_articles}&format=HTML&"
        f"STARTDATETIME={start.strftime('%Y%m%d%H%M%S')}&ENDDATETIME={end.strftime('%Y%m%d%H%M%S')}"
    )
    rows=[]
    try:
        r = requests.get(url, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        for row in soup.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 2:
                anchor = cols[1].find("a")
                title = cols[1].get_text(" ", strip=True)
                link  = anchor["href"].strip() if anchor and anchor.has_attr("href") else ""
                time_str = cols[0].get_text(" ", strip=True)
                pub_dt = pd.to_datetime(time_str, errors="coerce", utc=True)
                pub_dt = pub_dt.tz_convert(None) if not pd.isna(pub_dt) else pd.NaT
                if title:
                    rows.append({"source":"gdelt","title":title,"link":link,"published":pub_dt})
    except Exception as e:
        st.warning(f"GDELT error: {e}")
    df = pd.DataFrame(rows) if rows else _empty_news_df()
    if not df.empty:
        df["date"] = _to_naive_day(df["published"])
    return df

def score_news(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news is None or df_news.empty:
        return pd.DataFrame(columns=["date","source","title","link","score","published"])
    df = df_news.copy()
    df["title"] = df["title"].fillna("").str.strip()
    df["link"]  = df["link"].fillna("").str.strip()
    df = df[df["title"] != ""]
    df = df.drop_duplicates(subset=["title", "link"], keep="first")
    df["score"] = df["title"].apply(lambda t: sia.polarity_scores(t)["compound"])
    if "date" not in df.columns:
        df["date"] = pd.NaT
    df["date"] = _to_naive_day(df["date"])
    return df[["date","source","title","link","score","published"]]

def build_daily_sentiment(scored: pd.DataFrame, days_back: int = 60) -> pd.DataFrame:
    if scored is None or scored.empty:
        return pd.DataFrame(columns=["date","sentiment"])
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_back)
    df = scored[scored["date"] >= cutoff]
    daily = df.groupby("date", as_index=False)["score"].mean().rename(columns={"score":"sentiment"})
    return daily

# ====================== Align & Model Frame ======================
def make_model_frame(prices: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["date","Close","return","sentiment","sent_lag1"])
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None).dt.floor("D")
    if daily_sent is not None and not daily_sent.empty:
        s = daily_sent.copy()
        s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None).dt.floor("D")
        s = s.drop_duplicates("date")
        df = p.merge(s, on="date", how="left")
        df["sentiment"] = df["sentiment"].fillna(0.0)
    else:
        df = p.copy()
        df["sentiment"] = 0.0
    df["sent_lag1"] = df["sentiment"].shift(1)
    df = df.dropna(subset=["return","sent_lag1"]).reset_index(drop=True)
    return df

# ====================== Bayesian (Student-t) ======================
def fit_bayes_t_std(x, y, draws=1000, tune=1000, target_accept=0.92, seed=0):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x_mean = float(np.mean(x)); x_std = float(np.std(x)) or 1.0
    zx = (x - x_mean) / x_std
    with pm.Model() as m:
        alpha = pm.Normal("alpha", mu=0.0, sigma=0.02)
        beta  = pm.Normal("beta",  mu=0.0, sigma=0.05)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        nu    = pm.Exponential("nu", 1/10)
        mu    = alpha + beta * zx
        pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                          target_accept=target_accept, progressbar=False,
                          random_seed=int(seed))
    return idata, x_mean, x_std

def predict_next_day_price(idata, last_close: float, x_today: float, x_mean: float, x_std: float):
    post = az.extract(idata, var_names=["alpha","beta","sigma","nu"], group="posterior")
    alpha = post["alpha"].to_numpy()
    beta  = post["beta"].to_numpy()
    sigma = post["sigma"].to_numpy()
    nu    = np.maximum(post["nu"].to_numpy(), 2.0)

    zx = (x_today - x_mean) / (x_std if x_std != 0 else 1.0)
    mu = alpha + beta * zx                          # expected log-return
    rng = np.random.default_rng(0)
    t_sample = rng.standard_t(df=nu, size=len(mu))
    ret_draws = mu + sigma * t_sample               # log-return draws

    # 94% HDI for log-return
    hdi_low, hdi_high = az.hdi(ret_draws, hdi_prob=0.94).astype(float)
    mean_ret = float(np.mean(ret_draws))

    # Map to price
    pred_mean = last_close * float(np.exp(mean_ret))
    pred_low  = last_close * float(np.exp(hdi_low))
    pred_high = last_close * float(np.exp(hdi_high))
    return mean_ret, (hdi_low, hdi_high), pred_mean, pred_low, pred_high

# ===== Monte-Carlo price paths (log-return simulation, uses z-scored sentiment) =====
def forecast_price_paths(idata, last_close: float, sentiment_today: float,
                         x_mean: float, x_std: float,
                         days: int = 3, sims: int = 4000, seed: int = 0):
    """
    Use the same z-scaling as the fitted model:
      z = (sentiment_today - x_mean) / x_std
      r ~ StudentT(nu, mu = alpha + beta * z, sigma)   # r = log-return
      price_{t+1} = price_t * exp(r)
    """
    post = az.extract(idata, var_names=["alpha", "beta", "sigma", "nu"], group="posterior")
    alpha = post["alpha"].to_numpy()
    beta  = post["beta"].to_numpy()
    sigma = post["sigma"].to_numpy()
    nu    = np.maximum(post["nu"].to_numpy(), 2.0)

    z = (float(sentiment_today) - float(x_mean)) / (float(x_std) if x_std != 0 else 1.0)

    rng = np.random.default_rng(seed)
    draws = len(alpha)
    take = min(int(sims), int(draws))
    idx = rng.choice(np.arange(draws), size=take, replace=False)

    alpha = alpha[idx]; beta = beta[idx]; sigma = sigma[idx]; nu = nu[idx]
    mu = alpha + beta * z                             # expected log-return

    prices = np.full((take, days), np.nan, dtype=float)
    p = np.full(take, float(last_close), dtype=float)

    for d in range(days):
        t_sample = rng.standard_t(df=nu, size=take)
        r = mu + sigma * t_sample                     # log-return
        p = p * np.exp(r)
        prices[:, d] = p

    rows = []
    for d in range(days):
        col = prices[:, d]
        rows.append({
            "day_ahead": d + 1,
            "price_mean": float(np.mean(col)),
            "price_p05":  float(np.percentile(col, 5)),
            "price_p95":  float(np.percentile(col, 95)),
        })
    return pd.DataFrame(rows)

# ---------- Forecast plot ----------
def render_price_bands(name: str, R: dict):
    fc = R["forecast_prices"].copy()
    last_close = float(R["px"]["Close"].iloc[-1])

    xs = np.arange(0, len(fc) + 1)
    labels = ["Today"] + [f"D+{d}" for d in fc["day_ahead"]]
    means = np.array([last_close] + fc["price_mean"].tolist())
    lows  = np.array([last_close] + fc["price_p05"].tolist())
    highs = np.array([last_close] + fc["price_p95"].tolist())

    yerr = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots()
    ax.errorbar(xs, means, yerr=yerr, fmt="o-", capsize=5, elinewidth=2, label="Mean price + 90% PI", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{name} — Price forecast (next {len(fc)} days)")
    ax.fill_between(xs, lows, highs, alpha=0.10, zorder=1)

    y_min = float(np.min(lows)); y_max = float(np.max(highs))
    pad = 0.04 * max(1.0, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    style_legend(ax)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Vertical bars show the 90% prediction interval at each horizon from Monte-Carlo price paths.")

# ----------- CSV log helpers -----------
def _rows_from_result(ticker_name: str, R: dict, run_ts: datetime) -> pd.DataFrame:
    beta_mean = float(R["summary"].loc["beta", "mean"])
    base = {
        "run_ts_utc": run_ts.isoformat(timespec="seconds"),
        "ticker": ticker_name,
        "last_close": float(R["px"]["Close"].iloc[-1]),
        "today_sentiment": float(R["s_today"]),
        "mean_ret": float(R["mean_ret"]),
        "hdi_low": float(R["hdi_low"]),
        "hdi_high": float(R["hdi_high"]),
        "pred_price_mean": float(R["pred_price_mean"]),
        "pred_price_low": float(R["pred_price_low"]),
        "pred_price_high": float(R["pred_price_high"]),
        "beta_mean": beta_mean,
    }
    rows = []
    for _, r in R["forecast_prices"].iterrows():
        row = base.copy()
        row.update({
            "day_ahead": int(r["day_ahead"]),
            "price_mean_path": float(r["price_mean"]),
            "price_p05": float(r["price_p05"]),
            "price_p95": float(r["price_p95"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def _upsert_log_csv(csv_path: str, new_rows: pd.DataFrame) -> tuple[str, bool]:
    created = not os.path.exists(csv_path)
    if created:
        new_rows.to_csv(csv_path, index=False)
        return csv_path, True
    old = pd.read_csv(csv_path)
    combo = pd.concat([old, new_rows], ignore_index=True)
    combo.drop_duplicates(subset=["run_ts_utc", "ticker", "day_ahead"], keep="last", inplace=True)
    combo.to_csv(csv_path, index=False)
    return csv_path, False

# ====================== Sidebar Controls ======================
with st.sidebar:
    section("Inputs")
    user_a = st.text_input("Ticker A", value="", help="Enter a stock ticker (e.g., UL) or type a company name.")
    resolved_a, display_a = resolve_to_ticker(user_a)
    if resolved_a and display_a and display_a != resolved_a:
        st.caption(f"Resolved A → {display_a}")

    user_b = st.text_input("Optional: Ticker B", value="", help="Another ticker for comparison (e.g., TSLA).")
    resolved_b, display_b = resolve_to_ticker(user_b) if user_b else ("", "")
    if resolved_b and display_b and display_b != resolved_b:
        st.caption(f"Resolved B → {display_b}")

    section("Run Speed")
    preset = st.radio("Sampling Preset", ["Fast", "Standard", "Accurate"], index=1)

    with st.expander("Advanced (optional)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            sent_days = st.slider("Sentiment lookback (news days)", 5, 30, 14, 1)
            price_lookback = st.slider("Price lookback (days)", 90, 365, 180, 5)
            forecast_days = st.slider("Forecast horizon (days)", 1, 10, 3, 1)
            rough_only = st.checkbox("Comparison: show mean only (no intervals)", True)
        with c2:
            if preset == "Fast":
                d_default, t_default, ta_default = 600, 600, 0.90
            elif preset == "Accurate":
                d_default, t_default, ta_default = 1500, 1500, 0.95
            else:
                d_default, t_default, ta_default = 1000, 1000, 0.92
            draws = st.number_input("MCMC draws", 300, 4000, d_default, 100)
            tune  = st.number_input("MCMC tune", 300, 4000, t_default, 100)
            target_accept = st.slider("Target accept", 0.80, 0.99, ta_default, 0.01)
            seed = st.number_input("Random seed", 0, 10000, 0, 1)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        days_before = st.slider("Comparison: days BEFORE today", 10, 90, 30, 5)
        normalize_base = st.checkbox("Normalize both companies to 100 at 'today'", True)

    go = st.button("Analyze & Predict", use_container_width=True)

# ====================== Run helpers & UI ======================
def run_full_for_one(tick: str):
    g = fetch_google_news(tick); y = fetch_yahoo_rss(tick); d = fetch_gdelt_news(tick, days=sent_days)
    news_all = pd.concat([g, y, d], ignore_index=True)

    # Deduplicate before scoring
    if not news_all.empty:
        news_all["title"] = news_all["title"].fillna("").str.strip()
        news_all["link"]  = news_all["link"].fillna("").str.strip()
        news_all = news_all[news_all["title"] != ""]
        news_all = news_all.drop_duplicates(subset=["title", "link"], keep="first")

    scored = score_news(news_all)
    daily  = build_daily_sentiment(scored, days_back=max(45, sent_days))

    px = get_prices(tick, lookback_days=price_lookback)
    if px.empty:
        return {"error": "price"}

    model_df = make_model_frame(px, daily)
    if model_df.empty or len(model_df) < 20:
        return {"error": "overlap"}

    today_date = pd.Timestamp.today().normalize()
    s_today = scored.loc[scored["date"] == today_date, "score"].mean() if not scored.empty else np.nan
    if np.isnan(s_today):
        if not daily.empty:
            s_today = float(daily["sentiment"].tail(3).mean())
            source_note = "No headlines today — used 3-day average sentiment."
        else:
            s_today = 0.0
            source_note = "No headlines — treating sentiment as neutral (0)."
    else:
        source_note = "Computed from today's headlines."

    idata, x_mean, x_std = fit_bayes_t_std(
        model_df["sent_lag1"].values, model_df["return"].values,
        draws=int(draws), tune=int(tune), target_accept=float(target_accept), seed=int(seed)
    )

    last_close = float(px["Close"].iloc[-1])
    mean_ret, (hdi_low, hdi_high), pred_price_mean, pred_price_low, pred_price_high = \
        predict_next_day_price(idata, last_close, x_today=s_today, x_mean=x_mean, x_std=x_std)

    fc_prices = forecast_price_paths(
        idata, last_close=last_close, sentiment_today=float(s_today),
        x_mean=x_mean, x_std=x_std, days=int(forecast_days), sims=2000, seed=int(seed)
    )

    summary = az.summary(idata, var_names=["alpha","beta","sigma","nu"])

    # Quick numeric sanity check
    trace = idata
    alpha_v = float(trace.posterior['alpha'].mean().item())
    beta_v  = float(trace.posterior['beta'].mean().item())
    sigma_v = float(trace.posterior['sigma'].mean().item())
    nu_v    = float(trace.posterior['nu'].mean().item())
    z_t     = float((s_today - x_mean) / (x_std if x_std else 1.0))
    mu_v    = alpha_v + beta_v * z_t
    sanity = {"alpha": alpha_v, "beta": beta_v, "sigma": sigma_v, "nu": nu_v,
              "z_today": z_t, "mu_today": mu_v}

    return {
        "scored": scored, "daily": daily, "px": px, "model_df": model_df,
        "s_today": s_today, "source_note": source_note,
        "idata": idata, "x_mean": x_mean, "x_std": x_std,
        "mean_ret": mean_ret, "hdi_low": hdi_low, "hdi_high": hdi_high,
        "pred_price_mean": pred_price_mean, "pred_price_low": pred_price_low, "pred_price_high": pred_price_high,
        "forecast_prices": fc_prices,
        "summary": summary, "sanity": sanity
    }

def kpi_cards(mean_ret, hdi_low, hdi_high, price_mean, price_low, price_high, today_sent):
    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="kpi"><div class="label">Predicted Log-Return (mean, 94% HDI)</div>
      <div class="value">{mean_ret*100:.2f}%</div>
      <div class="delta">{hdi_low*100:.2f}% … {hdi_high*100:.2f}%</div></div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="kpi"><div class="label">Next Price (mean, 94% HDI)</div>
      <div class="value">${price_mean:,.2f}</div>
      <div class="delta">${price_low:,.2f} … ${price_high:,.2f}</div></div>
    """, unsafe_allow_html=True)
    chip = "positive" if today_sent > 0.05 else "negative" if today_sent < -0.05 else "neutral"
    st.markdown(f"""
      <div class="kpi"><div class="label">Today's Sentiment</div>
      <div class="value"><span class="chip {chip}">{today_sent:+.3f}</span></div>
      <div class="delta">VADER headline compound</div></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_headlines(scored, name):
    section(f"Latest Headlines — {name}")
    if scored is None or scored.empty:
        st.info("No recent headlines found — treating sentiment as neutral (0).")
        return
    latest = scored.sort_values(["published","date"], ascending=False).head(12)
    for _, row in latest.iterrows():
        cls = "positive" if row['score'] > 0.05 else "negative" if row['score'] < -0.05 else "neutral"
        title = row['title']; link = row.get('link') or ""
        item = f"<span class='news-link'><a href='{link}' target='_blank'>{title}</a></span>" if link else f"<b>{title}</b>"
        st.markdown(f"<div class='news-item'><span class='chip {cls}'>{row['score']:+.3f}</span> {item}</div>", unsafe_allow_html=True)
    st.caption("Headlines are scored with VADER (−1 to +1). We average scores by day to create the daily sentiment signal.")

def render_conclusion(summary_df, ticker_label: str):
    beta_row = summary_df.loc["beta"]
    hdi_cols = [c for c in summary_df.columns if "hdi" in c]
    hdi_low = float(beta_row[hdi_cols[0]]); hdi_high = float(beta_row[hdi_cols[-1]])
    beta_mean = float(beta_row["mean"])
    crosses_zero = (hdi_low <= 0.0 <= hdi_high)
    claim = (f"β (sentiment effect) for {ticker_label} ≈ {beta_mean:+.3f}, "
             f"94% HDI [{hdi_low:+.3f}, {hdi_high:+.3f}]. ")
    if crosses_zero:
        st.info(claim + "Interval includes 0 → no robust predictive signal from daily news sentiment for next-day returns.")
    else:
        st.success(claim + "Interval excludes 0 → sentiment shows a statistically meaningful association with next-day returns.")

def render_posterior_and_forecast(name, R):
    section("Posterior Summary")
    st.dataframe(R["summary"], use_container_width=True, height=240)
    render_conclusion(R["summary"], name)

    with st.expander("Quick numeric sanity check", expanded=False):
        s = R["sanity"]
        st.write(
            f"alpha = {s['alpha']:+.6f} | "
            f"beta = {s['beta']:+.6f} | "
            f"sigma = {s['sigma']:.6f} | "
            f"nu = {s['nu']:.3f}"
        )
        st.write(
            f"z_t (today's sentiment z-score) = {s['z_today']:+.4f} → "
            f"mu (expected log-return) = {s['mu_today']:+.6f} per day"
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section(f"{name}: Next {len(R['forecast_prices'])}-Day Price Forecast")
    st.dataframe(R["forecast_prices"], use_container_width=True, height=160)
    st.caption("Mean price and 90% prediction interval at each horizon from Monte-Carlo simulated paths.")
    render_price_bands(name, R)

def render_full_tab(tab, name, R, symbol=None):
    with tab:
        if symbol:
            render_company_blurb(symbol, name)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        render_headlines(R["scored"], name)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        kpi_cards(R["mean_ret"], R["hdi_low"], R["hdi_high"],
                  R["pred_price_mean"], R["pred_price_low"], R["pred_price_high"], R["s_today"])
        st.caption(R["source_note"])
        render_posterior_and_forecast(name, R)

# ----------- Comparison (history vs rough forecast only) -----------
def render_event_comparison(tick_a, R1, tick_b, R2, days_before=30, normalize=True):
    section("Side-by-Side: Before vs After Today")
    if (R1 is None) or (R2 is None):
        st.info("Need two valid tickers to compare."); return
    px_a = R1["px"].copy(); px_b = R2["px"].copy()
    last_date = min(px_a["date"].max(), px_b["date"].max())
    start_a = last_date - pd.Timedelta(days=days_before)
    a_hist = px_a[(px_a["date"] >= start_a) & (px_a["date"] <= last_date)]
    b_hist = px_b[(px_b["date"] >= start_a) & (px_b["date"] <= last_date)]

    def _mk_series(df, label):
        s = df.set_index("date")["Close"].copy()
        s = s.reindex(pd.date_range(start_a, last_date, freq="D")).ffill()
        offsets = (s.index - last_date).days
        return pd.Series(s.values, index=offsets, name=label)

    sa = _mk_series(a_hist, tick_a); sb = _mk_series(b_hist, tick_b)

    # mean (rough) forecast only
    def _forward_mean(R):
        fc = R["forecast_prices"]
        vals = [float(R["px"]["Close"].iloc[-1])] + fc["price_mean"].tolist()
        offs = list(range(0, len(vals)))  # 0 = today, 1..H = future days
        return pd.Series(vals, index=offs)

    fa = _forward_mean(R1); fb = _forward_mean(R2)

    if normalize:
        base_a = sa.loc[0] if 0 in sa.index else sa.iloc[-1]
        base_b = sb.loc[0] if 0 in sb.index else sb.iloc[-1]
        sa = sa / base_a * 100.0; sb = sb / base_b * 100.0
        fa = fa / fa.loc[0] * 100.0; fb = fb / fb.loc[0] * 100.0
        ylab = "Indexed Price (=100 at Today)"
    else:
        ylab = "Price ($)"

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(sa.index, sa.values, label=f"{tick_a} (past)")
    ax.plot(sb.index, sb.values, label=f"{tick_b} (past)")
    ax.plot(fa.index[1:], fa.values[1:], linestyle="--", marker="o", label=f"{tick_a} (forecast mean)")
    ax.plot(fb.index[1:], fb.values[1:], linestyle="--", marker="o", label=f"{tick_b} (forecast mean)")
    ax.axvline(0, color="#9ca3af", linestyle=":", linewidth=1)
    ax.set_xlabel("Days from Today (− = before, + = after)")
    ax.set_ylabel(ylab)
    ax.set_title("Comparison around 'Today': history vs mean forecast")
    ax.grid(True, alpha=0.25)
    style_legend(ax)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Solid lines: past prices. Dashed dots: rough mean forecast (intervals hidden).")

# ====================== Run ======================
if go:
    ticker = resolved_a or user_a.strip().upper()
    t2 = resolved_b or (user_b.strip().upper() if user_b else "")

    if not ticker:
        st.error("Please enter a valid Ticker A."); st.stop()

    with st.spinner(f"Collecting data and fitting for Ticker 1 ({ticker})…"):
        R1 = run_full_for_one(ticker)
    if "error" in R1:
        st.error(f"Not enough data for the first ticker ({ticker}) — prices or overlap. Try increasing lookback or check the symbol.")
        st.stop()

    R2 = None
    if t2:
        with st.spinner(f"Collecting data and fitting for Ticker 2 ({t2})…"):
            R2 = run_full_for_one(t2)

    run_ts = datetime.utcnow()
    log_rows = _rows_from_result(ticker, R1, run_ts)
    if t2 and (R2 is not None) and ("error" not in R2):
        log_rows = pd.concat([log_rows, _rows_from_result(t2, R2, run_ts)], ignore_index=True)

    CSV_PATH = "results/predictions_log.csv"
    csv_path, created = _upsert_log_csv(CSV_PATH, log_rows)
    (st.success if created else st.info)(f"{'Created' if created else 'Appended to'} log CSV → {csv_path}")
    try:
        st.download_button("Download run log", data=open(csv_path, "rb").read(),
                           file_name="predictions_log.csv", mime="text/csv")
    except Exception:
        pass

    tabs = st.tabs([f"Ticker 1 — {ticker}", f"Ticker 2 — {t2 if t2 else '—'}", "Comparison"])

    render_full_tab(tabs[0], display_a or ticker, R1, ticker)

    if not t2:
        with tabs[1]:
            st.info("Enter a second ticker to see its full details here.")
    elif "error" in (R2 or {}):
        with tabs[1]:
            st.error("Not enough data for the second ticker (prices or overlap).")
    else:
        render_full_tab(tabs[1], display_b or t2, R2, t2)

    with tabs[2]:
        if not t2 or (R2 is not None and "error" in R2):
            st.info("Provide a valid second ticker to view comparisons.")
        else:
            section("Beta (sentiment effect) & Day-1 forecast — quick table")
            def row_from(R, name):
                summ = R["summary"]
                beta_mean = float(summ.loc["beta","mean"])
                hdis = [c for c in summ.columns if "hdi_" in c]
                beta_low = float(summ.loc["beta", hdis[0]]) if hdis else np.nan
                beta_high = float(summ.loc["beta", hdis[-1]]) if hdis else np.nan
                d1 = R["forecast_prices"].iloc[0]
                return {
                    "ticker": name,
                    "beta_mean": beta_mean,
                    "beta_hdi_low": beta_low,
                    "beta_hdi_high": beta_high,
                    "d1_price_mean": float(d1["price_mean"]),
                    "d1_price_p05": float(d1["price_p05"]),
                    "d1_price_p95": float(d1["price_p95"]),
                }
            cmp_df = pd.DataFrame([row_from(R1, display_a or ticker), row_from(R2, display_b or t2)])
            st.dataframe(cmp_df, use_container_width=True)
            st.caption("β measures how strongly yesterday’s sentiment moves today’s **log-return**. Day-1 is the very next trading day.")
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            render_event_comparison(display_a or ticker, R1, display_b or t2, R2,
                                    days_before=int(days_before), normalize=bool(normalize_base))
