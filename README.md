<h1 align="center"> Bayesian Estimation of Sentiment Impact on Stock Prices</h1>

<p align="center">
  <i>Latest headlines → VADER sentiment → Bayesian Student-t regression (PyMC) → next-day log-return and multi-day price forecasts with uncertainty.</i>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B.svg?logo=streamlit&logoColor=white">
  <img alt="PyMC" src="https://img.shields.io/badge/PyMC-Bayesian%20inference-3776AB.svg">
  <img alt="Status" src="https://img.shields.io/badge/Release-v1.0.0-success.svg">
</p>

**Authors**  
Shreemadhi Babu Rajendra Prasad (24207575) · Saipavan Narayanasamy (24233785) 
*M.Sc. in Data & Computational Science, University College Dublin*

**Poster:** [Project Poster](./poster/final_project_poster_A0.pdf)

---

## Table of Contents
- [Overview](#overview)
- [Demo / Screenshots](#demo--screenshots)
- [What the app does](#what-the-app-does)
- [Model in one slide](#model-in-one-slide)
- [Install & Run](#install--run)
- [Using the app](#using-the-app)
- [Outputs & Run Log](#outputs--run-log)
- [Project Structure](#project-structure)
- [About the project](#about-the-project)
- [Limitations & future work](#limitations--future-work)
- [Tech stack](#tech-stack)
- [License](#license)
- [Cite](#cite)
- [Acknowledgments](#acknowledgments)
- [Maintainers](#maintainers)

---

## Overview

We build a small research app that:

1) pulls the latest news headlines per ticker,  
2) scores each headline with **NLTK VADER** (compound),  
3) aggregates to a **daily sentiment signal** \(z_t\), and  
4) fits a **Bayesian Student-t regression** (with PyMC) for **next-day log-return** and **3-day price forecasts**, reporting **94% HDIs** for parameters and **90% prediction intervals** (PIs) for prices.

---

## Workflow overview

<p align="center">
  <img src="outputs/flowchart.png"
       alt="End-to-end workflow: data → sentiment → model → forecasts"
       width="300">
</p>

## Demo / Screenshots

### Forecast charts

<p align="center">
  <img src="outputs/tsla_forecast.png" alt="TSLA — Price forecast (next 3 days)" width="48%">
  <img src="outputs/aapl_forecast.png" alt="AAPL — Price forecast (next 3 days)" width="48%">
</p>

### Comparison
<img src="outputs/comparison_history_vs_forecast.png"
     alt="Comparison around 'Today': history vs mean forecast">


---

## What the app does

- **Headlines → sentiment**: For each ticker, fetch recent public headlines and score with **VADER** (compound). Average by day to create \(z_t\).
- **Bayesian regression**: Fit a **Student-t** regression of next-day log-return on **yesterday’s** sentiment \(z_{t-1}\) (lag-1). Heavy tails robustify against outliers.
- **Uncertainty first-class**: Report **94% HDIs** for \(\alpha,\beta,\sigma,\nu\) and **90% PIs** for predicted prices.
- **Forecasts**: Produce next-3-day price forecast table and chart.
- **Comparison**: Side-by-side **β (sentiment effect)** table across two tickers + **indexed history vs mean forecast** plot.
- **Reproducible logging**: Append each run to a local CSV at `results/predictions_log.csv` (kept out of Git by `.gitignore`).

---

## Model in one slide

We model **daily log-returns** with heavy tails:

$$
r_t \;=\; \alpha + \beta\, z_{t-1} + \varepsilon_t,
\qquad
\varepsilon_t \sim \text{Student-t}(\nu, 0, \sigma)
$$

- $r_t$: next-day log-return  
- $z_{t-1}$: yesterday’s (lag-1) **VADER** daily average  
- Parameters $(\alpha,\beta,\sigma,\nu)$ are inferred with **PyMC** (NUTS).  
- **β answers:** _does yesterday’s sentiment move tomorrow’s return?_  
- Price forecasts are obtained by transforming simulated log-return paths to prices.

---

## Install & Run

> [Python **3.9+** recommended](https://www.python.org/downloads/) · [Docs](https://docs.python.org/3.9/)

### 1) Create & activate a virtual environment

**Windows**
~~~bash
python -m venv venv
venv\Scripts\activate
~~~

**macOS/Linux**
~~~bash
python -m venv venv
source venv/bin/activate
~~~

### 2) Install packages
~~~bash
pip install -r requirements.txt
~~~

### 3) One-time: download VADER lexicon used by NLTK
~~~bash
python -m nltk.downloader vader_lexicon
~~~

### 4) Launch the app
~~~bash
streamlit run app/streamlit_app.py
~~~

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

---

## Using the app

**Inputs**
- **Ticker A** (required) and **Ticker B** (optional)
- **Run Speed:** _Fast / Standard / Accurate_ (controls MCMC draws / tuning)

**Tabs**
- **Ticker 1 / Ticker 2:** company blurb, **Latest Headlines**, **Predicted Log-Return**, **Next Price (90% PI)**, **Today’s Sentiment**, **Posterior Summary**, **3-day Price Forecast** (table + chart).
- **Comparison:** quick table of **\(\beta\)** (mean + 94% HDI) and **Day-1 price forecast**; **indexed history vs mean forecast** dots.
- **Run log:** a banner displays whether the log CSV was **Created** or **Appended**. You can also download the log directly from the UI.

---

## Outputs & Run Log

**Figures & tables shown in the UI**
- **Predicted log-return** (mean + 94% HDI)  
- **Next price** (mean + 90% PI)  
- **Posterior summary** for \(\alpha, \beta, \sigma, \nu\) with diagnostics (**ESS**, **\(\hat{R}\)**)  
- **3-day price forecast:** `(day_ahead, price_mean, price_p05, price_p95)` + chart

**Run log CSV:** `results/predictions_log.csv` _(local; ignored by Git)_  
Contains timestamp, tickers, posterior summaries and key forecast numbers (including **day-ahead price mean** and **PI endpoints**).  
Useful for auditing, comparisons across runs, and lightweight experimentation.

---

## Project structure
```text
project/
├─ app/
│  └─ streamlit_app.py           # Streamlit UI + modeling/forecast glue
├─ poster/
│  └─ final_project_poster_A0.pdf
├─ literature/                   # (optional) add literature-review PDFs here
├─ outputs/                      # screenshots/exports (optional; .gitkeep added)
├─ results/
│  └─ predictions_log.csv        # run log (local; ignored by Git)
├─ requirements.txt
└─ README.md
├─ requirements.txt
└─ README.md
```

---

## About the project

**Goal.** Turn daily headlines into a **quantitative sentiment signal** and measure its **predictive effect** on **next-day returns**; produce **uncertainty-aware price forecasts** over short horizons.

**Why Student-t?** Heavy-tailed residuals guard against outliers and volatility clustering common in returns.

**Why Bayesian?** Full posteriors + diagnostics (ESS, $\hat{R}$) + calibrated prediction intervals.

**Why Streamlit?** A fast, transparent interface to explore data, diagnostics, and forecasts.

---

## Limitations & future work

- Predictability may be **weak/noisy**; real-world alpha is hard.  
- Headline sampling & VADER rules can bias the signal — try **domain-tuned** or **LLM** sentiment.  
- Extend to **multivariate** models (market/sector factors), **hierarchical** priors, or **state-space** models with **stochastic volatility**.  
- **Evaluation:** add rolling backtests; CRPS/quantile loss for PIs; compare with AR/ARX/GARCH baselines.  
- Scheduled data refresh, richer news sources, and caching.

> **Disclaimer:** For research/education only — not financial advice.

---

## Tech stack

**Python · Streamlit · PyMC · ArviZ · NumPy · pandas · Matplotlib · NLTK (VADER) · requests/bs4 · yfinance**

---

## License

**MIT** — see [`LICENSE`](LICENSE).

---

## Cite

If you reference this project:

> Narayanasamy, S.; Rajendra Prasad, S.B. (2025). _Bayesian Estimation of Sentiment Impact on Stock Prices_. Version 1.0.0. MIT License. Poster: `poster/final_project_poster_A0.pdf`.

---

## Acknowledgments

- **VADER** sentiment (NLTK)  
- **PyMC / ArviZ** for Bayesian modeling & diagnostics  
- Public headline sources used by the app; Yahoo price data  
- **UCD — ACM40960 Projects in Maths Modelling**

---

## Maintainers

- **Sai pavan Narayanasamy** — 24235785  
- **Shreemadhi Babu Rajendra Prasad** — 24207575
