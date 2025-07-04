# 📈 Stock Alpha-Beta Analyzer & Beta-Hedged Monte Carlo Simulator

A hands-on Streamlit-based web application to explore **beta hedging**, understand **alpha and beta estimation**, and simulate **market-neutral returns** using Monte Carlo techniques.

---

## 🧠 What Is Beta Hedging?

In quantitative finance, **beta hedging** is a technique used to eliminate **systematic risk** (market risk), isolating a stock's **alpha** — the performance not explained by overall market movements.

This app allows you to:
- Estimate a stock’s **beta** relative to the S&P 500
- Visualize how **alpha and beta evolve** over time with rolling regressions
- Calculate **residual returns** (beta-hedged)
- Run **Monte Carlo simulations** to project future performance of a market-neutral position

> 📘 Inspired by Quantopian's Beta Hedging lecture

---

## 🚀 Features

### 🔍 Alpha & Beta Estimation
- Uses **linear regression** on historical returns to estimate:
  - **Alpha** (intercept): stock-specific return
  - **Beta** (slope): sensitivity to the market (S&P 500)
- View **rolling estimates** with customizable window size

### 📉 Beta-Neutral Return Extraction
- Computes **residual returns**:  
  `Hedged Return = Stock Return – Beta × Market Return`
- Helps isolate the **true performance** of the stock, removing market effects

### 🎲 Monte Carlo Simulation
- Simulates beta-hedged return paths based on residuals
- Choose between:
  - **Normal distribution** (mean & std of residuals)
  - **Bootstrap sampling** (resample historical residuals)
- Visualize expected path, confidence intervals, and variability

### 🛠️ Customization Options
- Select stock (`AAPL`, `MSFT`, etc.)
- Choose historical period (`6mo` to `10y`)
- Configure rolling window size, number of simulations, and simulation horizon

---

## 🖥️ Try the App Online

🌐 [Launch the App](https://share.google/205uWhJFdwxZOKLYR)  
📁 [View the Code](https://github.com/HeroicKrishna160905/stock-alpha-beta-app)

> No installation required — just open and run in your browser!

---

## 📷 Screenshots

| Alpha & Beta Regression | Monte Carlo Simulation |
|-------------------------|------------------------|
| ![Regression](./screenshots/rolling_beta.png) | ![Monte Carlo](./screenshots/mc_simulation.png) |

---

## 📚 How It Works (Simplified)

Let:
- `R_stock`: return of the selected stock
- `R_market`: return of S&P 500 (benchmark)

Then:
```math
R_stock = α + β × R_market + ε
