import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

st.set_page_config(layout="wide")
st.title("📈 Stock Alpha-Beta Analyzer + Beta-Hedged Monte Carlo Simulation")

# --- ANALYSIS FORM ---
with st.form("analysis_form"):
    st.subheader("1) Analysis Parameters")
    ticker = st.text_input("Ticker (e.g., AAPL)", "AAPL").upper()
    period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y", "10y"], index=1)
    window = st.slider("Rolling Window Size (Days)", 10, 60, 20)
    run_analysis = st.form_submit_button("Run Analysis")

@st.cache_data(show_spinner=False)
def compute_analysis(ticker, period, window):
    # Download data and compute returns
    df = yf.download(ticker, period=period)
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()

    sp = yf.download("^GSPC", start=df.index.min(), end=df.index.max())
    sp["Daily Return"] = sp["Close"].pct_change()

    returns = pd.DataFrame({
        "Stock": df["Daily Return"],
        "SP500": sp["Daily Return"].reindex(df.index)
    }).dropna()

    # Global regression for alpha & beta
    reg_global = LinearRegression().fit(returns[["SP500"]], returns["Stock"])
    alpha, beta = reg_global.intercept_, reg_global.coef_[0]

    # Compute hedged residuals
    returns["Residual"] = returns["Stock"] - beta * returns["SP500"]

    # Rolling regressions (for diagnostics)
    dates, rolling_alphas, rolling_betas = [], [], []
    for i in range(window, len(returns) + 1):
        sub = returns.iloc[i - window : i]
        r = LinearRegression().fit(sub[["SP500"]], sub["Stock"])
        rolling_alphas.append(r.intercept_)
        rolling_betas.append(r.coef_[0])
        dates.append(sub.index[-1])

    return df, returns, alpha, beta, dates, rolling_alphas, rolling_betas

if run_analysis:
    df, returns, alpha, beta, dates, rolling_alphas, rolling_betas = compute_analysis(ticker, period, window)

    st.subheader("Analysis Results")
    st.write(f"**Global Regression:** {ticker} = {beta:.4f} × SP500 + {alpha:.4f}")

    # Show rolling-window stats
    ra = np.array(rolling_alphas)
    rb = np.array(rolling_betas)
    st.write("#### Rolling Window Statistics")
    c1, c2 = st.columns(2)
    c1.metric("Mean Rolling Alpha", f"{ra.mean():.4f}")
    c1.metric("Std Rolling Alpha", f"{ra.std():.4f}")
    c2.metric("Mean Rolling Beta", f"{rb.mean():.4f}")
    c2.metric("Std Rolling Beta", f"{rb.std():.4f}")

    # Show residual (hedged) performance
    st.write("#### Residual (Beta-Hedged) Return Summary")
    st.dataframe(returns["Residual"].describe())
    cum_hedged = (1 + returns["Residual"]).cumprod().iloc[-1]
    st.metric("Cumulative Hedged Return", f"{cum_hedged:.4f}")

    # Plot rolling alpha & beta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(dates, rb, label="Rolling Beta"); ax1.set_title("Rolling Beta"); ax1.legend(); ax1.grid(True)
    ax2.plot(dates, ra, color="orange", label="Rolling Alpha"); ax2.set_title("Rolling Alpha"); ax2.legend(); ax2.grid(True)
    st.pyplot(fig)

    # Save residuals for simulation
    st.session_state["residuals"] = returns["Residual"].values
    st.session_state["analysis_done"] = True

# --- SIMULATION FORM ---
if st.session_state.get("analysis_done", False):
    with st.form("simulation_form"):
        st.subheader("2) Monte Carlo Simulation Parameters")
        num_simulations = st.number_input("Number of Simulations", min_value=10, max_value=2000, value=100, step=10)
        sim_days = st.number_input("Simulation Length (Days)", min_value=1, max_value=252*5, value=252, step=1)
        initial = st.number_input("Initial Investment ($)", value=1000)
        dist_method = st.selectbox("Sampling Method", ["Normal Approximation", "Bootstrap Historical"], index=0)
        run_sim = st.form_submit_button("Run Simulation")

    if run_sim:
        resid = st.session_state["residuals"]
        mu, sigma = resid.mean(), resid.std()
        paths = np.zeros((num_simulations, sim_days + 1))

        for i in range(num_simulations):
            if dist_method == "Normal Approximation":
                draws = np.random.normal(mu, sigma, sim_days)
            else:
                draws = np.random.choice(resid, size=sim_days, replace=True)
            p = [initial]
            for r in draws:
                p.append(p[-1] * (1 + r))
            paths[i] = p

        mean_path = paths.mean(axis=0)
        std_path = paths.std(axis=0)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for r in paths:
            ax2.plot(r, color='gray', alpha=0.2)
        ax2.plot(mean_path, color='blue', label='Mean Path', linewidth=2)
        ax2.fill_between(
            range(sim_days + 1),
            mean_path - 2 * std_path,
            mean_path + 2 * std_path,
            color='orange', alpha=0.2,
            label='±2 Std Dev'
        )
        ax2.set_title(f"Beta-Hedged Monte Carlo over {sim_days} Days ({num_simulations} Sims)")
        ax2.set_xlabel("Days"); ax2.set_ylabel("Portfolio Value ($)"); ax2.legend(); ax2.grid(True)
        st.pyplot(fig2)

        final_amt = mean_path[-1]
        total_ret = (final_amt / initial - 1) * 100
        st.metric("Mean Final Amount", f"${final_amt:.2f}")
        st.metric("Mean Total Return", f"{total_ret:.2f}%")
