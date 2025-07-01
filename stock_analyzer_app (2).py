import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

st.set_page_config(layout="wide")
st.title("📈 Stock Alpha-Beta Analyzer + Monte Carlo Simulation")

# --- ANALYSIS FORM ---
with st.form("analysis_form"):
    st.subheader("1) Analysis Parameters")
    ticker = st.text_input("Ticker (e.g., AAPL)", "AAPL").upper()
    period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y", "10y"], index=1)
    window = st.slider("Rolling Window Size (Days)", 10, 60, 20)
    run_analysis = st.form_submit_button("Run Analysis")

@st.cache_data(show_spinner=False)
def compute_analysis(ticker, period, window):
    # Download & compute returns
    df = yf.download(ticker, period=period)
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()

    sp = yf.download("^GSPC", start=df.index.min(), end=df.index.max())
    sp["Daily Return"] = sp["Close"].pct_change()

    returns = pd.DataFrame({
        "Stock": df["Daily Return"],
        "SP500": sp["Daily Return"].reindex(df.index)
    }).dropna()

    # Global regression
    X, y = returns[["SP500"]], returns["Stock"]
    reg = LinearRegression().fit(X, y)
    alpha, beta = reg.intercept_, reg.coef_[0]

    # Rolling regression
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

    # Display basic stats
    st.subheader("Analysis Results")
    st.write(f"**Regression:** {ticker} = {beta:.4f} × SP500 + {alpha:.4f}")
    st.write("Daily Return Stats:")
    st.dataframe(df["Daily Return"].describe())
    st.metric("Cumulative Return", f"{df['Cumulative Return'].iloc[-1]:.4f}")

    # Rolling plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(dates, rolling_betas, label="Beta"); ax1.set_title("Rolling Beta"); ax1.legend(); ax1.grid(True)
    ax2.plot(dates, rolling_alphas, color="orange", label="Alpha"); ax2.set_title("Rolling Alpha"); ax2.legend(); ax2.grid(True)
    st.pyplot(fig)

    # Store rolling alphas in session state for simulation
    st.session_state["rolling_alphas"] = rolling_alphas
    st.session_state["window"] = window

# --- SIMULATION FORM ---
if "rolling_alphas" in st.session_state:
    with st.form("sim_form"):
        st.subheader("2) Monte Carlo Parameters")
        num_simulations = st.number_input("Number of Simulations", 10, 2000, 100, step=10)
        sim_steps = st.slider("Simulation Length (Number of Windows)", 1, 60, 10)
        initial = st.number_input("Initial Investment ($)", 1000)
        run_sim = st.form_submit_button("Run Simulation")

    if run_sim:
        ra = np.array(st.session_state["rolling_alphas"])
        w = st.session_state["window"]
        mu, sigma = ra.mean(), ra.std()

        paths = np.zeros((num_simulations, sim_steps + 1))
        for i in range(num_simulations):
            draws = np.random.normal(mu, sigma, sim_steps)
            p = [initial]
            for a in draws:
                p.append(p[-1] * (1 + a) ** w)
            paths[i] = p

        mean_path = paths.mean(axis=0)
        std_path = paths.std(axis=0)

        # Plot
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for r in paths:
            ax2.plot(r, color="gray", alpha=0.2)
        ax2.plot(mean_path, color="blue", label="Mean Path", linewidth=2)
        ax2.fill_between(range(sim_steps + 1),
                         mean_path - 2 * std_path,
                         mean_path + 2 * std_path,
                         color="orange", alpha=0.2,
                         label="±2 Std Dev")
        ax2.set_title(f"Monte Carlo: {num_simulations} Sims, {sim_steps} Steps")
        ax2.set_xlabel("Step"); ax2.set_ylabel("Amount ($)")
        ax2.legend(); ax2.grid(True)
        st.pyplot(fig2)

        st.metric("Mean Final Amount", f"${mean_path[-1]:.2f}")
        st.metric("Mean Total Return", f"{((mean_path[-1]/initial)-1)*100:.2f}%")
