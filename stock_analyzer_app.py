
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

st.set_page_config(layout="wide")

st.title("📈 Stock Alpha-Beta Analyzer")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", value="AAPL").upper()
window = st.slider("Rolling Window Size (Days)", min_value=10, max_value=60, value=20)

if st.button("Run Analysis"):

    st.write(f"### Running analysis for: {ticker}")

    data = yf.download(ticker, period="1y")
    if data.empty:
        st.error("No data found for the ticker.")
    else:
        data['Daily Return'] = data['Close'].pct_change()
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

        st.write("#### Summary Statistics:")
        st.dataframe(data['Daily Return'].describe())

        st.metric("Cumulative Return (Last Day)", f"{data['Cumulative Return'].iloc[-1]:.4f}")

        sp500 = yf.download("^GSPC", start=data.index.min(), end=data.index.max())
        sp500['Daily Return'] = sp500['Close'].pct_change()

        returns_df = pd.DataFrame({
            ticker: data['Daily Return'].values,
            'SP500': sp500['Daily Return'].reindex(data.index).values
        }, index=data.index).dropna()

        X = returns_df[['SP500']].values
        y = returns_df[ticker].values
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        alpha = reg.intercept_

        st.write(f"#### Regression Equation:
**{ticker} returns = {beta:.4f} * (S&P 500 returns) + {alpha:.4f}**")

        rolling_betas, rolling_alphas, rolling_dates = [], [], []
        for i in range(window, len(returns_df) + 1):
            sub = returns_df.iloc[i - window:i]
            reg = LinearRegression().fit(sub[['SP500']], sub[ticker])
            rolling_betas.append(reg.coef_[0])
            rolling_alphas.append(reg.intercept_)
            rolling_dates.append(sub.index[-1])

        fig, axs = plt.subplots(2, 1, figsize=(14, 6))
        axs[0].plot(rolling_dates, rolling_betas, label=f'Rolling Beta ({window}d)')
        axs[0].set_title(f'{window}-Day Rolling Beta for {ticker}')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(rolling_dates, rolling_alphas, color='orange', label=f'Rolling Alpha ({window}d)')
        axs[1].set_title(f'{window}-Day Rolling Alpha for {ticker}')
        axs[1].grid(True)
        axs[1].legend()

        st.pyplot(fig)

        rolling_betas = np.array(rolling_betas)
        rolling_alphas = np.array(rolling_alphas)

        st.write("#### Rolling Coefficient Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Rolling Beta Mean", f"{rolling_betas.mean():.4f}")
        col1.metric("Rolling Beta Std", f"{rolling_betas.std():.4f}")
        col2.metric("Rolling Alpha Mean", f"{rolling_alphas.mean():.4f}")
        col2.metric("Rolling Alpha Std", f"{rolling_alphas.std():.4f}")

        beta_shapiro = stats.shapiro(rolling_betas)
        alpha_shapiro = stats.shapiro(rolling_alphas)
        beta_ttest = stats.ttest_1samp(rolling_betas, 0)
        alpha_ttest = stats.ttest_1samp(rolling_alphas, 0)

        def runs_test(x):
            median = np.median(x)
            runs, n1, n2 = 0, 0, 0
            prev = None
            for i in x:
                curr = 1 if i >= median else 0
                if curr == 1: n1 += 1
                else: n2 += 1
                if prev is None or curr != prev:
                    runs += 1
                prev = curr
            expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1 if (n1 + n2) > 0 else 0
            std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))) if (n1 + n2 - 1) > 0 else 0
            z = (runs - expected_runs) / std_runs if std_runs > 0 else 0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            return runs, expected_runs, z, p

        beta_runs = runs_test(rolling_betas)
        alpha_runs = runs_test(rolling_alphas)

        st.write("#### Statistical Tests")
        st.write(f"- **Shapiro-Wilk Normality Test (Beta)**: p = {beta_shapiro.pvalue:.4f}")
        st.write(f"- **Shapiro-Wilk Normality Test (Alpha)**: p = {alpha_shapiro.pvalue:.4f}")
        st.write(f"- **One-sample T-test (Beta ≠ 0)**: p = {beta_ttest.pvalue:.4f}")
        st.write(f"- **One-sample T-test (Alpha ≠ 0)**: p = {alpha_ttest.pvalue:.4f}")
        st.write(f"- **Runs Test (Beta)**: z = {beta_runs[2]:.2f}, p = {beta_runs[3]:.4f}")
        st.write(f"- **Runs Test (Alpha)**: z = {alpha_runs[2]:.2f}, p = {alpha_runs[3]:.4f}")
