import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

st.set_page_config(layout="wide")
st.title("📈 Stock Alpha-Beta Analyzer + Monte Carlo Simulation")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", value="AAPL").upper()
period = st.selectbox("Select Data Period:", options=["6mo", "1y", "2y", "5y", "10y"], index=1)
window = st.slider("Rolling Window Size (Days):", min_value=10, max_value=60, value=20)
simulate = st.checkbox("Run Monte Carlo Simulation using Alpha")

if st.button("Run Analysis"):
    st.write(f"### Analysis for: {ticker} over {period}")
    data = yf.download(ticker, period=period)
    if data.empty:
        st.error("No data found for the ticker.")
    else:
        # Calculate returns & summary
        data['Daily Return'] = data['Close'].pct_change()
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
        st.write("#### Summary Statistics:")
        st.dataframe(data['Daily Return'].describe())
        st.metric("Cumulative Return (Last Day)", f"{data['Cumulative Return'].iloc[-1]:.4f}")

        # Benchmark
        sp500 = yf.download("^GSPC", start=data.index.min(), end=data.index.max())
        sp500['Daily Return'] = sp500['Close'].pct_change()

        # Align returns
        returns_df = pd.DataFrame({
            ticker: data['Daily Return'].values,
            'SP500': sp500['Daily Return'].reindex(data.index).values
        }, index=data.index).dropna()

        # Alpha & Beta regression
        X, y = returns_df[['SP500']].values, returns_df[ticker].values
        reg = LinearRegression().fit(X, y)
        beta, alpha = reg.coef_[0], reg.intercept_
        st.write(f"#### Regression Equation:\n**{ticker} returns = {beta:.4f} × S&P 500 returns + {alpha:.4f}**")

        # Rolling regression
        rolling_betas, rolling_alphas, rolling_dates = [], [], []
        for i in range(window, len(returns_df) + 1):
            sub = returns_df.iloc[i - window:i]
            r = LinearRegression().fit(sub[['SP500']], sub[ticker])
            rolling_betas.append(r.coef_[0])
            rolling_alphas.append(r.intercept_)
            rolling_dates.append(sub.index[-1])

        # Plot rolling Beta & Alpha
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
        ax1.plot(rolling_dates, rolling_betas, label=f'Rolling Beta ({window}d)')
        ax1.set_title(f'{window}-Day Rolling Beta'); ax1.grid(True); ax1.legend()
        ax2.plot(rolling_dates, rolling_alphas, color='orange', label=f'Rolling Alpha ({window}d)')
        ax2.set_title(f'{window}-Day Rolling Alpha'); ax2.grid(True); ax2.legend()
        st.pyplot(fig)

        # Summary stats & tests
        rb, ra = np.array(rolling_betas), np.array(rolling_alphas)
        st.write("#### Rolling Coeff Stats")
        c1, c2 = st.columns(2)
        c1.metric("Beta Mean", f"{rb.mean():.4f}"); c1.metric("Beta Std", f"{rb.std():.4f}")
        c2.metric("Alpha Mean", f"{ra.mean():.4f}"); c2.metric("Alpha Std", f"{ra.std():.4f}")

        sh_b, sh_a = stats.shapiro(rb), stats.shapiro(ra)
        t_b, t_a = stats.ttest_1samp(rb, 0), stats.ttest_1samp(ra, 0)
        def runs_test(x):
            med = np.median(x); runs = n1 = n2 = 0; prev = None
            for xi in x:
                cur = 1 if xi >= med else 0
                n1 += cur; n2 += (cur == 0)
                if prev is None or cur != prev: runs += 1
                prev = cur
            exp = (2 * n1 * n2) / (n1 + n2) + 1 if n1 + n2 > 0 else 0
            std = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))) if n1 + n2 - 1 > 0 else 0
            z = (runs - exp) / std if std > 0 else 0; p = 2 * (1 - stats.norm.cdf(abs(z)))
            return runs, exp, z, p
        rt_b, rt_a = runs_test(rb), runs_test(ra)

        st.write("#### Statistical Tests")
        st.write(f"- **Shapiro-Wilk**: Beta p={sh_b.pvalue:.4f}, Alpha p={sh_a.pvalue:.4f}")
        st.write(f"- **T-test**: Beta p={t_b.pvalue:.4f}, Alpha p={t_a.pvalue:.4f}")
        st.write(f"- **Runs Test**: Beta z={rt_b[2]:.2f}, p={rt_b[3]:.4f}; Alpha z={rt_a[2]:.2f}, p={rt_a[3]:.4f}")

        # Monte Carlo Simulation
        if simulate:
            st.subheader("🔁 Monte Carlo on Alpha")
            num_simulations = st.number_input("Number of Simulations", min_value=10, max_value=2000, value=100, step=10)
            sim_steps = st.slider("Simulation Length (Number of Windows)", min_value=1, max_value=60, value=10)
            initial = st.number_input("Initial Investment ($)", value=1000)

            ap_mean, ap_std = ra.mean(), ra.std()
            paths = np.zeros((num_simulations, sim_steps + 1))
            for i in range(num_simulations):
                sims_alpha = np.random.normal(ap_mean, ap_std, sim_steps)
                p = [initial]
                for a in sims_alpha:
                    p.append(p[-1] * (1 + a) ** window)
                paths[i] = p

            mean_p = paths.mean(axis=0); std_p = paths.std(axis=0)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            for r in paths: ax2.plot(r, color='gray', alpha=0.2)
            ax2.plot(mean_p, color='blue', linewidth=2, label='Mean Path')
            ax2.fill_between(range(sim_steps + 1), mean_p - 2*std_p, mean_p + 2*std_p, alpha=0.2, label='±2 Std Dev')
            ax2.set_title(f"Simulated Compounded Amount over {sim_steps} Steps ({num_simulations} Sims)")
            ax2.set_xlabel("Window Step"); ax2.set_ylabel("Amount ($)"); ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)

            final_amt = mean_p[-1]; total_ret = (final_amt / initial - 1) * 100
            st.metric("Mean Final Amount", f"${final_amt:.2f}")
            st.metric("Mean Total Return", f"{total_ret:.2f}%")

