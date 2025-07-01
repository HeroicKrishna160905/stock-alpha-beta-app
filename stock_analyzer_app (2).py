
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
    # Download data
    data = yf.download(ticker, period=period)
    if data.empty:
        st.error("No data found for the ticker.")
    else:
        # Calculate returns
        data['Daily Return'] = data['Close'].pct_change()
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
        # Summary
        st.write("#### Summary Statistics:")
        st.dataframe(data['Daily Return'].describe())
        st.metric("Cumulative Return (Last Day)", f"{data['Cumulative Return'].iloc[-1]:.4f}")
        # Market benchmark
        sp500 = yf.download("^GSPC", start=data.index.min(), end=data.index.max())
        sp500['Daily Return'] = sp500['Close'].pct_change()
        # Align
        returns_df = pd.DataFrame({
            ticker: data['Daily Return'].values,
            'SP500': sp500['Daily Return'].reindex(data.index).values
        }, index=data.index).dropna()
        # Regression
        X = returns_df[['SP500']].values
        y = returns_df[ticker].values
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        alpha = reg.intercept_
        st.write(f"#### Regression Equation:\n**{ticker} returns = {beta:.4f} × S&P 500 returns + {alpha:.4f}**")
        # Rolling regression
        rolling_betas, rolling_alphas, rolling_dates = [], [], []
        for i in range(window, len(returns_df)+1):
            sub = returns_df.iloc[i-window:i]
            r = LinearRegression().fit(sub[['SP500']], sub[ticker])
            rolling_betas.append(r.coef_[0])
            rolling_alphas.append(r.intercept_)
            rolling_dates.append(sub.index[-1])
        # Plot rolling beta & alpha
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,6))
        ax1.plot(rolling_dates, rolling_betas, label=f'Rolling Beta ({window}d)')
        ax1.set_title(f'{window}-Day Rolling Beta')
        ax1.grid(True); ax1.legend()
        ax2.plot(rolling_dates, rolling_alphas, color='orange', label=f'Rolling Alpha ({window}d)')
        ax2.set_title(f'{window}-Day Rolling Alpha')
        ax2.grid(True); ax2.legend()
        st.pyplot(fig)
        # Stats
        rb = np.array(rolling_betas); ra = np.array(rolling_alphas)
        st.write("#### Rolling Coeff Stats")
        c1, c2 = st.columns(2)
        c1.metric("Beta Mean", f"{rb.mean():.4f}"); c1.metric("Beta Std", f"{rb.std():.4f}")
        c2.metric("Alpha Mean", f"{ra.mean():.4f}"); c2.metric("Alpha Std", f"{ra.std():.4f}")
        # Tests
        sh_b = stats.shapiro(rb); sh_a = stats.shapiro(ra)
        t_b = stats.ttest_1samp(rb,0); t_a = stats.ttest_1samp(ra,0)
        def runs_test(x):
            med = np.median(x); runs=n1=n2=0; prev=None
            for xi in x:
                cur = 1 if xi>=med else 0
                n1+=cur; n2 += (cur==0)
                if prev is None or cur!=prev: runs+=1
                prev=cur
            exp = (2*n1*n2)/(n1+n2)+1 if n1+n2>0 else 0
            std = np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1))) if n1+n2-1>0 else 0
            z=(runs-exp)/std if std>0 else 0; p=2*(1-stats.norm.cdf(abs(z)))
            return runs,exp,z,p
        rt_b = runs_test(rb); rt_a = runs_test(ra)
        st.write("#### Statistical Tests")
        st.write(f"- Shapiro Beta p: {sh_b.pvalue:.4f}, Alpha p: {sh_a.pvalue:.4f}")
        st.write(f"- T-test Beta p: {t_b.pvalue:.4f}, Alpha p: {t_a.pvalue:.4f}")
        st.write(f"- Runs Beta z: {rt_b[2]:.2f}, p: {rt_b[3]:.4f}")
        st.write(f"- Runs Alpha z: {rt_a[2]:.2f}, p: {rt_a[3]:.4f}")
        # Monte Carlo Simulation
        if simulate:
            st.subheader("🔁 Monte Carlo on Alpha")
            years = st.slider("Horizon (Years)", 1,10,5)
            initial = st.number_input("Initial ($)",1000)
            sims, nw = 100, years
            ap_mean, ap_std = ra.mean(), ra.std()
            paths = np.zeros((sims, nw+1))
            for i in range(sims):
                sims_al = np.random.normal(ap_mean, ap_std, nw)
                p=[initial]
                for a in sims_al: p.append(p[-1]*(1+a)**window)
                paths[i]=p
            mean_p = paths.mean(0); std_p = paths.std(0)
            fig2 = plt.figure(figsize=(12,6))
            for r in paths: plt.plot(r, color='gray', alpha=0.2)
            plt.plot(mean_p, color='blue', linewidth=2, label='Mean Path')
            plt.fill_between(range(nw+1), mean_p-2*std_p, mean_p+2*std_p, color='orange', alpha=0.2, label='±2 Std Dev')
            plt.title(f"Simulated Compounded Amount over {years} Years ({sims} Sims)")
            plt.xlabel("Window"); plt.ylabel("Amount ($)"); plt.legend(); plt.grid(True)
            st.pyplot(fig2)
            final=mean_p[-1]; ret=(final/initial-1)*100
            st.metric("Mean Final Amount", f"${final:.2f}")
            st.metric("Mean Total Return", f"{ret:.2f}%")
