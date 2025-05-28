
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import seaborn as sns
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.float_format', '{:.2%}'.format)

CSV_PATHS = {
    'SHI Income': 'SHI_INCOME_28Mai2025.csv',
    'SHI Alpha': 'SHI_ALPHA_28Mai2025.csv'
}

YAHOO_TICKERS = {
    'BW-Bank Potenzial T1': '0P0000J5K3.F',
    'BW-Bank Potenzial T2': '0P0000J5K8.F',
    'BW-Bank Potenzial T4': '0P0000JM36.F',
    'BW-Bank Aktienallokation 75 P Dis': '0P0001HPL2.F'
}

RISK_FREE_RATE = 0.02  # 2% p.a.

def to_1d_series(ret):
    if isinstance(ret, pd.DataFrame):
        ret = ret.iloc[:, 0]
    return pd.to_numeric(ret, errors='coerce').dropna()

def get_yahoo_display_names(yahoo_tickers):
    name_map = {}
    for nickname, ticker in yahoo_tickers.items():
        try:
            info = yf.Ticker(ticker).info
            realname = info.get("shortName") or info.get("longName") or ticker
            display_name = f"{nickname} ({realname})"
        except Exception as e:
            display_name = nickname
        name_map[nickname] = display_name
    return name_map

@st.cache_data(show_spinner="Lade Yahoo-Namen...", ttl=3600)
def cached_display_names_map():
    return get_yahoo_display_names(YAHOO_TICKERS)

def load_returns_from_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = pd.to_numeric(df['Close'], errors='coerce').ffill().dropna()
    returns = close.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def load_returns_from_yahoo(ticker, start, end):
    df = yf.download(ticker, start=start, end=end+timedelta(days=1), progress=False)['Close'].dropna()
    returns = df.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def load_and_sync_data(start, end, display_names_map):
    returns_dict, cumulative_dict = {}, {}

    # CSV
    for name, path in CSV_PATHS.items():
        try:
            ret, cum = load_returns_from_csv(path)
            ret = ret.loc[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]
            cum = cum.loc[(cum.index >= pd.Timestamp(start)) & (cum.index <= pd.Timestamp(end))]
            returns_dict[name] = ret
            cumulative_dict[name] = cum
        except Exception as e:
            st.warning(f"CSV konnte nicht geladen werden: {path} ({e})")

    # Yahoo
    for nickname, ticker in YAHOO_TICKERS.items():
        display_name = display_names_map[nickname]
        try:
            ret, cum = load_returns_from_yahoo(ticker, start, end)
            returns_dict[display_name] = ret
            cumulative_dict[display_name] = cum
        except Exception as e:
            st.warning(f"Yahoo Finance konnte nicht geladen werden: {ticker} ({e})")

    # Schnittmenge der Indizes
    all_indexes = [set(r.index) for r in returns_dict.values() if len(r) > 0]
    if all_indexes:
        common_index = sorted(set.intersection(*all_indexes))
    else:
        common_index = []
    for name in returns_dict:
        returns_dict[name] = returns_dict[name].loc[common_index]
        cumulative_dict[name] = cumulative_dict[name].loc[common_index]
    return returns_dict, cumulative_dict

def sortino_ratio(returns, risk_free=0.0, annualization=252):
    """Annualized Sortino Ratio"""
    # Downside only
    downside = returns[returns < risk_free]
    downside_std = downside.std(ddof=0)
    mean_ret = returns.mean()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    daily_sortino = (mean_ret - risk_free) / downside_std
    return daily_sortino * np.sqrt(annualization)

def omega_ratio(returns, risk_free=0.0):
    """Annäherung Omega Ratio, Schwelle risk_free (meist 0)"""
    gain = (returns > risk_free).sum()
    loss = (returns <= risk_free).sum()
    if loss == 0:
        return np.nan
    return gain / loss

def tail_ratio(returns):
    """p95 / |p5| der Return-Verteilung"""
    try:
        return np.percentile(returns, 95) / abs(np.percentile(returns, 5))
    except Exception:
        return np.nan

def calculate_metrics(returns_dict, cumulative_dict):
    metrics = pd.DataFrame()
    for name in returns_dict:
        ret = returns_dict[name]
        cum = cumulative_dict[name]
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        if ret.empty or cum.empty:
            continue

        days = (cum.index[-1] - cum.index[0]).days
        total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1)
        annual_ret = float((1 + total_ret)**(365/days) - 1) if days > 0 else np.nan
        annual_vol = float(ret.std() * np.sqrt(252))
        sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
        sortino = sortino_ratio(ret, risk_free=0.0)
        drawdowns = (cum / cum.cummax() - 1)
        mdd = float(drawdowns.min()) if not drawdowns.empty else np.nan
        calmar = annual_ret / abs(mdd) if (not np.isnan(mdd) and mdd < 0) else np.nan
        var_95 = float(ret.quantile(0.05))
        cvar_95 = float(ret[ret <= var_95].mean())
        omega = omega_ratio(ret, risk_free=0.0)
        tail = tail_ratio(ret)
        win_rate = float(len(ret[ret > 0]) / len(ret))
        avg_win = float(ret[ret > 0].mean())
        avg_loss = float(ret[ret < 0].mean())
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan
        monthly_ret = ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = float((monthly_ret > 0).mean())

        metrics.loc[name, 'Total Return'] = total_ret
        metrics.loc[name, 'Annual Return'] = annual_ret
        metrics.loc[name, 'Annual Volatility'] = annual_vol
        metrics.loc[name, 'Sharpe Ratio'] = sharpe
        metrics.loc[name, 'Sortino Ratio'] = sortino
        metrics.loc[name, 'Max Drawdown'] = mdd
        metrics.loc[name, 'Calmar Ratio'] = calmar
        metrics.loc[name, 'VaR (95%)'] = var_95
        metrics.loc[name, 'CVaR (95%)'] = cvar_95
        metrics.loc[name, 'Omega Ratio'] = omega
        metrics.loc[name, 'Tail Ratio'] = tail
        metrics.loc[name, 'Win Rate'] = win_rate
        metrics.loc[name, 'Avg Win'] = avg_win
        metrics.loc[name, 'Avg Loss'] = avg_loss
        metrics.loc[name, 'Profit Factor'] = profit_factor
        metrics.loc[name, 'Positive Months'] = positive_months
    return metrics

def plot_performance(cumulative_dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for name, cum in cumulative_dict.items():
        if cum is None or len(cum) == 0:
            continue
        ax.plot(cum.index, cum / cum.iloc[0], label=name, linewidth=1.5)
    ax.set_title("Kumulative Performance (Start = 1.0)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Datum", fontsize=12)
    ax.set_ylabel("Indexierte Entwicklung", fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)

    # Drawdown robust
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for name, cum in cumulative_dict.items():
        if cum is None or len(cum) == 0:
            continue
        drawdown = (cum / cum.cummax()) - 1
        if isinstance(drawdown, pd.DataFrame):
            drawdown = drawdown.iloc[:, 0]
        drawdown = drawdown.dropna()
        if drawdown.empty or len(drawdown) < 2:
            continue
        x = drawdown.index.values
        y = drawdown.values
        ax2.fill_between(x, y, 0, alpha=0.3)
        ax2.plot(x, y, linewidth=1, label=name)
    ax2.set_title("Drawdown-Verlauf", fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig2)

def analyze_correlations(returns_dict):
    returns_clean = {}
    for name, ret in returns_dict.items():
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        returns_clean[name] = ret
    returns_df = pd.DataFrame(returns_clean)
    corr_matrix = returns_df.corr()
    if corr_matrix.empty:
        st.warning("Zu wenig Daten für Korrelationsmatrix!")
        return corr_matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Korrelationsmatrix der täglichen Renditen", fontsize=14, pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    return corr_matrix

def analyze_rolling_performance(returns_dict, window=126):
    rolling_sharpe = pd.DataFrame()
    rolling_vol = pd.DataFrame()
    for name, ret in returns_dict.items():
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        if len(ret) < window:
            continue
        rolling_mean = ret.rolling(window).mean() * 252
        rolling_std = ret.rolling(window).std() * np.sqrt(252)
        rolling_sharpe[name] = (rolling_mean - RISK_FREE_RATE) / rolling_std
        rolling_vol[name] = rolling_std
    if rolling_sharpe.empty:
        st.warning("Zu wenig Daten für rollierende Kennzahlen!")
        return rolling_sharpe, rolling_vol
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in rolling_sharpe:
        ax.plot(rolling_sharpe.index, rolling_sharpe[name], label=name, linewidth=1)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage Fenster)", fontsize=14, pad=20)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    st.pyplot(fig)
    return rolling_sharpe, rolling_vol

def main():
    st.title("Portfolio Analyse Dashboard (ohne empyrical)")
    st.markdown("Vergleich verschiedener Strategien und Fonds (inkl. echte Namen aus Yahoo Finance)")

    # Zeitauswahl (Timeline)
    min_date = datetime(2024, 1, 1)
    max_date = datetime.today()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Startdatum", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end = st.date_input("Enddatum", min_value=min_date, max_value=max_date, value=max_date)
    if start > end:
        st.error("Das Startdatum darf nicht nach dem Enddatum liegen!")
        st.stop()

    # Lade Yahoo-Namen
    display_names_map = cached_display_names_map()

    # Lade alle Daten
    with st.spinner("Daten werden geladen..."):
        returns_dict, cumulative_dict = load_and_sync_data(start, end, display_names_map)
        metrics = calculate_metrics(returns_dict, cumulative_dict)

    # Metriken-Tabellen
    st.subheader("Risikokennzahlen & Performance")
    if metrics.empty:
        st.warning("Keine Daten für diesen Zeitraum!")
    else:
        st.dataframe(metrics.style.format({
            'Total Return': "{:.2%}", 'Annual Return': "{:.2%}",
            'Annual Volatility': "{:.2%}", 'Sharpe Ratio': "{:.2f}",
            'Sortino Ratio': "{:.2f}", 'Max Drawdown': "{:.2%}",
            'Calmar Ratio': "{:.2f}", 'VaR (95%)': "{:.2%}",
            'CVaR (95%)': "{:.2%}", 'Omega Ratio': "{:.2f}",
            'Tail Ratio': "{:.2f}", 'Win Rate': "{:.2%}",
            'Avg Win': "{:.2%}", 'Avg Loss': "{:.2%}",
            'Profit Factor': "{:.2f}", 'Positive Months': "{:.2%}"
        }), height=400)

    # Plots
    st.subheader("Kumulative Performance & Drawdown")
    plot_performance(cumulative_dict)

    st.subheader("Rolling Sharpe Ratio")
    analyze_rolling_performance(returns_dict, window=126)

    st.subheader("Korrelation der Tagesrenditen")
    analyze_correlations(returns_dict)

    # Monatsrenditen als Heatmap
    st.subheader("Monatliche Renditen")
    monthly_returns = pd.DataFrame({
        name: to_1d_series(ret).resample('M').apply(lambda x: (1 + x).prod() - 1)
        for name, ret in returns_dict.items()
    })
    if not monthly_returns.empty:
        fig, ax = plt.subplots(figsize=(12, max(3, len(monthly_returns.columns))))
        sns.heatmap(monthly_returns.T, annot=True, fmt='.1%', cmap='RdYlGn', center=0, linewidths=0.5, ax=ax)
        ax.set_title("Monatliche Renditen", fontsize=14, pad=20)
        ax.set_xticklabels(
            [pd.to_datetime(label.get_text()).strftime('%Y-%m') for label in ax.get_xticklabels()],
            rotation=45, ha='right'
        )
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Keine Monatsrenditen für diesen Zeitraum vorhanden.")

if __name__ == "__main__":
    main()
