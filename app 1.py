import streamlit as st
import numpy as np
import pandas as pd

from simulation import simulate_earthquakes, get_interarrival_times, monte_carlo_simulation, simulate_with_coordinates
from probability import (
    poisson_probability, probability_at_least_k, probability_at_least_one,
    expected_earthquakes, mean_interarrival_time, confidence_interval_monte_carlo,
    exceedance_probability,
)
from visualization import (
    plot_interarrival_histogram, plot_poisson_pmf, plot_monte_carlo, plot_geo_map,
)

st.set_page_config(
    page_title="Seismic Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: #f97316 !important;
    font-size: 1.7rem !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #f97316 !important; letter-spacing: -1px; }
h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e6edf3 !important; }
hr { border-color: #30363d !important; }
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #f97316;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #f97316;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 16px;
}
.prob-result {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #f97316;
    border-radius: 10px;
    padding: 20px 24px;
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
    margin: 10px 0;
}
.prob-result .value { font-size: 2.6rem; color: #f97316; font-weight: 600; }
.prob-result .label { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f97316 !important;
    border-bottom: 2px solid #f97316 !important;
}
.stButton button {
    background: #f97316 !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}
.stButton button:hover {
    background: #fb923c !important;
    box-shadow: 0 0 16px rgba(249,115,22,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">⚙ Parameters</p>', unsafe_allow_html=True)
    st.markdown("# 🌍 Seismic Simulator")
    st.markdown('<div class="info-box">Stochastic earthquake modeling via Poisson processes & Monte Carlo methods.</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**Earthquake Rate λ (events/year)**")
    rate = st.slider("", min_value=0.5, max_value=20.0, value=5.0, step=0.5, key="rate")

    st.markdown("**Simulation Time (years)**")
    time_period = st.slider("", min_value=1, max_value=100, value=10, step=1, key="time")

    st.markdown("**Magnitude Threshold (M ≥)**")
    mag_threshold = st.slider("", min_value=2.0, max_value=8.0, value=2.0, step=0.5, key="mag")

    st.markdown("**Monte Carlo Simulations**")
    n_sims = st.slider("", min_value=500, max_value=20000, value=5000, step=500, key="nsims")

    st.divider()
    run = st.button("▶  RUN SIMULATION", use_container_width=True)

    st.divider()
    st.markdown(f"""
    <div class="info-box">
    λ = {rate} eq/yr<br>
    t = {time_period} yr<br>
    E[N(t)] = <b style="color:#f97316">{expected_earthquakes(rate, time_period):.1f}</b><br>
    E[T] = <b style="color:#f97316">{mean_interarrival_time(rate):.3f} yr</b>
    </div>
    """, unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
if "events" not in st.session_state or run:
    st.session_state.events = simulate_earthquakes(rate, time_period, mag_threshold)
    st.session_state.geo_events = simulate_with_coordinates(rate, time_period, mag_threshold)
    st.session_state.mc_counts = monte_carlo_simulation(rate, time_period, n_sims, mag_threshold)
    st.session_state.sim_rate = rate
    st.session_state.sim_time_period = time_period
    st.session_state.sim_mag_threshold = mag_threshold

events = st.session_state.events
geo_events = st.session_state.geo_events
mc_counts = st.session_state.mc_counts
_rate = st.session_state.sim_rate
_time = st.session_state.sim_time_period
_eff_rate = len(events) / _time if _time > 0 and len(events) > 0 else _rate


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# EARTHQUAKE OCCURRENCE SIMULATOR")
st.markdown('<p class="section-label">Poisson Process · Stochastic Modeling · Monte Carlo</p>', unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Events", len(events))
with k2:
    st.metric("Expected E[N(t)]", f"{expected_earthquakes(_eff_rate, _time):.1f}")
with k3:
    iat = get_interarrival_times(events)
    st.metric("Avg Interarrival", f"{np.mean(iat):.3f} yr" if len(iat) > 0 else "N/A")
with k4:
    mags = [e[1] for e in events]
    st.metric("Max Magnitude", f"M{max(mags):.1f}" if mags else "N/A")
with k5:
    p1 = probability_at_least_one(_rate, _time)
    st.metric("P(≥1 event)", f"{p1:.4f}")

st.divider()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Interarrival Analysis",
    "🎲  Probability Calculator",
    "🔄  Monte Carlo",
    "🗺️  Geographic Map",
])


# ═══ TAB 1: Interarrival Analysis ═════════════════════════════════════════════
with tab1:
    st.markdown("### Interarrival Time Distribution")
    iat = get_interarrival_times(events)

    if len(iat) >= 2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Simulated Mean", f"{np.mean(iat):.4f} yr")
        with col2:
            st.metric("Theoretical Mean (1/λ)", f"{mean_interarrival_time(_eff_rate):.4f} yr")
        with col3:
            st.metric("Std Dev", f"{np.std(iat):.4f} yr")

        st.plotly_chart(plot_interarrival_histogram(iat, _eff_rate), use_container_width=True)

        st.markdown('<div class="info-box">The exponential distribution is <b>memoryless</b>: P(T > s+t | T > s) = P(T > t). The time until the next earthquake is independent of how long we have already waited.</div>', unsafe_allow_html=True)
    else:
        st.warning("Not enough events to analyze. Increase λ or simulation time.")


# ═══ TAB 2: Probability Calculator ════════════════════════════════════════════
with tab2:
    st.markdown("### Poisson Probability Calculator")

    col1, col2, col3 = st.columns(3)
    with col1:
        calc_lam = st.number_input("λ (rate)", min_value=0.1, max_value=50.0, value=float(_rate), step=0.5)
    with col2:
        calc_t = st.number_input("Time period (yr)", min_value=0.1, max_value=200.0, value=float(_time), step=1.0)
    with col3:
        calc_k = st.number_input("Number of earthquakes k", min_value=0, max_value=200, value=5, step=1)

    p_exact = poisson_probability(calc_lam, calc_t, calc_k)
    p_atleast = probability_at_least_k(calc_lam, calc_t, calc_k)
    p_atmost = 1.0 - p_atleast + p_exact

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""<div class="prob-result">
            <div class="value">{p_exact:.5f}</div>
            <div class="label">P(N(t) = {calc_k})</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="prob-result">
            <div class="value">{p_atleast:.5f}</div>
            <div class="label">P(N(t) ≥ {calc_k})</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div class="prob-result">
            <div class="value">{p_atmost:.5f}</div>
            <div class="label">P(N(t) ≤ {calc_k})</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Poisson PMF Distribution")
    st.plotly_chart(plot_poisson_pmf(calc_lam, calc_t, highlight_k=calc_k), use_container_width=True)

    st.markdown(f"""<div class="info-box">
    <b>Formula:</b>  P(N(t) = k) = (λt)ᵏ · e^(−λt) / k! <br>
    With λ={calc_lam}, t={calc_t}, k={calc_k}: μ = λt = {calc_lam*calc_t:.2f}
    </div>""", unsafe_allow_html=True)


# ═══ TAB 3: Monte Carlo ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### Monte Carlo Earthquake Simulation")
    st.markdown(f'<div class="info-box">Ran <b>{n_sims:,}</b> independent simulations of {_time}-year periods with λ={_rate}.</div>', unsafe_allow_html=True)

    mu, lo, hi = confidence_interval_monte_carlo(mc_counts)
    theoretical = expected_earthquakes(_eff_rate, _time)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Monte Carlo Mean", f"{mu:.2f}")
    with c2:
        st.metric("Theoretical E[N(t)]", f"{theoretical:.2f}")
    with c3:
        st.metric("95% CI Lower", f"{lo:.0f}")
    with c4:
        st.metric("95% CI Upper", f"{hi:.0f}")

    mc_eff_counts = monte_carlo_simulation(_eff_rate, _time, n_sims)
    st.plotly_chart(plot_monte_carlo(mc_eff_counts, theoretical), use_container_width=True)

    st.markdown("### Risk Estimation")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        risk_t = st.slider("Risk horizon (years)", 1, 50, 20, key="risk_t")
        risk_k = st.slider("Threshold k (at least k earthquakes)", 1, 100, 1, key="risk_k")
        risk_mag = st.slider("Magnitude filter (M ≥)", 2.0, 8.0, 6.0, step=0.5, key="risk_mag")

    with risk_col2:
        p_risk = probability_at_least_k(_rate, risk_t, risk_k)
        st.markdown(f"""<div class="prob-result" style="margin-top:32px">
            <div class="value">{p_risk:.4f}</div>
            <div class="label">P(N({risk_t} yr) ≥ {risk_k}) with λ={_rate}</div>
        </div>""", unsafe_allow_html=True)

        risk_counts = monte_carlo_simulation(_rate, risk_t, 5000, risk_mag)
        mc_prob = np.mean(risk_counts >= risk_k)
        st.markdown(f"""<div class="prob-result">
            <div class="value">{mc_prob:.4f}</div>
            <div class="label">MC Estimate (M≥{risk_mag}, N≥{risk_k}, {risk_t} yr)</div>
        </div>""", unsafe_allow_html=True)


# ═══ TAB 4: Geographic Map ════════════════════════════════════════════════════
with tab4:
    st.markdown("### Geographic Distribution")
    st.markdown('<div class="info-box">Earthquakes plotted at random coordinates for visualization. In a real system, historical catalogs or fault-line-based spatial models would be used.</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_geo_map(geo_events), use_container_width=True)

    if geo_events:
        df_geo = pd.DataFrame(geo_events)
        df_geo["magnitude"] = df_geo["magnitude"].round(2)
        df_geo["time"] = df_geo["time"].round(4)
        df_geo["depth"] = df_geo["depth"].round(1)
        df_geo["lat"] = df_geo["lat"].round(3)
        df_geo["lon"] = df_geo["lon"].round(3)
        st.dataframe(df_geo.rename(columns={
            "time": "Time (yr)", "magnitude": "Magnitude",
            "lat": "Latitude", "lon": "Longitude", "depth": "Depth (km)"
        }).reset_index(drop=True), use_container_width=True, height=300)
