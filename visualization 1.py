import numpy as np
import plotly.graph_objects as go
from scipy.stats import expon
import pandas as pd


PALETTE = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "accent": "#f97316",
    "accent2": "#fb923c",
    "accent_soft": "rgba(249,115,22,0.15)",
    "text": "#e6edf3",
    "subtext": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
}


def base_layout(**extra):
    layout = dict(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["text"], family="IBM Plex Mono, monospace"),
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )
    layout.update(extra)
    return layout


def plot_event_timeline(events, time_period):
    if not events:
        fig = go.Figure()
        fig.update_layout(**base_layout(title="No Events to Display"))
        return fig

    times = [e[0] for e in events]
    mags = [e[1] for e in events]
    sizes = [max(4, (m - 2) * 5) for m in mags]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=[0] * len(times),
        mode="markers",
        marker=dict(
            size=sizes,
            color=mags,
            colorscale=[[0, "#fb923c"], [0.5, "#f97316"], [1, "#ef4444"]],
            colorbar=dict(title="Magnitude", tickfont=dict(color=PALETTE["text"])),
            line=dict(color=PALETTE["accent"], width=0.5),
            opacity=0.85,
        ),
        text=[f"M{m:.1f} @ {t:.3f} yr" for t, m in events],
        hovertemplate="<b>%{text}</b><extra></extra>",
        name="Earthquake",
    ))

    fig.add_shape(type="line", x0=0, x1=time_period, y0=0, y1=0,
                  line=dict(color=PALETTE["accent"], width=2))

    fig.update_layout(**base_layout(
        title=dict(text=f"Earthquake Event Timeline - {len(events)} events over {time_period} years",
                   font=dict(size=14, color=PALETTE["accent"])),
        xaxis_title="Time (years)",
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        showlegend=False,
        height=240,
    ))
    return fig


def plot_interarrival_histogram(interarrival_times, rate):
    if len(interarrival_times) < 2:
        fig = go.Figure()
        fig.update_layout(**base_layout(title="Not enough data"))
        return fig

    x_max = np.percentile(interarrival_times, 99)
    x_range = np.linspace(0, x_max, 300)
    theoretical = expon.pdf(x_range, scale=1.0 / rate)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=interarrival_times,
        nbinsx=40,
        histnorm="probability density",
        marker=dict(color=PALETTE["accent_soft"], line=dict(color=PALETTE["accent"], width=1)),
        name="Simulated",
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=theoretical,
        mode="lines",
        line=dict(color=PALETTE["green"], width=2.5, dash="dot"),
        name=f"Theoretical Exp(rate={rate})",
    ))
    fig.update_layout(**base_layout(
        title=dict(text="Interarrival Time Distribution", font=dict(size=14, color=PALETTE["accent"])),
        xaxis_title="Time Between Earthquakes (years)",
        yaxis_title="Probability Density",
        legend=dict(bgcolor=PALETTE["card"], bordercolor=PALETTE["border"], borderwidth=1),
        height=380,
    ))
    return fig


def plot_poisson_pmf(lam, t, highlight_k=None):
    from scipy.stats import poisson
    mu = lam * t
    k_max = int(mu + 4 * np.sqrt(mu)) + 5
    k_values = np.arange(0, k_max + 1)
    probs = poisson.pmf(k_values, mu)

    colors = [PALETTE["accent"] if (highlight_k is not None and k == highlight_k)
              else PALETTE["accent_soft"] for k in k_values]
    border_colors = [PALETTE["accent2"] if (highlight_k is not None and k == highlight_k)
                     else PALETTE["accent"] for k in k_values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_values,
        y=probs,
        marker=dict(color=colors, line=dict(color=border_colors, width=1)),
        name="P(N(t)=k)",
        hovertemplate="k=%{x}<br>P=%{y:.5f}<extra></extra>",
    ))
    if highlight_k is not None:
        fig.add_vline(x=highlight_k, line=dict(color=PALETTE["accent2"], width=2, dash="dash"))

    fig.update_layout(**base_layout(
        title=dict(text=f"Poisson PMF  (lam={lam}, t={t} yr, mu={mu:.1f})",
                   font=dict(size=14, color=PALETTE["accent"])),
        xaxis_title="Number of Earthquakes k",
        yaxis_title="Probability P(N(t)=k)",
        height=380,
    ))
    return fig


def plot_monte_carlo(counts, theoretical_mean):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=counts,
        nbinsx=50,
        marker=dict(color=PALETTE["accent_soft"], line=dict(color=PALETTE["accent"], width=1)),
        name="Simulated Count",
        opacity=0.85,
    ))
    fig.add_vline(x=np.mean(counts), line=dict(color=PALETTE["green"], width=2.5),
                  annotation_text=f"Sim Mean: {np.mean(counts):.1f}",
                  annotation_font_color=PALETTE["green"])
    fig.add_vline(x=theoretical_mean, line=dict(color=PALETTE["blue"], width=2, dash="dash"),
                  annotation_text=f"Theory: {theoretical_mean:.1f}",
                  annotation_font_color=PALETTE["blue"])
    fig.update_layout(**base_layout(
        title=dict(text="Monte Carlo: Distribution of Earthquake Counts",
                   font=dict(size=14, color=PALETTE["accent"])),
        xaxis_title="Number of Earthquakes",
        yaxis_title="Frequency",
        legend=dict(bgcolor=PALETTE["card"], bordercolor=PALETTE["border"]),
        height=380,
    ))
    return fig


def plot_geo_map(geo_events):
    if not geo_events:
        fig = go.Figure()
        fig.update_layout(**base_layout(title="No events to map"))
        return fig

    df = pd.DataFrame(geo_events)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=df["lat"],
        lon=df["lon"],
        mode="markers",
        marker=dict(
            size=np.clip((df["magnitude"] - 2) * 3 + 4, 4, 22),
            color=df["magnitude"],
            colorscale=[[0, "#fb923c"], [0.5, "#f97316"], [1, "#ef4444"]],
            colorbar=dict(title="Magnitude", tickfont=dict(color=PALETTE["text"])),
            line=dict(color="white", width=0.3),
            opacity=0.75,
        ),
        text=[f"M{r['magnitude']:.1f} | {r['time']:.2f} yr | {r['depth']:.0f} km depth"
              for _, r in df.iterrows()],
        hovertemplate="<b>%{text}</b><br>Lat: %{lat:.2f}, Lon: %{lon:.2f}<extra></extra>",
    ))
    fig.update_layout(
        geo=dict(
            showland=True, landcolor="#1c2128",
            showocean=True, oceancolor="#0d1117",
            showcoastlines=True, coastlinecolor=PALETTE["border"],
            showframe=False,
            bgcolor=PALETTE["bg"],
        ),
        paper_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"], family="IBM Plex Mono, monospace"),
        title=dict(text=f"Geographic Distribution - {len(geo_events)} Events",
                   font=dict(size=14, color=PALETTE["accent"])),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    return fig


def plot_magnitude_distribution(events):
    if not events:
        fig = go.Figure()
        fig.update_layout(**base_layout(title="No events"))
        return fig

    mags = [e[1] for e in events]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mags,
        nbinsx=30,
        marker=dict(color=PALETTE["accent_soft"], line=dict(color=PALETTE["accent"], width=1)),
        name="Magnitude",
    ))
    fig.update_layout(**base_layout(
        title=dict(text="Magnitude Distribution (Gutenberg-Richter)",
                   font=dict(size=14, color=PALETTE["accent"])),
        xaxis_title="Magnitude (M)",
        yaxis_title="Count",
        height=320,
    ))
    return fig


def plot_cumulative_events(events, time_period):
    if not events:
        fig = go.Figure()
        fig.update_layout(**base_layout(title="No events"))
        return fig

    times = [0] + [e[0] for e in events] + [time_period]
    counts = list(range(len(events) + 1)) + [len(events)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=counts,
        mode="lines",
        line=dict(color=PALETTE["accent"], width=2, shape="hv"),
        fill="tozeroy",
        fillcolor=PALETTE["accent_soft"],
        name="Cumulative Events",
    ))
    fig.add_trace(go.Scatter(
        x=[0, time_period],
        y=[0, len(events)],
        mode="lines",
        line=dict(color=PALETTE["blue"], width=1.5, dash="dash"),
        name="Expected (linear)",
    ))
    fig.update_layout(**base_layout(
        title=dict(text="Cumulative Earthquake Count Over Time",
                   font=dict(size=14, color=PALETTE["accent"])),
        xaxis_title="Time (years)",
        yaxis_title="Cumulative Events",
        legend=dict(bgcolor=PALETTE["card"], bordercolor=PALETTE["border"]),
        height=320,
    ))
    return fig
