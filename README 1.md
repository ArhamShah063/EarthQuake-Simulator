# 🌍 Earthquake Occurrence Simulator

> Stochastic earthquake modeling using Poisson processes, exponential distributions, and Monte Carlo methods — built with Python and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📌 Overview

This interactive simulator models earthquake occurrences as a **Poisson process**, where:
- Events are independent and occur at a constant average rate **λ** (earthquakes/year)
- Interarrival times follow an **Exponential distribution**
- Magnitudes follow the **Gutenberg–Richter law**

---

## 🧮 Mathematical Background

### Poisson Process
$$P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}$$

### Exponential Interarrival Times
$$f(t) = \lambda e^{-\lambda t}, \quad E[T] = \frac{1}{\lambda}$$

### Gutenberg–Richter Law (Magnitude)
$$\log_{10} N = a - bM$$

---

## 🚀 Features

| Module | Description |
|--------|-------------|
| **Event Timeline** | Interactive scatter plot of earthquake events sized by magnitude |
| **Interarrival Analysis** | Histogram vs. theoretical exponential curve |
| **Probability Calculator** | Compute P(N(t)=k), P(N(t)≥k), P(N(t)≤k) |
| **Monte Carlo Simulation** | Run 500–20,000 simulations; confidence intervals |
| **Geographic Map** | World map visualization with magnitude & depth |

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/earthquake-simulator.git
cd earthquake-simulator
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
earthquake_simulator/
├── app.py              # Main Streamlit application
├── simulation.py       # Core Poisson process simulation
├── probability.py      # Probability calculations
├── visualization.py    # Plotly chart builders
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🎛 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| λ (rate) | Average earthquakes per year | 5.0 |
| Time Period | Simulation horizon (years) | 10 |
| Magnitude Threshold | Minimum magnitude to record | 2.0 |
| N Simulations | Monte Carlo sample size | 5000 |

---

## 📊 Example Results

With λ = 5 earthquakes/year over 10 years:
- **Expected events:** E[N(t)] = λt = **50**
- **Mean interarrival time:** 1/λ = **0.2 years**
- **P(≥1 event in 1 year):** 1 − e⁻⁵ ≈ **0.9933**

---

## 🚢 Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **main file**: `app.py`
5. Click **Deploy**

---

## 📚 Learning Outcomes

- Poisson processes and stochastic simulation
- Exponential distribution and memoryless property
- Monte Carlo methods for probability estimation
- Gutenberg–Richter magnitude modeling
- Interactive data visualization with Plotly + Streamlit

---

## 📄 License

MIT License — free to use, modify, and distribute.
