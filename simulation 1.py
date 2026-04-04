import numpy as np

# ─── Real seismic zone regions (lat_center, lon_center, lat_spread, lon_spread, weight)
SEISMIC_ZONES = [
    # Ring of Fire - West Pacific
    (35.0,  140.0, 8.0,  10.0, 8.0),   # Japan
    (0.0,   120.0, 8.0,  10.0, 6.0),   # Indonesia/Philippines
    (-5.0,  150.0, 6.0,  10.0, 5.0),   # Papua New Guinea
    (-20.0, 170.0, 6.0,  8.0,  4.0),   # Vanuatu / Tonga
    (-35.0, -70.0, 8.0,  6.0,  6.0),   # Chile
    (0.0,   -78.0, 6.0,  5.0,  5.0),   # Ecuador / Colombia
    (15.0,  -92.0, 5.0,  5.0,  4.0),   # Mexico / Central America
    (55.0,  160.0, 6.0,  8.0,  4.0),   # Kamchatka / Russia
    (60.0, -150.0, 5.0,  8.0,  4.0),   # Alaska
    (45.0, -122.0, 4.0,  5.0,  3.0),   # Pacific Northwest USA
    # Ring of Fire - East
    (-15.0, -75.0, 6.0,  5.0,  4.0),   # Peru
    # Himalayas / Alpine Belt
    (30.0,   80.0, 5.0,  10.0, 5.0),   # Himalayas / Tibet
    (38.0,   40.0, 5.0,  10.0, 4.0),   # Turkey / Iran
    (36.0,   15.0, 4.0,  8.0,  3.0),   # Mediterranean
    # Mid-Atlantic Ridge
    (65.0,  -20.0, 4.0,  5.0,  2.0),   # Iceland
    (0.0,   -25.0, 5.0,  3.0,  2.0),   # Mid-Atlantic
    # East Africa Rift
    (5.0,    37.0, 8.0,  4.0,  2.0),   # East Africa
    # Caribbean
    (17.0,  -72.0, 3.0,  4.0,  2.0),   # Caribbean
]

ZONE_WEIGHTS = np.array([z[4] for z in SEISMIC_ZONES])
ZONE_WEIGHTS = ZONE_WEIGHTS / ZONE_WEIGHTS.sum()


def simulate_earthquakes(rate, time_period, magnitude_threshold=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    events = []
    current_time = 0.0

    while current_time < time_period:
        interarrival = np.random.exponential(1.0 / rate)
        current_time += interarrival
        if current_time < time_period:
            magnitude = simulate_magnitude()
            if magnitude >= magnitude_threshold:
                events.append((current_time, magnitude))

    return events


def simulate_magnitude(a=8.0, b=1.0, m_min=2.0, m_max=9.5):
    beta = b * np.log(10)
    u = np.random.uniform(0, 1)
    magnitude = m_min - (1.0 / beta) * np.log(1 - u * (1 - np.exp(-beta * (m_max - m_min))))
    return np.clip(magnitude, m_min, m_max)


def get_interarrival_times(events):
    if len(events) < 2:
        return np.array([])
    times = np.array([e[0] for e in events])
    return np.diff(times)


def monte_carlo_simulation(rate, time_period, n_simulations=10000, magnitude_threshold=0.0):
    counts = []
    for _ in range(n_simulations):
        events = simulate_earthquakes(rate, time_period, magnitude_threshold)
        counts.append(len(events))
    return np.array(counts)


def _random_seismic_location():
    """Pick a random location biased toward real seismic zones."""
    zone_idx = np.random.choice(len(SEISMIC_ZONES), p=ZONE_WEIGHTS)
    lat_c, lon_c, lat_s, lon_s, _ = SEISMIC_ZONES[zone_idx]
    lat = np.random.normal(lat_c, lat_s)
    lon = np.random.normal(lon_c, lon_s)
    lat = np.clip(lat, -85, 85)
    lon = np.clip(lon, -180, 180)
    return lat, lon


def simulate_with_coordinates(rate, time_period, magnitude_threshold=0.0, **kwargs):
    events = simulate_earthquakes(rate, time_period, magnitude_threshold)
    geo_events = []
    for t, mag in events:
        lat, lon = _random_seismic_location()
        geo_events.append({
            "time": t,
            "magnitude": mag,
            "lat": lat,
            "lon": lon,
            "depth": np.random.exponential(30),
        })
    return geo_events
