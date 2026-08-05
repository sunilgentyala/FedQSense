"""
Beijing Multi-Site Air-Quality data loader for the FedQSense hierarchical
federated learning testbed.

Source: Zhang, S., Guo, B., Dong, A., et al. "Cautionary Tales on Air-Quality
Improvement in Beijing." Proceedings of the Royal Society A, 473(2205), 2017.
Distributed via the UCI Machine Learning Repository (dataset 501).

Each of the 12 monitoring stations is treated as one federated edge client.
Hourly readings are aggregated to daily means (RAIN uses daily sum) to give a
next-day "heavy pollution event" early-warning task, consistent with the
Chinese Ministry of Ecology and Environment AQI breakpoint table (HJ 633-2012),
where a 24h PM2.5 mean above 150 micrograms/m^3 falls in the "heavily
polluted" (Level 5) band.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HEAVY_POLLUTION_PM25_THRESHOLD = 150.0  # micrograms/m^3, HJ 633-2012 Level 5 cut

NUMERIC_COLS = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "WSPM",
]
SUM_COLS = ["RAIN"]
FEATURE_COLS = NUMERIC_COLS + SUM_COLS  # 11 raw daily features
MIN_VALID_HOURS_PER_DAY = 18


def _station_name(path):
    base = os.path.basename(path)
    # PRSA_Data_<Station>_20130301-20170228.csv
    return base.split("PRSA_Data_")[1].split("_2013")[0]


def load_station_daily(csv_dir):
    """Load every station CSV, aggregate hourly readings to daily rows.

    Returns a dict: station_name -> DataFrame indexed by date with
    FEATURE_COLS plus a 'label' column (next-day heavy pollution event).
    """
    paths = sorted(glob.glob(os.path.join(csv_dir, "PRSA_Data_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No station CSVs found under {csv_dir}")

    stations = {}
    for path in paths:
        name = _station_name(path)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])

        valid_hours = df.groupby("date")["PM2.5"].apply(lambda s: s.notna().sum())

        agg = {c: "mean" for c in NUMERIC_COLS}
        agg.update({c: "sum" for c in SUM_COLS})
        daily = df.groupby("date").agg(agg)
        daily = daily.loc[valid_hours[valid_hours >= MIN_VALID_HOURS_PER_DAY].index]
        daily = daily.sort_index()
        daily = daily.dropna(subset=FEATURE_COLS)

        daily["label"] = (daily["PM2.5"].shift(-1) > HEAVY_POLLUTION_PM25_THRESHOLD).astype(float)
        daily = daily.iloc[:-1]  # drop last row: no next-day label available

        stations[name] = daily

    return stations


def temporal_train_test_split(stations, test_days=180):
    """Reserve the final `test_days` calendar days of each station as test."""
    train, test = {}, {}
    for name, df in stations.items():
        cutoff = df.index.max() - pd.Timedelta(days=test_days)
        train[name] = df[df.index <= cutoff]
        test[name] = df[df.index > cutoff]
    return train, test


def fit_global_scaler_and_pca(train_stations, n_components):
    """Fit a StandardScaler + PCA jointly on the pooled training split only.

    Pooling avoids leaking any per-station test-period statistics while
    keeping feature scaling comparable across clients, matching standard
    federated-learning preprocessing practice.
    """
    pooled = pd.concat([df[FEATURE_COLS] for df in train_stations.values()], axis=0)
    scaler = StandardScaler().fit(pooled.values)
    if n_components is None:
        return scaler, None, 1.0
    pooled_scaled = scaler.transform(pooled.values)
    pca = PCA(n_components=n_components, random_state=0).fit(pooled_scaled)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return scaler, pca, explained


def transform_station(df, scaler, pca=None):
    x_full = scaler.transform(df[FEATURE_COLS].values)
    y = df["label"].values.astype(np.float32)
    if pca is not None:
        x = pca.transform(x_full).astype(np.float32)
    else:
        x = x_full.astype(np.float32)
    return x, y


def cluster_stations_into_fog_groups(train_stations, n_groups=4, seed=0):
    """Data-driven, size-balanced fog-tier grouping.

    Each station's mean raw feature profile is projected onto its first
    principal component (a one-dimensional pollutant/meteorology severity
    axis fit on the station profiles themselves); stations are ranked along
    this axis and cut into `n_groups` equal-size consecutive buckets. This
    keeps every fog cluster the same size (required for a clean edge/fog/
    cloud demonstration) while remaining fully data-driven and reproducible,
    with no external geographic metadata required.
    """
    names = sorted(train_stations.keys())
    profiles = np.stack(
        [train_stations[n][FEATURE_COLS].mean(axis=0).values for n in names]
    )
    profiles = StandardScaler().fit_transform(profiles)
    axis = PCA(n_components=1, random_state=seed).fit_transform(profiles).ravel()
    order = np.argsort(axis)
    ranked_names = [names[i] for i in order]

    groups = {g: [] for g in range(n_groups)}
    for rank, name in enumerate(ranked_names):
        groups[rank % n_groups].append(name)
    return groups
