import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.loader import (  # noqa: E402
    HEAVY_POLLUTION_PM25_THRESHOLD,
    cluster_stations_into_fog_groups,
    fit_global_scaler_and_pca,
    load_station_daily,
    temporal_train_test_split,
    transform_station,
)

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data_raw", "PRSA", "PRSA_Data_20130301-20170228"
)
skip_if_no_data = pytest.mark.skipif(
    not os.path.isdir(DATA_DIR), reason="Beijing air-quality CSVs not present; see README to download"
)


@skip_if_no_data
def test_load_station_daily_shapes():
    stations = load_station_daily(DATA_DIR)
    assert len(stations) == 12
    for name, df in stations.items():
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0.0, 1.0})
        assert df.index.is_monotonic_increasing


@skip_if_no_data
def test_temporal_split_no_leakage():
    stations = load_station_daily(DATA_DIR)
    train, test = temporal_train_test_split(stations, test_days=180)
    for name in stations:
        assert train[name].index.max() < test[name].index.min()


@skip_if_no_data
def test_pca_transform_shape():
    stations = load_station_daily(DATA_DIR)
    train, _ = temporal_train_test_split(stations, test_days=180)
    scaler, pca, explained = fit_global_scaler_and_pca(train, n_components=4)
    assert 0.0 < explained <= 1.0
    x, y = transform_station(train["Aotizhongxin"], scaler, pca)
    assert x.shape[1] == 4
    assert x.shape[0] == len(y)


@skip_if_no_data
def test_fog_groups_balanced():
    stations = load_station_daily(DATA_DIR)
    train, _ = temporal_train_test_split(stations, test_days=180)
    groups = cluster_stations_into_fog_groups(train, n_groups=4)
    sizes = [len(v) for v in groups.values()]
    assert sum(sizes) == 12
    assert max(sizes) - min(sizes) <= 1


def test_threshold_matches_regulatory_value():
    assert HEAVY_POLLUTION_PM25_THRESHOLD == 150.0
