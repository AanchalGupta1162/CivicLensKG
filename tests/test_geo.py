"""Deterministic geospatial tests; no network or Streamlit session needed."""
from __future__ import annotations

import math
import socket
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

try:
    import folium
    from folium.plugins import MarkerCluster
except ModuleNotFoundError:
    folium = None

requires_folium = pytest.mark.skipif(folium is None, reason="Folium is not installed locally")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geo  # noqa: E402


@pytest.fixture
def complaints():
    return pd.DataFrame({
        "complaint_id": ["sample-far", "sample-center", "sample-near", "sample-invalid"],
        "latitude": [0, "0", "0", None],
        "longitude": [0.03, "0", "0.01", 0],
        "area": ["Beta", "Alpha", "Alpha", "Beta"],
        "status": ["Resolved", "Unresolved", "In Progress", "Unresolved"],
        "description": ["far", "center", "near", "missing"],
    }, index=[9, 3, 3, 7])


def test_identical_and_known_distances():
    assert geo.haversine_distance(19, 72, 19, 72) == pytest.approx(0, abs=1e-12)
    # Equator-to-pole is a quarter circumference, independently R*pi/2.
    assert geo.haversine_distance(0, 0, 90, 0) == pytest.approx(10007.543398, abs=1e-6)
    # One equatorial degree: circumference divided by 360.
    assert geo.haversine_distance("0", "0", "0", "1") == pytest.approx(111.1949266, abs=1e-6)


@pytest.mark.parametrize("position,value", [
    (0, 91), (0, -91), (2, 91), (1, 181), (1, -181), (3, 181),
    (0, float("nan")), (1, float("inf")), (2, -float("inf")),
    (3, "not numeric"), (0, None), (1, True),
])
def test_invalid_coordinates(position, value):
    coordinates = [0, 0, 0, 0]
    coordinates[position] = value
    with pytest.raises(ValueError, match="lat|lon"):
        geo.haversine_distance(*coordinates)


@pytest.mark.parametrize("points", [(0, 0, 0, 180), (90, 180, -90, -180),
                                    (-40, 170, 20, -170), (0, 180, 0, -180)])
def test_distance_finite_nonnegative_and_symmetric(points):
    distance = geo.haversine_distance(*points)
    assert math.isfinite(distance) and 0 <= distance <= math.pi * 6371
    assert distance == pytest.approx(geo.haversine_distance(*points[2:], *points[:2]))


def test_radius_filter_inclusive_sorted_preserves_input(complaints):
    original = complaints.copy(deep=True)
    result = geo.filter_by_radius(complaints, "0", "0", 2)
    assert result.complaint_id.tolist() == ["sample-center", "sample-near"]
    assert result.index.tolist() == [3, 3]
    assert result.longitude.tolist() == ["0", "0.01"]
    assert result.columns.tolist() == complaints.columns.tolist() + ["distance_km"]
    assert result.distance_km.tolist() == pytest.approx([0, math.pi * 6371 / 18000])
    assert result.distance_km.is_monotonic_increasing
    assert_frame_equal(complaints, original)


def test_radius_zero_and_exact_boundary(complaints):
    assert geo.filter_by_radius(complaints, 0, 0, 0).complaint_id.tolist() == ["sample-center"]
    boundary = geo.haversine_distance(0, 0, 0, 0.01)
    assert "sample-near" in geo.filter_by_radius(complaints, 0, 0, boundary).complaint_id.tolist()
    assert "sample-near" not in geo.filter_by_radius(complaints, 0, 0, boundary - 1e-9).complaint_id.tolist()


def test_invalid_rows_excluded():
    df = pd.DataFrame({"latitude": [None, "bad", 91, float("inf"), 0, 0, True],
                       "longitude": [0, 0, 0, 0, float("nan"), 181, 0]})
    assert geo.filter_by_radius(df, 0, 0, 20000).empty


@pytest.mark.parametrize("radius", [-1, float("nan"), float("inf"), "bad", None, True])
def test_invalid_radius(complaints, radius):
    with pytest.raises(ValueError, match="radius_km"):
        geo.filter_by_radius(complaints, 0, 0, radius)


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181), (None, 0), (0, "bad")])
def test_invalid_filter_center(complaints, lat, lon):
    with pytest.raises(ValueError, match="center_"):
        geo.filter_by_radius(complaints, lat, lon, 2)


def test_empty_schemas():
    df = pd.DataFrame(columns=["complaint_id", "latitude", "longitude", "area", "status"])
    result = geo.filter_by_radius(df, 0, 0, 1)
    assert result.empty and result.columns.tolist() == df.columns.tolist() + ["distance_km"]
    assert pd.api.types.is_float_dtype(result.distance_km)
    assert geo.get_area_centroids(df).columns.tolist() == ["area", "latitude", "longitude"]
    hotspots = geo.get_hotspots(df)
    assert hotspots.empty
    assert hotspots.columns.tolist() == ["area", "complaint_count", "unresolved_count", "latitude", "longitude"]


@requires_folium
def test_empty_map():
    df = pd.DataFrame(columns=["latitude", "longitude"])
    assert isinstance(geo.create_complaint_map(df), folium.Map)


def test_centroids_hotspot_counts_order_and_no_mutation():
    df = pd.DataFrame({"area": ["Beta", "Alpha", "Alpha", "Beta", "Gamma", " ", None],
                       "latitude": [10, "0", "2", 91, None, 0, 0],
                       "longitude": [20, "4", "6", 0, 0, 0, 0],
                       "status": ["Resolved", "Unresolved", "unresolved", "Unresolved",
                                  "In Progress", "Unresolved", "Unresolved"]})
    original = df.copy(deep=True)
    centroids = geo.get_area_centroids(df)
    assert centroids.area.tolist() == ["Alpha", "Beta"]
    assert centroids.latitude.tolist() == pytest.approx([1, 10])
    assert centroids.longitude.tolist() == pytest.approx([5, 20])
    hotspots = geo.get_hotspots(df)
    assert hotspots.area.tolist() == ["Alpha", "Beta", "Gamma"]
    assert hotspots.complaint_count.tolist() == [2, 2, 1]
    assert hotspots.unresolved_count.tolist() == [1, 1, 0]
    assert pd.isna(hotspots.iloc[2].latitude)
    assert_frame_equal(df, original)


@pytest.mark.parametrize("function", [geo.get_area_centroids, geo.get_hotspots])
def test_invalid_only_centroids_and_hotspots(function):
    df = pd.DataFrame({"area": ["Alpha"], "status": [None], "latitude": [float("inf")], "longitude": [0]})
    result = function(df)
    if function is geo.get_area_centroids:
        assert result.empty
    else:
        assert result.complaint_count.tolist() == [1]
        assert result.unresolved_count.tolist() == [0]
        assert result.latitude.isna().all()


@pytest.mark.parametrize("function,kwargs", [
    (geo.filter_by_radius, {"center_lat": 0, "center_lon": 0, "radius_km": 1}),
    (geo.get_area_centroids, {}), (geo.get_hotspots, {}), (geo.create_complaint_map, {}),
])
def test_missing_coordinate_columns(function, kwargs):
    with pytest.raises(ValueError, match="Missing required columns.*longitude"):
        function(pd.DataFrame(columns=["area", "status", "latitude"]), **kwargs)


def test_missing_aggregation_columns():
    with pytest.raises(ValueError, match="area"):
        geo.get_area_centroids(pd.DataFrame(columns=["latitude", "longitude"]))
    with pytest.raises(ValueError, match="status"):
        geo.get_hotspots(pd.DataFrame(columns=["area", "latitude", "longitude"]))


@requires_folium
def test_shared_dataset_map_without_network(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Map construction attempted a network request")
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    df = pd.read_csv(ROOT / "complaints.csv", dtype=str, keep_default_na=False)
    original = df.copy(deep=True)
    map_object = geo.create_complaint_map(df)
    assert isinstance(map_object, folium.Map)
    cluster = next(child for child in map_object._children.values() if isinstance(child, MarkerCluster))
    assert len(cluster._children) == len(df)
    assert "illustrative sample data" in map_object.get_root().render()
    assert_frame_equal(df, original)


@requires_folium
def test_map_center_radius_and_layers(complaints):
    filtered = geo.filter_by_radius(complaints, 0, 0, 2)
    map_object = geo.create_complaint_map(filtered, "1", "2", "2.5")
    assert map_object.location == [1, 2]
    layers = [child for child in map_object._children.values() if isinstance(child, folium.FeatureGroup)]
    circle = next(child for layer in layers for child in layer._children.values()
                  if isinstance(child, folium.Circle))
    assert circle.location == [1, 2]
    assert circle.options["radius"] == 2500
    hotspots = next(layer for layer in layers if layer.layer_name == "Area complaint counts")
    assert len(hotspots._children) == len(geo.get_area_centroids(filtered))
    assert hotspots.show is False
    assert any(isinstance(child, folium.LayerControl) for child in map_object._children.values())
    assert "Distance: 0.000 km" in map_object.get_root().render()


@requires_folium
def test_map_automatic_center_and_status_colors(complaints):
    map_object = geo.create_complaint_map(complaints)
    assert map_object.location == pytest.approx([0, 0.04 / 3])
    cluster = next(child for child in map_object._children.values() if isinstance(child, MarkerCluster))
    colors = []
    for marker in cluster._children.values():
        icon = next(child for child in marker._children.values() if isinstance(child, folium.Icon))
        colors.append(icon.options["marker_color"])
    assert colors == ["green", "red", "orange"]


@requires_folium
def test_map_optional_fields_escaping_and_invalid_only():
    df = pd.DataFrame({"latitude": [0], "longitude": [0],
                       "complaint_id": ['<script>alert("x")</script>']})
    rendered = geo.create_complaint_map(df).get_root().render()
    assert '<script>alert("x")</script>' not in rendered
    assert "&lt;script&gt;" in rendered
    assert isinstance(geo.create_complaint_map(df[["latitude", "longitude"]]), folium.Map)
    df.latitude = float("nan")
    map_object = geo.create_complaint_map(df)
    assert map_object.location == [0, 0]
    cluster = next(child for child in map_object._children.values() if isinstance(child, MarkerCluster))
    assert not cluster._children
    assert geo.create_complaint_map(df, 10, 20).location == [10, 20]


@pytest.mark.parametrize("kwargs", [
    {"center_lat": 0}, {"center_lon": 0}, {"radius_km": 1},
    {"center_lat": 91, "center_lon": 0}, {"center_lat": 0, "center_lon": 181},
    {"center_lat": 0, "center_lon": 0, "radius_km": -1},
    {"center_lat": 0, "center_lon": 0, "radius_km": float("inf")},
])
def test_invalid_map_arguments(complaints, kwargs):
    with pytest.raises(ValueError, match="center_|radius_km"):
        geo.create_complaint_map(complaints, **kwargs)
