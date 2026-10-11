"""Geographic retrieval and Folium maps for illustrative civic complaints.

Distances are kilometres. Centroids are complaint-coordinate averages, not
official boundaries. Hotspots are area counts, not a geographic density model.
No data is loaded and no network requests are made by this module.
"""
from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import folium


def _number(value: object, name: str) -> float:
    """Accept finite numbers and numeric strings, rejecting missing values."""
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite numeric value") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value")
    return number


def _coordinate(value: object, name: str, limit: float) -> float:
    number = _number(value, name)
    if not -limit <= number <= limit:
        raise ValueError(f"{name} must be between {-limit:g} and {limit:g}")
    return number


def _radius(value: object) -> float:
    radius = _number(value, "radius_km")
    if radius < 0:
        raise ValueError("radius_km must be non-negative")
    return radius


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _usable_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric coordinates, dropping unusable rows."""
    _require_columns(df, ("latitude", "longitude"))
    clean = df.copy()
    for name, limit in (("latitude", 90), ("longitude", 180)):
        def convert(value: object) -> float:
            try:
                return _coordinate(value, name, limit)
            except ValueError:
                return float("nan")
        clean[name] = clean[name].map(convert).astype(float)
    return clean.dropna(subset=["latitude", "longitude"])


def _area_rows(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ("area",))
    valid = df["area"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    return df.loc[valid].copy()


def haversine_distance(
    lat1: float | str, lon1: float | str, lat2: float | str, lon2: float | str,
) -> float:
    """Return great-circle kilometres on a sphere of mean radius 6,371 km.

    Invalid, non-finite, or out-of-range coordinates raise ValueError.
    """
    lat1 = math.radians(_coordinate(lat1, "lat1", 90))
    lon1 = math.radians(_coordinate(lon1, "lon1", 180))
    lat2 = math.radians(_coordinate(lat2, "lat2", 90))
    lon2 = math.radians(_coordinate(lon2, "lon2", 180))
    value = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    value = min(1.0, max(0.0, value))
    return 6371.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def filter_by_radius(
    df: pd.DataFrame, center_lat: float | str, center_lon: float | str,
    radius_km: float | str,
) -> pd.DataFrame:
    """Copy rows within an inclusive radius, adding sorted numeric distance_km.

    Original columns, coordinate values, and index labels are preserved.
    Invalid row coordinates are excluded; invalid centre/radius raises ValueError.
    """
    center_lat = _coordinate(center_lat, "center_lat", 90)
    center_lon = _coordinate(center_lon, "center_lon", 180)
    radius = _radius(radius_km)
    # A positional index prevents duplicate caller index labels from joining rows.
    clean = _usable_coordinates(df.reset_index(drop=True))
    distances = [haversine_distance(center_lat, center_lon, lat, lon)
                 for lat, lon in zip(clean["latitude"], clean["longitude"])]
    result = df.iloc[clean.index].copy()
    result["distance_km"] = pd.Series(distances, dtype=float).to_numpy()
    return result.loc[result["distance_km"] <= radius].sort_values(
        "distance_km", kind="stable",
    )


def get_area_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Return area-sorted mean coordinates from rows with valid areas/coordinates.

    These are approximate complaint averages, not administrative centroids.
    Requires area, latitude and longitude columns, even for empty input.
    """
    clean = _usable_coordinates(_area_rows(df))
    return clean.groupby("area", sort=True)[["latitude", "longitude"]].mean().reset_index()


def get_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """Count rows per valid area and exact Unresolved statuses, attaching centroids.

    This is area-level complaint volume, not statistically validated density.
    Each input row counts once, including rows without usable coordinates.
    Areas without coordinates have NaN centroids and cannot be mapped.
    Requires area, status, latitude and longitude columns.
    """
    _require_columns(df, ("area", "status", "latitude", "longitude"))
    areas = _area_rows(df)
    counts = areas.groupby("area", sort=True).size().rename("complaint_count")
    unresolved = areas["status"].eq("Unresolved").fillna(False)
    unresolved_counts = unresolved.groupby(areas["area"], sort=True).sum()
    result = pd.concat([counts, unresolved_counts.rename("unresolved_count")], axis=1)
    result = result.reset_index().merge(get_area_centroids(df), on="area", how="left")
    result["complaint_count"] = result["complaint_count"].astype(int)
    result["unresolved_count"] = result["unresolved_count"].astype(int)
    return result[["area", "complaint_count", "unresolved_count", "latitude", "longitude"]].sort_values(
        ["complaint_count", "area"], ascending=[False, True], ignore_index=True,
    )


def _popup(row: pd.Series) -> str:
    lines = ["<b>Illustrative sample complaint</b>"]
    for column, label in (("complaint_id", "Complaint ID"), ("issue_type", "Issue type"),
                          ("status", "Status"), ("area", "Area"), ("road_name", "Road")):
        value = row.get(column)
        if value is not None and pd.notna(value):
            lines.append(f"{label}: {html.escape(str(value), quote=True)}")
    if "distance_km" in row:
        try:
            distance = _number(row["distance_km"], "distance_km")
            if distance >= 0:
                lines.append(f"Distance: {distance:.3f} km")
        except ValueError:
            pass
    return "<br>".join(lines)


def create_complaint_map(
    df: pd.DataFrame, center_lat: float | str | None = None,
    center_lon: float | str | None = None, radius_km: float | str | None = None,
) -> folium.Map:
    """Return clustered sample markers, optional radius and togglable hotspot bubbles.

    Coordinates are required; display fields are optional. Hotspots are added
    (hidden initially) when area and status columns exist. An explicit centre
    takes precedence over the mean usable coordinates; empty maps use (0, 0).
    Map construction makes no web requests. Browser rendering uses Folium's
    usual external JavaScript/CSS and OpenStreetMap tiles.
    """
    if (center_lat is None) != (center_lon is None):
        raise ValueError("center_lat and center_lon must be supplied together")
    if radius_km is not None and center_lat is None:
        raise ValueError("radius_km requires center_lat and center_lon")
    radius = None if radius_km is None else _radius(radius_km)
    clean = _usable_coordinates(df)
    if center_lat is not None:
        center = [_coordinate(center_lat, "center_lat", 90),
                  _coordinate(center_lon, "center_lon", 180)]
    elif not clean.empty:
        center = [float(clean["latitude"].mean()), float(clean["longitude"].mean())]
    else:
        center = [0.0, 0.0]
    # Geographic calculations work with Pandas alone; only maps need Folium.
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ModuleNotFoundError as error:
        raise ImportError("Complaint maps require Folium and its dependencies; install folium") from error
    map_object = folium.Map(location=center, zoom_start=12 if center_lat is not None or not clean.empty else 2)
    map_object.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;top:10px;left:50px;z-index:1000;'
        'background:white;padding:6px">CivicLens KG — illustrative sample data</div>'
    ))
    markers = MarkerCluster(name="Sample complaints").add_to(map_object)
    colors = {"Unresolved": "red", "In Progress": "orange", "Resolved": "green"}
    for _, row in clean.iterrows():
        status = row.get("status")
        color = colors.get(status, "blue") if isinstance(status, str) else "blue"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(_popup(row), max_width=350),
            icon=folium.Icon(color=color),
        ).add_to(markers)
    if radius is not None:
        layer = folium.FeatureGroup(name="Search radius").add_to(map_object)
        folium.Circle(location=center, radius=radius * 1000, color="blue", fill=False,
                      tooltip=f"Search radius: {radius:g} km").add_to(layer)
    if {"area", "status"} <= set(df.columns):
        layer = folium.FeatureGroup(name="Area complaint counts", show=False).add_to(map_object)
        for row in get_hotspots(df).dropna(subset=["latitude", "longitude"]).itertuples(index=False):
            popup = (f"<b>Sample area complaint counts</b><br>Area: {html.escape(str(row.area), quote=True)}"
                     f"<br>Total: {row.complaint_count}<br>Unresolved: {row.unresolved_count}"
                     "<br>Approximate complaint centroid")
            folium.CircleMarker(
                location=[row.latitude, row.longitude], radius=5 + 3 * math.sqrt(row.complaint_count),
                color="purple", fill=True, fill_opacity=0.35,
                popup=folium.Popup(popup, max_width=350),
            ).add_to(layer)
    folium.LayerControl().add_to(map_object)
    return map_object
