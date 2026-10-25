
import sys

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from distributed_ir import DistributedIR
from geo import haversine_distance


CSV_PATH = "complaints.csv"

# Initialize the distributed search system.
ir = DistributedIR(csv_path=CSV_PATH)

# Load the dataset to choose a real coordinate for the geographic test.
df = pd.read_csv(CSV_PATH)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

valid_coords = df.dropna(subset=["latitude", "longitude"])

if valid_coords.empty:
    raise ValueError("No complaints have valid latitude and longitude values.")


def get_value(result, field):
    """Read a field from either a SearchResult object or a dictionary."""
    if isinstance(result, dict):
        return result.get(field)
    return getattr(result, field, None)


def get_id(result):
    return str(get_value(result, "complaint_id"))


def show_results(results):
    for result in results:
        print(
            f"ID={get_id(result)}, "
            f"Department={get_value(result, 'department')}, "
            f"Status={get_value(result, 'status')}, "
            f"Score={get_value(result, 'ranking_score')}"
        )


# --------------------------------------------------
# TEST 1: Department isolation
# --------------------------------------------------
print("\n" + "=" * 55)
print("TEST 1: DEPARTMENT ISOLATION — ROADS ONLY")
print("=" * 55)

roads_results = ir.search_all_indexes(
    query="pothole near bus stop",
    departments=["Roads"],
    top_k=10
)

show_results(roads_results)

wrong_departments = [
    get_id(r)
    for r in roads_results
    if str(get_value(r, "department")).strip().casefold() != "roads"
]

if wrong_departments:
    print("FAIL: Non-Roads complaints returned:", wrong_departments)
else:
    print("PASS: Every returned complaint belongs to Roads.")

# An empty result is also possible, so check whether the test had
# any results to verify.
if not roads_results:
    print("NOTE: No results returned; isolation could not be fully verified.")


# --------------------------------------------------
# TEST 2: Status filtering
# --------------------------------------------------
print("\n" + "=" * 55)
print("TEST 2: STATUS FILTERING — UNRESOLVED ONLY")
print("=" * 55)

unresolved_results = ir.search_all_indexes(
    query="pothole",
    top_k=20,
    status="Unresolved"
)

show_results(unresolved_results)

wrong_status = [
    get_id(r)
    for r in unresolved_results
    if str(get_value(r, "status")).strip().casefold() != "unresolved"
]

if wrong_status:
    print("FAIL: Non-Unresolved complaints returned:", wrong_status)
else:
    print("PASS: Every returned complaint is Unresolved.")

if not unresolved_results:
    print("NOTE: No results returned; status filtering could not be fully verified.")


# --------------------------------------------------
# TEST 3: Geographic filtering — within 2 km
# --------------------------------------------------
print("\n" + "=" * 55)
print("TEST 3: GEOGRAPHIC FILTERING — WITHIN 2 KM")
print("=" * 55)

# Prefer C101 as the center if it exists; otherwise use the first
# complaint with valid coordinates.
c101 = valid_coords[
    valid_coords["complaint_id"].astype(str) == "C101"
]

center_row = c101.iloc[0] if not c101.empty else valid_coords.iloc[0]

center_lat = float(center_row["latitude"])
center_lon = float(center_row["longitude"])

print(f"Center complaint: {center_row['complaint_id']}")
print(f"Center coordinates: ({center_lat}, {center_lon})")
print("Search radius: 2 km")

nearby_results = ir.search_nearby(
    query="pothole",
    center_lat=center_lat,
    center_lon=center_lon,
    radius_km=2,
    top_k=20,
    issue_type="Pothole"
)

show_results(nearby_results)

outside_radius = []

for result in nearby_results:
    lat = get_value(result, "latitude")
    lon = get_value(result, "longitude")

    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        outside_radius.append((get_id(result), "Missing coordinates"))
        continue

    distance = haversine_distance(
        center_lat,
        center_lon,
        float(lat),
        float(lon)
    )

    print(f"  -> {get_id(result)} is {distance:.3f} km from center")

    if distance > 2.0:
        outside_radius.append((get_id(result), round(distance, 3)))

if outside_radius:
    print("FAIL: Complaints outside the radius:", outside_radius)
else:
    print("PASS: Every returned complaint is within 2 km.")

if not nearby_results:
    print("NOTE: No results returned; geographic filtering could not be fully verified.")
