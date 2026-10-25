
from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

from distributed_ir import DistributedIR


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "complaints.csv"

st.set_page_config(
    page_title="CivicLens KG",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. INITIALIZATION AND HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)


@st.cache_resource
def get_ir(csv_path):
    return DistributedIR(csv_path=str(csv_path))


def to_dict(result):
    """Convert supported result objects to dictionaries."""
    if isinstance(result, dict):
        return result

    if hasattr(result, "to_dict"):
        return result.to_dict()

    if hasattr(result, "__dict__"):
        return vars(result)

    return {"result": str(result)}


def clean_value(value):
    """Convert values into readable strings."""
    if value is None:
        return "Not available"

    try:
        if pd.isna(value):
            return "Not available"
    except (TypeError, ValueError):
        pass

    return str(value)


# ============================================================
# 3. NATURAL-LANGUAGE FILTER EXTRACTION
# ============================================================

def extract_query_filters(query, df):
    """
    Detect department, issue type, status and area values
    by matching text against values present in the CSV.
    """

    text = query.lower()
    extracted = {}

    columns = {
        "department": "department",
        "issue_type": "issue_type",
        "status": "status",
        "area": "area",
    }

    for filter_name, column in columns.items():
        if column not in df.columns:
            continue

        values = df[column].dropna().astype(str).unique()

        for value in sorted(values, key=len, reverse=True):
            value = value.strip()

            if not value:
                continue

            pattern = (
                r"(?<!\w)"
                + re.escape(value.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                extracted[filter_name] = value
                break

    # Support common status synonyms.
    if "status" not in extracted and "status" in df.columns:
        status_values = df["status"].dropna().astype(str).unique()

        aliases = {
            "unresolved": ["open", "pending", "unresolved"],
            "resolved": ["resolved", "closed", "completed"],
            "in progress": [
                "in progress",
                "ongoing",
                "being handled",
            ],
        }

        for canonical, words in aliases.items():
            found = any(
                re.search(
                    r"(?<!\w)" + re.escape(word) + r"(?!\w)",
                    text,
                )
                for word in words
            )

            if found:
                matching_status = next(
                    (
                        value
                        for value in status_values
                        if value.strip().lower() == canonical
                    ),
                    None,
                )

                if matching_status:
                    extracted["status"] = matching_status
                    break

    return extracted


# ============================================================
# 4. COMPLAINT RESULT DISPLAY
# ============================================================

def display_results(results):
    """Display complaint search results as expandable cards."""

    if not results:
        st.info(
            "No matching complaints found. Try a broader query "
            "or adjust your filters."
        )
        return

    st.caption(f"{len(results)} complaint(s) returned")

    for index, item in enumerate(results, start=1):
        data = to_dict(item)

        complaint_id = clean_value(
            data.get("complaint_id", "Unknown")
        )
        issue = clean_value(data.get("issue_type"))
        status = clean_value(data.get("status"))
        department = clean_value(data.get("department"))
        area = clean_value(data.get("area"))
        description = clean_value(data.get("description"))

        with st.expander(
            f"{index}. {complaint_id} — {issue} · {status}",
            expanded=(index == 1),
        ):
            left, right = st.columns([2, 1])

            with left:
                st.write(description)
                st.write(f"**Department:** {department}")
                st.write(f"**Area:** {area}")
                st.write(
                    f"**Road:** {clean_value(data.get('road_name'))}"
                )
                st.write(
                    f"**Road importance:** "
                    f"{clean_value(data.get('road_importance'))}"
                )
                st.write(
                    f"**Created:** "
                    f"{clean_value(data.get('created_date'))}"
                )

            with right:
                score = data.get("ranking_score")

                if score is None:
                    score = data.get("hybrid_score")

                if score is not None:
                    try:
                        st.metric(
                            "Retrieval score",
                            f"{float(score):.4f}",
                        )
                    except (ValueError, TypeError):
                        st.write(f"**Retrieval score:** {score}")

                st.write(
                    f"**Method:** "
                    f"{clean_value(data.get('retrieval_method'))}"
                )

                image_path = data.get("image_path")

                if image_path:
                    path = Path(str(image_path))

                    if not path.is_absolute():
                        path = ROOT / path

                    if path.is_file():
                        st.image(
                            str(path),
                            caption=complaint_id,
                            use_container_width=True,
                        )

            explanation = data.get("explanation")

            if explanation:
                st.markdown("**Why this result matched**")
                st.write(explanation)

            geo_score = data.get("geo_score")

            if geo_score is not None:
                st.write(f"**Geographic score:** {geo_score}")

            reasoning = data.get("reasoning_paths")

            if reasoning:
                st.markdown("**Knowledge graph paths**")

                if isinstance(reasoning, (list, tuple)):
                    for path in reasoning:
                        st.write(f"- {path}")
                else:
                    st.write(reasoning)


# ============================================================
# 5. FOLIUM INTERACTIVE MAP
# ============================================================

def display_folium_map(results, center_lat, center_lon):
    """
    Display an interactive map with:
    - Search center marker
    - Clustered complaint markers
    - Status-based marker colors
    - Clickable complaint details
    """

    rows = [to_dict(result) for result in results]

    if not rows:
        st.info("No complaints to display on the map.")
        return

    map_df = pd.DataFrame(rows)

    required_coordinates = {"latitude", "longitude"}

    if not required_coordinates.issubset(map_df.columns):
        st.warning(
            "Search results do not contain latitude and longitude."
        )
        return

    map_df["latitude"] = pd.to_numeric(
        map_df["latitude"],
        errors="coerce",
    )

    map_df["longitude"] = pd.to_numeric(
        map_df["longitude"],
        errors="coerce",
    )

    map_df = map_df.dropna(
        subset=["latitude", "longitude"]
    )

    map_df = map_df[
        map_df["latitude"].between(-90, 90)
        & map_df["longitude"].between(-180, 180)
    ]

    if map_df.empty:
        st.info(
            "The returned complaints have no valid coordinates."
        )
        return

    # Create the map centered on the requested coordinates.
    complaint_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Search center marker.
    folium.Marker(
        location=[center_lat, center_lon],
        tooltip="Search center",
        popup="Selected search coordinates",
        icon=folium.Icon(
            color="blue",
            icon="info-sign",
        ),
    ).add_to(complaint_map)

    # Group nearby markers into clusters.
    cluster = MarkerCluster(
        name="Complaints",
    ).add_to(complaint_map)

    status_colors = {
        "resolved": "green",
        "unresolved": "red",
        "in progress": "orange",
    }

    for _, row in map_df.iterrows():
        complaint_id = clean_value(
            row.get("complaint_id")
        )
        issue = clean_value(row.get("issue_type"))
        status = clean_value(row.get("status"))
        area = clean_value(row.get("area"))
        department = clean_value(row.get("department"))
        description = clean_value(row.get("description"))
        road = clean_value(row.get("road_name"))
        importance = clean_value(row.get("road_importance"))

        color = status_colors.get(
            status.lower(),
            "gray",
        )

        # Escape CSV text before placing it inside HTML.
        popup_html = f"""
        <div style="width:260px; font-size:13px">
            <h4>{html.escape(complaint_id)}</h4>
            <b>Issue:</b> {html.escape(issue)}<br>
            <b>Description:</b> {html.escape(description)}<br>
            <b>Status:</b> {html.escape(status)}<br>
            <b>Department:</b> {html.escape(department)}<br>
            <b>Area:</b> {html.escape(area)}<br>
            <b>Road:</b> {html.escape(road)}<br>
            <b>Road importance:</b> {html.escape(importance)}
        </div>
        """

        folium.Marker(
            location=[
                float(row["latitude"]),
                float(row["longitude"]),
            ],
            tooltip=f"{complaint_id} — {issue}",
            popup=folium.Popup(
                popup_html,
                max_width=350,
            ),
            icon=folium.Icon(
                color=color,
                icon="info-sign",
            ),
        ).add_to(cluster)

    folium.LayerControl().add_to(complaint_map)

    # Avoid sending map interaction events back to Streamlit.
    # The map is still interactive: users can zoom, pan and open popups.
    st_folium(
        complaint_map,
        use_container_width=True,
        height=550,
        key="civiclens_nearby_map",
        returned_objects=[],
    )

    st.caption(
        "Red = unresolved | Green = resolved | "
        "Orange = in progress | Blue = search center"
    )


# ============================================================
# 6. LOAD DATASET AND RETRIEVAL SYSTEM
# ============================================================

st.title("🏙️ CivicLens KG")

st.subheader(
    "Distributed Information Retrieval for Civic Complaints"
)

st.markdown(
    """
    Search civic complaints across departments, explore relationships
    between complaints, locate nearby issues, and find visually similar
    complaint photographs.
    """
)

if not CSV_PATH.is_file():
    st.error(f"Dataset not found: {CSV_PATH}")
    st.stop()

try:
    df = load_data(str(CSV_PATH))
    ir = get_ir(str(CSV_PATH))

except Exception as exc:
    st.error(f"Could not initialize CivicLens KG: {exc}")
    st.stop()

required_columns = {
    "complaint_id",
    "issue_type",
    "description",
    "department",
    "status",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "Missing CSV columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# ============================================================
# 7. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Search controls")

departments = sorted(
    df["department"].dropna().astype(str).unique().tolist()
)

statuses = sorted(
    df["status"].dropna().astype(str).unique().tolist()
)

issue_types = sorted(
    df["issue_type"].dropna().astype(str).unique().tolist()
)

areas = (
    sorted(
        df["area"].dropna().astype(str).unique().tolist()
    )
    if "area" in df.columns
    else []
)

selected_departments = st.sidebar.multiselect(
    "Departments",
    options=departments,
    default=departments,
)

selected_status = st.sidebar.selectbox(
    "Complaint status",
    options=["All statuses"] + statuses,
)

selected_issue = st.sidebar.selectbox(
    "Issue type",
    options=["All issue types"] + issue_types,
)

selected_area = st.sidebar.selectbox(
    "Area",
    options=["All areas"] + areas,
)

top_k = st.sidebar.slider(
    "Maximum results",
    min_value=1,
    max_value=30,
    value=10,
)

status_filter = (
    None
    if selected_status == "All statuses"
    else selected_status
)

issue_filter = (
    None
    if selected_issue == "All issue types"
    else selected_issue
)

area_filter = (
    None
    if selected_area == "All areas"
    else selected_area
)


# ============================================================
# 8. DATASET OVERVIEW
# ============================================================

total_complaints = len(df)

unresolved_count = (
    df["status"]
    .astype(str)
    .str.lower()
    .eq("unresolved")
    .sum()
)

department_count = df["department"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total complaints", total_complaints)
col2.metric("Unresolved complaints", int(unresolved_count))
col3.metric("Departments", int(department_count))

st.divider()


# ============================================================
# 9. MAIN DASHBOARD TABS
# ============================================================

search_tab, geo_tab, graph_tab, image_tab, data_tab = st.tabs(
    [
        "🔎 Complaint Search",
        "📍 Nearby Complaints",
        "🕸️ Knowledge Graph",
        "🖼️ Image Similarity",
        "📊 Dataset",
    ]
)


# ============================================================
# 10. DISTRIBUTED COMPLAINT SEARCH
# ============================================================

with search_tab:
    st.header("Distributed Complaint Search")

    query = st.text_input(
        "Describe the civic issue",
        placeholder=(
            "e.g. unresolved pothole complaints in Andheri "
            "handled by Roads"
        ),
        key="text_search_query",
    )

    st.caption(
        "Mention the issue, status, area or department directly "
        "in your query. Detected values take precedence over "
        "the corresponding sidebar filters."
    )

    if st.button(
        "Search complaints",
        type="primary",
        key="run_text_search",
    ):
        if not query.strip():
            st.warning("Enter a search query first.")

        else:
            try:
                query_filters = extract_query_filters(
                    query,
                    df,
                )

                final_departments = (
                    [query_filters["department"]]
                    if "department" in query_filters
                    else selected_departments
                )

                final_status = query_filters.get(
                    "status",
                    status_filter,
                )

                final_issue = query_filters.get(
                    "issue_type",
                    issue_filter,
                )

                final_area = query_filters.get(
                    "area",
                    area_filter,
                )

                st.markdown(
                    "**Filters detected from your query**"
                )

                if query_filters:
                    st.json(query_filters)
                else:
                    st.caption(
                        "No structured filters detected; "
                        "using sidebar selections."
                    )

                if not final_departments:
                    st.warning(
                        "Select at least one department."
                    )

                else:
                    with st.spinner(
                        "Searching complaint indexes..."
                    ):
                        results = ir.search_all_indexes(
                            query=query.strip(),
                            departments=final_departments,
                            top_k=top_k,
                            status=final_status,
                            issue_type=final_issue,
                            area=final_area,
                        )

                    display_results(results)

            except Exception as exc:
                st.error(
                    f"Complaint search failed: {exc}"
                )


# ============================================================
# 11. NEARBY COMPLAINTS — PERSISTENT MAP FIX
# ============================================================

with geo_tab:
    st.header("Find Complaints Nearby")

    st.write(
        "Enter coordinates and a radius to retrieve complaints "
        "near a selected location."
    )

    default_lat = 19.1197
    default_lon = 72.8468

    if "latitude" in df.columns:
        valid_latitudes = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        ).dropna()

        valid_latitudes = valid_latitudes[
            valid_latitudes.between(-90, 90)
        ]

        if not valid_latitudes.empty:
            default_lat = float(valid_latitudes.iloc[0])

    if "longitude" in df.columns:
        valid_longitudes = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        ).dropna()

        valid_longitudes = valid_longitudes[
            valid_longitudes.between(-180, 180)
        ]

        if not valid_longitudes.empty:
            default_lon = float(valid_longitudes.iloc[0])

    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        center_lat = st.number_input(
            "Center latitude",
            min_value=-90.0,
            max_value=90.0,
            value=default_lat,
            format="%.6f",
            key="nearby_center_lat_input",
        )

    with geo_col2:
        center_lon = st.number_input(
            "Center longitude",
            min_value=-180.0,
            max_value=180.0,
            value=default_lon,
            format="%.6f",
            key="nearby_center_lon_input",
        )

    radius_km = st.slider(
        "Search radius (km)",
        min_value=0.5,
        max_value=20.0,
        value=2.0,
        step=0.5,
        key="nearby_radius_input",
    )

    geo_query = st.text_input(
        "Issue description",
        value="pothole",
        key="geo_search_query",
    )

    # Search button: perform the search and save the result.
    if st.button(
        "Find nearby complaints",
        type="primary",
        key="find_nearby_complaints",
    ):
        if not geo_query.strip():
            st.warning("Enter an issue description.")

        else:
            filters = {}

            if selected_departments:
                filters["department"] = selected_departments

            if status_filter is not None:
                filters["status"] = status_filter

            if issue_filter is not None:
                filters["issue_type"] = issue_filter

            if area_filter is not None:
                filters["area"] = area_filter

            try:
                with st.spinner(
                    "Searching within the selected radius..."
                ):
                    results = ir.search_nearby(
                        query=geo_query.strip(),
                        center_lat=center_lat,
                        center_lon=center_lon,
                        radius_km=radius_km,
                        top_k=top_k,
                        filters=filters,
                    )

                # Convert to dictionaries before saving.
                # This makes the data stable across reruns.
                st.session_state["nearby_results"] = [
                    to_dict(result)
                    for result in results
                ]

                st.session_state["nearby_center_lat"] = center_lat
                st.session_state["nearby_center_lon"] = center_lon
                st.session_state["nearby_search_done"] = True

                # Keep a record of the search parameters.
                st.session_state["nearby_search_query"] = geo_query.strip()
                st.session_state["nearby_search_radius"] = radius_km

            except Exception as exc:
                st.session_state["nearby_search_done"] = False
                st.error(
                    f"Nearby search failed: {exc}"
                )

    # IMPORTANT:
    # Render saved results outside the button block.
    # The map can now be recreated after Streamlit reruns.

    if st.session_state.get("nearby_search_done", False):
        saved_results = st.session_state.get(
            "nearby_results",
            [],
        )

        saved_lat = st.session_state.get(
            "nearby_center_lat",
            default_lat,
        )

        saved_lon = st.session_state.get(
            "nearby_center_lon",
            default_lon,
        )

        saved_query = st.session_state.get(
            "nearby_search_query",
            "",
        )

        saved_radius = st.session_state.get(
            "nearby_search_radius",
            radius_km,
        )

        st.markdown(
            f"**Search:** {saved_query} · "
            f"**Radius:** {saved_radius} km"
        )

        st.subheader("Interactive Complaint Map")

        display_folium_map(
            saved_results,
            center_lat=saved_lat,
            center_lon=saved_lon,
        )

        st.subheader("Complaint Details")

        display_results(saved_results)

        if st.button(
            "Clear nearby search results",
            key="clear_nearby_results",
        ):
            st.session_state["nearby_results"] = []
            st.session_state["nearby_search_done"] = False
            st.rerun()


# ============================================================
# 12. KNOWLEDGE GRAPH
# ============================================================

with graph_tab:
    st.header("Knowledge Graph Explorer")

    complaint_ids = sorted(
        df["complaint_id"].dropna().astype(str).tolist()
    )

    if not complaint_ids:
        st.warning("No complaint IDs are available.")

    else:
        default_index = (
            complaint_ids.index("C101")
            if "C101" in complaint_ids
            else 0
        )

        selected_complaint = st.selectbox(
            "Select a complaint",
            options=complaint_ids,
            index=default_index,
            key="graph_complaint_id",
        )

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            show_explanation = st.button(
                "Explain complaint relationships",
                type="primary",
                key="explain_graph",
            )

        with graph_col2:
            same_issue_only = st.checkbox(
                "Only show related complaints of the same issue",
                value=False,
                key="graph_same_issue_only",
            )

        if show_explanation:
            try:
                with st.spinner(
                    "Building knowledge graph..."
                ):
                    ir.build_knowledge_graph()

                explanation = ir.get_graph_explanation(
                    selected_complaint
                )

                st.subheader("Graph explanation")
                st.write(explanation)

                related = ir.get_related_complaints(
                    selected_complaint,
                    same_issue_only=same_issue_only,
                )

                st.subheader("Related complaints")

                if related:
                    related_df = pd.DataFrame(
                        [to_dict(item) for item in related]
                    )

                    st.dataframe(
                        related_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info(
                        "No related complaints were returned."
                    )

            except Exception as exc:
                st.error(
                    f"Knowledge graph operation failed: {exc}"
                )


# ============================================================
# 13. IMAGE SIMILARITY
# ============================================================

with image_tab:
    st.header("Visual Complaint Search")

    st.write(
        "Upload a complaint photograph to find visually similar "
        "images in the dataset."
    )

    backend = st.selectbox(
        "Image embedding backend",
        options=["handcrafted", "auto", "clip"],
        index=0,
        help=(
            "Handcrafted uses visual features. Auto chooses an "
            "available backend. CLIP requires its dependencies "
            "and model weights."
        ),
        key="image_backend",
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="similarity_upload",
    )

    exclude_self = st.checkbox(
        "Exclude the exact matching complaint if it exists",
        value=True,
        key="image_exclude_self",
    )

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Query image",
            width=300,
        )

    if st.button(
        "Build / load image index",
        key="build_image_index",
    ):
        try:
            with st.spinner(
                "Indexing complaint images..."
            ):
                ir.build_image_search_index(
                    backend=backend,
                    base_dir=ROOT,
                )

            st.session_state["civiclens_image_index_ready"] = True
            st.session_state["civiclens_image_backend"] = backend

            st.success("Image index is ready.")

        except Exception as exc:
            st.session_state["civiclens_image_index_ready"] = False

            st.error(
                f"Could not build image index: {exc}"
            )

    if st.button(
        "Find similar images",
        type="primary",
        key="find_similar_images",
    ):
        if uploaded_image is None:
            st.warning("Upload an image first.")

        elif not st.session_state.get(
            "civiclens_image_index_ready",
            False,
        ):
            st.warning(
                "Build the image index before searching."
            )

        elif st.session_state.get(
            "civiclens_image_backend"
        ) != backend:
            st.warning(
                "The selected backend differs from the indexed "
                "backend. Rebuild the image index first."
            )

        else:
            try:
                with st.spinner(
                    "Comparing image features..."
                ):
                    results = ir.search_similar_images(
                        query_image=uploaded_image.getvalue(),
                        k=top_k,
                    )

                if exclude_self:
                    st.caption(
                        "The source complaint ID of the uploaded "
                        "image is not automatically known. "
                        "Exact-self exclusion requires identifying "
                        "that ID first."
                    )

                if not results:
                    st.info(
                        "No similar images were found."
                    )

                else:
                    st.subheader(
                        "Similar complaint images"
                    )

                    for item in results:
                        result = to_dict(item)

                        cid = clean_value(
                            result.get("complaint_id")
                        )

                        score = result.get("score")
                        path_value = result.get("image_path")

                        with st.container(border=True):
                            img_col, details_col = st.columns([1, 2])

                            with img_col:
                                if path_value:
                                    path = Path(str(path_value))

                                    if not path.is_absolute():
                                        path = ROOT / path

                                    if path.is_file():
                                        st.image(
                                            str(path),
                                            use_container_width=True,
                                        )
                                    else:
                                        st.caption(
                                            "Image file not found."
                                        )

                            with details_col:
                                st.markdown(
                                    f"**Complaint: {cid}**"
                                )

                                if score is not None:
                                    try:
                                        st.metric(
                                            "Visual similarity score",
                                            f"{float(score):.4f}",
                                        )
                                    except (ValueError, TypeError):
                                        st.write(
                                            f"Score: {score}"
                                        )

                                st.write(
                                    f"**Backend:** "
                                    f"{clean_value(result.get('backend_label', result.get('backend')))}"
                                )

                                matching_row = df[
                                    df["complaint_id"].astype(str)
                                    == cid
                                ]

                                if not matching_row.empty:
                                    row = matching_row.iloc[0]

                                    st.write(
                                        f"**Issue:** "
                                        f"{clean_value(row.get('issue_type'))}"
                                    )

                                    st.write(
                                        f"**Description:** "
                                        f"{clean_value(row.get('description'))}"
                                    )

                                    st.write(
                                        f"**Department:** "
                                        f"{clean_value(row.get('department'))}"
                                    )

                                    st.write(
                                        f"**Status:** "
                                        f"{clean_value(row.get('status'))}"
                                    )

            except Exception as exc:
                st.error(
                    f"Image similarity search failed: {exc}"
                )


# ============================================================
# 14. DATASET EXPLORER
# ============================================================

with data_tab:
    st.header("Complaint Dataset")

    filtered_df = df.copy()

    if selected_departments:
        filtered_df = filtered_df[
            filtered_df["department"].astype(str).isin(
                selected_departments
            )
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if status_filter is not None:
        filtered_df = filtered_df[
            filtered_df["status"].astype(str) == status_filter
        ]

    if issue_filter is not None:
        filtered_df = filtered_df[
            filtered_df["issue_type"].astype(str) == issue_filter
        ]

    if area_filter is not None and "area" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["area"].astype(str) == area_filter
        ]

    st.caption(
        f"{len(filtered_df)} row(s) match the sidebar filters."
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered complaints as CSV",
        data=csv_bytes,
        file_name="civiclens_filtered_complaints.csv",
        mime="text/csv",
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "CivicLens KG | Distributed retrieval, geographic search, "
    "knowledge graph reasoning, and visual complaint similarity."
)
