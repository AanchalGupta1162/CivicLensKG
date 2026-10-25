
"""
CivicLens KG — Member 5
Distributed Information Retrieval and Integration

Uses the actual team module APIs:
    retrieval.py
    knowledge_graph.py
    geo.py
    multimedia.py
"""

from pathlib import Path

import pandas as pd

from retrieval import ComplaintRetriever

# Optional modules: text retrieval can still be used without these.
try:
    import knowledge_graph as kg
except ImportError:
    kg = None

try:
    import geo
except ImportError:
    geo = None

try:
    import multimedia
except ImportError:
    multimedia = None


DEPARTMENTS = ("Roads", "Sanitation", "Streetlight")


class DistributedIR:
    """Coordinate search across department-specific complaint indexes."""

    def __init__(
        self,
        csv_path="complaints.csv",
        alpha=0.4,
        encoder=None,
        parallel=False,
    ):
        self.csv_path = Path(csv_path)
        self.alpha = alpha
        self.encoder = encoder
        self.parallel = parallel

        self.complaints = self._load_data()
        self.indexes = {}
        self.index_counts = {}
        self.graph = None
        self.image_index = None

        self.build_indexes()

    # ---------------------------------------------------------
    # 1. LOAD AND VALIDATE DATA
    # ---------------------------------------------------------

    def _load_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path.resolve()}"
            )

        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.astype(str).str.strip()

        required = {
            "complaint_id",
            "issue_type",
            "description",
            "latitude",
            "longitude",
            "area",
            "road_name",
            "road_importance",
            "department",
            "status",
            "image_path",
            "created_date",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        if df["complaint_id"].isna().any():
            raise ValueError("Some complaint IDs are missing.")

        df["complaint_id"] = (
            df["complaint_id"].astype(str).str.strip()
        )

        if df["complaint_id"].eq("").any():
            raise ValueError("Some complaint IDs are blank.")

        if df["complaint_id"].duplicated().any():
            duplicates = df.loc[
                df["complaint_id"].duplicated(keep=False),
                "complaint_id",
            ].tolist()
            raise ValueError(
                f"Duplicate complaint IDs: {duplicates}"
            )

        df["department"] = (
            df["department"].astype("string").str.strip()
        )

        # Missing or invalid coordinates become NaN.
        df["latitude"] = pd.to_numeric(
            df["latitude"], errors="coerce"
        )
        df["longitude"] = pd.to_numeric(
            df["longitude"], errors="coerce"
        )

        return df

    # ---------------------------------------------------------
    # 2. BUILD THREE DEPARTMENT INDEXES
    # ---------------------------------------------------------

    def build_indexes(self):
        """
        Build one ComplaintRetriever for each non-empty department.

        The source remains one CSV. Each index receives a filtered
        DataFrame containing only its department's complaints.
        """
        self.indexes.clear()
        self.index_counts.clear()

        for department in DEPARTMENTS:
            subset = self.complaints[
                self.complaints["department"]
                .str.lower()
                .eq(department.lower())
            ].copy()

            if subset.empty:
                print(f"Skipping empty index: {department}")
                continue

            self.indexes[department] = ComplaintRetriever(
                complaints=subset,
                alpha=self.alpha,
                encoder=self.encoder,
            )

            self.index_counts[department] = len(subset)

        return self.indexes

    def index_stats(self):
        """Return the number of records in each non-empty index."""
        return self.index_counts.copy()

    # ---------------------------------------------------------
    # 3. DISTRIBUTED SEARCH AND GLOBAL MERGING
    # ---------------------------------------------------------

    @staticmethod
    def _score(result):
        """Read the score fields exposed by SearchResult."""
        value = getattr(result, "ranking_score", None)

        if value is None or pd.isna(value):
            value = getattr(result, "hybrid_score", 0.0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        return value if pd.notna(value) else 0.0

    def _search_one(
        self,
        department,
        query,
        top_k,
        filters,
        search_kwargs,
    ):
        """Run search against one department's retriever."""
        kwargs = dict(search_kwargs)

        # The team's retriever accepts G for graph-aware results.
        if self.graph is not None and "G" not in kwargs:
            kwargs["G"] = self.graph

        return self.indexes[department].search(
            query=query,
            top_k=top_k,
            filters=filters,
            **kwargs,
        )

    def search_all_indexes(
        self,
        query,
        top_k=10,
        filters=None,
        departments=None,
        parallel=None,
        **search_kwargs,
    ):
        """
        Search selected indexes and merge the returned SearchResults.

        filters:
            Filters passed to ComplaintRetriever.search(), such as
            status, issue_type, area, and road_importance.

        departments:
            Optional department name or list of department names.

        search_kwargs:
            Options supported by the team's retriever.search(),
            including rank_by, alpha, geo_scores, G,
            ranking_weights, and renormalize_missing.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        filters = dict(filters or {})

        # Department selection belongs at the coordinator level.
        # Do not forward it as a per-index search filter.
        filter_department = filters.pop("department", None)
        kw_department = search_kwargs.pop("department", None)

        requested = departments
        if requested is None:
            requested = (
                filter_department
                if filter_department is not None
                else kw_department
            )

        if requested is None:
            selected = list(self.indexes.keys())
        else:
            if isinstance(requested, str):
                requested = [requested]

            requested_names = {
                str(name).strip().lower() for name in requested
            }

            known_names = {name.lower() for name in DEPARTMENTS}
            unknown = requested_names - known_names

            if unknown:
                raise ValueError(
                    f"Unknown departments: {sorted(unknown)}"
                )

            selected = [
                name for name in self.indexes
                if name.lower() in requested_names
            ]

        if not selected:
            return []

        use_parallel = (
            self.parallel if parallel is None else parallel
        )

        collected = []

        if use_parallel and len(selected) > 1:
            # Parallel mode is optional. Sequential mode is the
            # safer default while validating shared model inference.
            from concurrent.futures import (
                ThreadPoolExecutor,
                as_completed,
            )

            with ThreadPoolExecutor(
                max_workers=len(selected)
            ) as executor:
                futures = [
                    executor.submit(
                        self._search_one,
                        department,
                        query.strip(),
                        top_k,
                        filters,
                        search_kwargs,
                    )
                    for department in selected
                ]

                for future in as_completed(futures):
                    collected.extend(future.result())
        else:
            for department in selected:
                collected.extend(
                    self._search_one(
                        department,
                        query.strip(),
                        top_k,
                        filters,
                        search_kwargs,
                    )
                )

        # Deduplicate by complaint ID.
        # Keep the highest-scoring result if an ID appears twice.
        unique = {}

        for result in collected:
            cid = str(result.complaint_id)
            previous = unique.get(cid)

            if (
                previous is None
                or self._score(result) > self._score(previous)
            ):
                unique[cid] = result

        # Global ordering across the collected results.
        merged = sorted(
            unique.values(),
            key=lambda result: (
                -self._score(result),
                -self._score_hybrid(result),
                str(result.complaint_id),
            ),
        )

        return merged[:top_k]

    @staticmethod
    def _score_hybrid(result):
        try:
            value = float(result.hybrid_score)
            return value if pd.notna(value) else 0.0
        except (AttributeError, TypeError, ValueError):
            return 0.0

    # ---------------------------------------------------------
    # 4. KNOWLEDGE GRAPH INTEGRATION
    # ---------------------------------------------------------

    def build_knowledge_graph(self):
        """Build the team's NetworkX graph from the complaint CSV."""
        if kg is None:
            raise ImportError(
                "Could not import knowledge_graph.py."
            )

        self.graph = kg.load_graph(str(self.csv_path))
        return self.graph

    def get_related_complaints(
        self,
        complaint_id,
        same_issue_only=False,
    ):
        """Find complaints connected by road or location."""
        if kg is None:
            raise ImportError(
                "Could not import knowledge_graph.py."
            )

        if self.graph is None:
            self.build_knowledge_graph()

        return kg.related_complaints(
            self.graph,
            str(complaint_id),
            same_issue_only=same_issue_only,
        )

    def get_graph_explanation(self, complaint_id):
        """Return the graph explanation for a complaint."""
        if kg is None:
            raise ImportError(
                "Could not import knowledge_graph.py."
            )

        if self.graph is None:
            self.build_knowledge_graph()

        return kg.graph_explanation(
            self.graph,
            str(complaint_id),
        )

    # ---------------------------------------------------------
    # 5. GEOGRAPHIC SEARCH
    # ---------------------------------------------------------

    def search_nearby(
        self,
        query,
        center_lat,
        center_lon,
        radius_km,
        top_k=10,
        filters=None,
        **search_kwargs,
    ):
        """
        Search complaints within a geographic radius.

        Uses geo.filter_by_radius() as a hard filter. It does not
        invent geographic relevance scores.
        """
        if geo is None:
            raise ImportError("Could not import geo.py.")

        nearby = geo.filter_by_radius(
            self.complaints,
            center_lat,
            center_lon,
            radius_km,
        )

        nearby_ids = set(
            nearby["complaint_id"].astype(str)
        )

        if not nearby_ids:
            return []

        # Retrieve enough candidates before applying the geographic
        # restriction; searching only top_k first could discard
        # nearby complaints that rank lower globally.
        candidates = self.search_all_indexes(
            query=query,
            top_k=max(len(self.complaints), top_k),
            filters=filters,
            **search_kwargs,
        )

        results = [
            result
            for result in candidates
            if str(result.complaint_id) in nearby_ids
        ]

        return results[:top_k]

    # ---------------------------------------------------------
    # 6. IMAGE SIMILARITY INTEGRATION
    # ---------------------------------------------------------

    def build_image_search_index(
        self,
        backend="auto",
        base_dir=None,
    ):
        """Build the image index using the team's multimedia module."""
        if multimedia is None:
            raise ImportError(
                "Could not import multimedia.py."
            )

        self.image_index = multimedia.build_image_index(
            csv_path=str(self.csv_path),
            backend=backend,
            base_dir=base_dir,
        )

        return self.image_index

    def search_similar_images(
        self,
        query_image,
        k=3,
        exclude_ids=None,
    ):
        """Find complaints with visually similar images."""
        if multimedia is None:
            raise ImportError(
                "Could not import multimedia.py."
            )

        if self.image_index is None:
            self.build_image_search_index()

        return multimedia.search_similar(
            self.image_index,
            query=query_image,
            k=k,
            exclude_ids=exclude_ids,
        )

    # ---------------------------------------------------------
    # 7. LOOK UP A COMPLAINT
    # ---------------------------------------------------------

    def get_complaint(self, complaint_id):
        """Return the original CSV record as a dictionary."""
        matches = self.complaints[
            self.complaints["complaint_id"].eq(
                str(complaint_id)
            )
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()


# -------------------------------------------------------------
# COMMAND-LINE SMOKE TEST
# -------------------------------------------------------------

if __name__ == "__main__":
    engine = DistributedIR("complaints.csv")

    print("\nDepartment index counts:")
    for department, count in engine.index_stats().items():
        print(f"  {department}: {count}")

    query = input("\nEnter a complaint query: ").strip()

    results = engine.search_all_indexes(
        query=query,
        top_k=10,
    )

    if not results:
        print("\nNo matching complaints found.")
    else:
        print("\nRanked results:")

        for result in results:
            print(
                f"\n{result.complaint_id} | "
                f"{result.issue_type} | "
                f"{result.department} | "
                f"{result.status}"
            )
            print(f"Description: {result.description}")
            print(f"Score: {result.ranking_score}")
            print(f"Explanation: {result.explanation}")
