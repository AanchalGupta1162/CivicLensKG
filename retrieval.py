"""CivicLens KG - Member 2: Hybrid Information Retrieval, Ranking, Explainability, and Duplicates.

Combines TF-IDF lexical retrieval and Sentence Transformer dense semantic search
(sentence-transformers/all-MiniLM-L6-v2) for explainable civic complaint retrieval,
multi-criteria relevance ranking, and duplicate complaint detection.

Architecture & Responsibilities (Member 2):
1. Lexical retrieval: TF-IDF vectorization over searchable complaint text with domain query expansion.
2. Dense semantic retrieval: all-MiniLM-L6-v2 embeddings with cosine similarity.
3. Hybrid score fusion: H(q, d) = alpha * L(q, d) + (1 - alpha) * D(q, d) (default alpha = 0.4).
4. Hard filtering: issue_type, status, department, area, etc. applied prior to ranking.
5. Relevance ranking: S = 0.5*T + 0.3*G + 0.2*R (interpret T as hybrid score; G is optional geo score;
   R is road importance derived from graph or metadata). Supports text-only or composite ranking.
6. Explainability: truthful, evidence-based explanations with matched filters, score decomposition,
   and optional knowledge graph reasoning paths.
7. Duplicate detection: multi-signal duplicate detection combining semantic similarity, lexical overlap,
   issue type, road, area, spatial distance (Haversine), and landmark evidence. Categorizes candidates
   into likely duplicates, related issues, and low-confidence matches.
8. Fallback: transparent, documented lexical-only fallback if the dense model is unavailable.

SDGs: 11 (Sustainable Cities) and 16 (Strong Institutions). All sample data is illustrative.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional import of Member 1's knowledge graph
try:
    import knowledge_graph as kg
except ImportError:  # pragma: no cover
    kg = None

logger = logging.getLogger(__name__)

# Global cache for the default SentenceTransformer model instance
_CACHED_DENSE_ENCODER: Any = None
_CACHED_ENCODER_MODEL_NAME: str | None = None

# Known Mumbai civic road abbreviations and aliases
ROAD_NAME_ALIASES: dict[str, str] = {
    "weh": "Western Express Highway",
    "western express": "Western Express Highway",
    "sv road": "S.V. Road",
    "sv rd": "S.V. Road",
    "s.v. road": "S.V. Road",
    "jvlr": "JVLR",
    "ak road": "Andheri-Kurla Road",
    "andheri kurla road": "Andheri-Kurla Road",
    "andheri kurla": "Andheri-Kurla Road",
    "stn road": "Station Road",
    "station rd": "Station Road",
    "4 bungalows": "Four Bungalows Road",
    "four bungalows": "Four Bungalows Road",
    "pali hill": "Pali Hill Road",
}

# Civic domain issue synonyms for lexical query expansion
ISSUE_SYNONYMS: dict[str, str] = {
    "crater": "pothole",
    "craters": "potholes",
    "trash": "garbage",
    "waste": "garbage",
    "debris": "garbage",
    "rubbish": "garbage",
    "litter": "garbage",
    "street light": "streetlight",
    "street lights": "streetlight",
    "lamp post": "streetlight",
    "lamppost": "streetlight",
    "lighting": "streetlight",
    "broken road": "damaged road",
    "caved in": "damaged road",
    "broken asphalt": "damaged road",
    "peeling asphalt": "damaged road",
}

# Known civic landmarks for disambiguating locations along corridors
KNOWN_CIVIC_LANDMARKS = [
    "bus stop", "bus depot", "station exit", "station approach", "station bridge",
    "station entrance", "station west", "station east", "station", "flyover bus stop",
    "flyover ramp", "flyover", "metro station", "market junction", "market lane",
    "market", "school gate", "school", "college", "temple", "hospital", "petrol pump",
    "post office", "court gate", "signal", "depot gate", "depot exit", "depot",
    "lake junction", "lake", "mall", "apartment gate", "railway bridge", "bridge",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """A single retrieved and ranked civic complaint with full explainability."""

    complaint_id: str
    issue_type: str
    description: str
    status: str
    area: str
    road_name: str
    road_importance: str
    department: str
    tfidf_score: float
    dense_score: float | None
    hybrid_score: float
    ranking_score: float
    retrieval_method: str = "hybrid"  # "hybrid" or "lexical_fallback"
    explanation: str = ""
    geo_score: float | None = None
    road_score: float | None = None
    matched_filters: dict[str, Any] = field(default_factory=dict)
    reasoning_paths: list[str] = field(default_factory=list)
    image_path: str = ""
    created_date: str = ""
    latitude: float | None = None
    longitude: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a plain dictionary for API/Streamlit integration."""
        return asdict(self)


@dataclass
class DuplicateCandidate:
    """Pair of complaints flagged as candidate duplicates with multi-signal evidence."""

    complaint_id_1: str
    complaint_id_2: str
    similarity: float
    dense_similarity: float | None
    tfidf_similarity: float
    issue_type_1: str
    issue_type_2: str
    description_1: str
    description_2: str
    same_issue_type: bool
    same_road: bool
    same_area: bool
    confidence: str = "likely_duplicate"  # "likely_duplicate", "related_issue", "low_confidence"
    distance_meters: float | None = None
    shared_landmarks: list[str] = field(default_factory=list)
    status: str = "potential_duplicate"
    explanation: str = ""
    graph_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert duplicate candidate to a dictionary."""
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Metrics for an individual test evaluation query."""

    query: str
    expected_ids: list[str]
    retrieved_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float
    retrieval_method: str


# ---------------------------------------------------------------------------
# Mock Encoder for offline/deterministic testing
# ---------------------------------------------------------------------------
class MockDenseEncoder:
    """Deterministic dense encoder for offline testing without torch/network.

    Generates unit-normalized vectors via word hashing. Words sharing character
    grams or stems will have correlated representations.
    """

    def __init__(self, dim: int = 32, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def encode(
        self,
        texts: Sequence[str] | str,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            clean = str(text).lower().replace(":", " ").replace("-", " ")
            tokens = [t for t in clean.split() if t]
            if not tokens:
                embeddings.append(vec)
                continue

            for token in tokens:
                h = hash((token, self.seed))
                idx = abs(h) % self.dim
                sign = 1.0 if (abs(h // self.dim) % 2 == 0) else -1.0
                vec[idx] += sign

            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers: Normalization, Distance & Text Construction
# ---------------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS coordinates in meters."""
    r_earth = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def extract_landmarks(text: str) -> list[str]:
    """Extract civic landmark indicators from complaint description text."""
    clean = str(text).lower()
    found: list[str] = []
    for lm in KNOWN_CIVIC_LANDMARKS:
        if lm in clean:
            found.append(lm)
    return found


def expand_query_for_lexical(query: str) -> str:
    """Expand query with road abbreviations and issue synonyms for TF-IDF matching."""
    expanded = query.strip()
    q_lower = expanded.lower()

    # Check for road aliases
    for alias, formal_name in ROAD_NAME_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, q_lower):
            expanded += f" {formal_name}"

    # Check for issue synonyms
    for syn, formal_issue in ISSUE_SYNONYMS.items():
        pattern = r"\b" + re.escape(syn) + r"\b"
        if re.search(pattern, q_lower):
            expanded += f" {formal_issue}"

    return expanded


def build_searchable_text(row: dict[str, Any] | pd.Series) -> str:
    """Build searchable representation from a complaint's fields.

    Combines issue_type and description to capture both topical category
    and fine-grained issue details.
    """
    if isinstance(row, pd.Series):
        issue = str(row.get("issue_type", "") or "").strip()
        desc = str(row.get("description", "") or "").strip()
    else:
        issue = str(row.get("issue_type") or "").strip()
        desc = str(row.get("description") or "").strip()

    if issue and desc:
        return f"{issue}: {desc}"
    elif desc:
        return desc
    elif issue:
        return issue
    return ""


def normalize_dense_score(score: float, mode: str = "clip") -> float:
    """Normalize dense cosine similarity into [0.0, 1.0].

    Cosine similarity for unit embeddings lies in [-1.0, 1.0].
    Modes:
      'clip': Non-negative relevance where negative similarity denotes 0 relevance.
              np.clip(score, 0.0, 1.0). (Default for IR relevance).
      'scale': Linear mapping from [-1.0, 1.0] to [0.0, 1.0]: (score + 1.0) / 2.0.
    """
    if math.isnan(score):
        return 0.0
    if mode == "clip":
        return float(np.clip(score, 0.0, 1.0))
    elif mode == "scale":
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown dense normalization mode '{mode}'. Choose 'clip' or 'scale'.")


def calculate_road_score(road_importance: str | None) -> float:
    """Normalize road importance to [0.0, 1.0].

    Major road -> 1.0 (higher priority arterial road)
    Minor road -> 0.5 (secondary residential / lane)
    Unknown -> 0.5
    """
    if not road_importance:
        return 0.5
    imp = str(road_importance).strip().capitalize()
    if imp == "Major":
        return 1.0
    elif imp == "Minor":
        return 0.5
    return 0.5


def get_sentence_transformer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Any | None:
    """Load or retrieve the cached SentenceTransformer model instance.

    Returns None if sentence-transformers is missing or model download fails.
    Does not misrepresent normal Hugging Face loading warnings as failures.
    """
    global _CACHED_DENSE_ENCODER, _CACHED_ENCODER_MODEL_NAME
    if _CACHED_DENSE_ENCODER is not None and _CACHED_ENCODER_MODEL_NAME == model_name:
        return _CACHED_DENSE_ENCODER

    try:
        from sentence_transformers import SentenceTransformer

        # Load public sentence-transformers model without requiring a Hugging Face token
        model = SentenceTransformer(model_name)
        _CACHED_DENSE_ENCODER = model
        _CACHED_ENCODER_MODEL_NAME = model_name
        return model
    except Exception as e:
        logger.warning(
            "Could not load SentenceTransformer model '%s': %s. "
            "Falling back to lexical-only retrieval.",
            model_name,
            e,
        )
        return None


# ---------------------------------------------------------------------------
# Reusable Complaint Retriever
# ---------------------------------------------------------------------------
class ComplaintRetriever:
    """Reusable index for hybrid lexical and dense semantic search over civic complaints.

    Maintains:
    - Preprocessed complaint records
    - Fitted TF-IDF matrix
    - Precomputed dense embeddings
    - Retrieval method indicator ('hybrid' or 'lexical_fallback')
    """

    def __init__(
        self,
        complaints: pd.DataFrame | list[dict[str, Any]] | str | Path,
        alpha: float = 0.4,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Any | None = None,
        dense_norm_mode: str = "clip",
        auto_build: bool = True,
    ) -> None:
        """Initialize retriever.

        Args:
            complaints: DataFrame, list of dicts, or CSV path of complaint records.
            alpha: Weight for lexical TF-IDF score in [0.0, 1.0]. Dense weight is (1 - alpha).
            model_name: Hugging Face model identifier for dense embeddings.
            encoder: Optional pre-initialized encoder object with .encode(texts) method.
                     If None, attempts to load SentenceTransformer(model_name).
            dense_norm_mode: 'clip' (default) or 'scale' for normalizing cosine similarity.
            auto_build: If True, builds index during initialization.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Lexical weight alpha must be between 0.0 and 1.0; got {alpha}")

        self.alpha = float(alpha)
        self.model_name = model_name
        self.dense_norm_mode = dense_norm_mode

        # Standardize complaints data into DataFrame
        self.complaints_df = self._standardize_complaints(complaints)

        # Encoder setup
        self.encoder = encoder
        self.retrieval_method = "hybrid"

        # Cached index structures
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any | None = None
        self.dense_embeddings: np.ndarray | None = None
        self.is_indexed: bool = False

        if auto_build:
            self.build_index()

    @staticmethod
    def _standardize_complaints(
        data: pd.DataFrame | list[dict[str, Any]] | str | Path,
    ) -> pd.DataFrame:
        """Convert input data to a standardized DataFrame with consistent string columns."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.is_file():
                raise FileNotFoundError(f"Complaints file not found: {path}")
            if kg is not None:
                df = kg.load_complaints(path)
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
                df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
                df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
                df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            for col in df.columns:
                if col in ("latitude", "longitude"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].fillna("").astype(str).str.strip()
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            for col in df.columns:
                if col in ("latitude", "longitude"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            raise TypeError(f"Unsupported complaints data type: {type(data)}")

        if "complaint_id" not in df.columns:
            df["complaint_id"] = [f"C{i+1:03d}" for i in range(len(df))]

        return df.reset_index(drop=True)

    def build_index(self) -> None:
        """Fit TF-IDF on corpus and precompute dense embeddings."""
        if len(self.complaints_df) == 0:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
            self.dense_embeddings = np.empty((0, 0), dtype=np.float32)
            self.is_indexed = True
            return

        corpus_texts = [
            build_searchable_text(row) for _, row in self.complaints_df.iterrows()
        ]

        # 1. Fit TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            norm="l2",
            token_pattern=r"(?u)\b\w+\b",
        )
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
        except ValueError:
            # Handle empty vocabulary corpus edge case
            self.tfidf_vectorizer = TfidfVectorizer(stop_words=None, norm="l2")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)

        # 2. Encode Dense Embeddings
        if self.encoder is None:
            self.encoder = get_sentence_transformer(self.model_name)

        if self.encoder is not None:
            try:
                embeddings = self.encoder.encode(
                    corpus_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings = np.array(embeddings, dtype=np.float32)
                # Ensure L2 unit norm
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self.dense_embeddings = embeddings / norms
                self.retrieval_method = "hybrid"
            except Exception as e:
                logger.warning("Dense encoding failed (%s). Falling back to lexical-only.", e)
                self.dense_embeddings = None
                self.retrieval_method = "lexical_fallback"
        else:
            self.dense_embeddings = None
            self.retrieval_method = "lexical_fallback"

        self.is_indexed = True

    def _get_query_dense_embedding(self, query: str) -> np.ndarray | None:
        """Encode query into a unit-normalized dense embedding."""
        if self.encoder is None or self.retrieval_method == "lexical_fallback":
            return None
        if not query.strip():
            return None

        try:
            q_emb = self.encoder.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            q_emb = np.array(q_emb[0], dtype=np.float32)
            norm = float(np.linalg.norm(q_emb))
            if norm > 1e-9:
                q_emb = q_emb / norm
            return q_emb
        except Exception as e:
            logger.warning("Query dense encoding failed: %s", e)
            return None

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        issue_type: str | None = None,
        status: str | None = None,
        department: str | None = None,
        area: str | None = None,
        road_importance: str | None = None,
        geo_scores: dict[str, float] | None = None,
        G: Any | None = None,
        alpha: float | None = None,
        rank_by: str = "hybrid",
        ranking_weights: dict[str, float] | None = None,
        renormalize_missing: bool = False,
    ) -> list[SearchResult]:
        """Search complaints using hybrid retrieval, hard filtering, and multi-factor ranking.

        Args:
            query: User's search query string.
            top_k: Maximum number of results to return.
            filters: Optional dictionary of {column: value_or_list_of_values}.
            issue_type: Optional hard filter on issue_type.
            status: Optional hard filter on status (e.g. 'Unresolved').
            department: Optional hard filter on department.
            area: Optional hard filter on area.
            road_importance: Optional hard filter on road_importance ('Major', 'Minor').
            geo_scores: Optional dictionary of {complaint_id: proximity_score in [0.0, 1.0]}.
                        Integration point for Member 3's geospatial module.
                        If None, geographic relevance is treated as unmeasured, geo_score remains
                        None in results, and no fake values are fabricated.
            G: Optional NetworkX knowledge graph from Member 1 (for road importance & graph paths).
            alpha: Optional override for lexical weight (default self.alpha).
            rank_by: 'hybrid' for text-only ranking, or 'composite' for S = 0.5T + 0.3G + 0.2R.
            ranking_weights: Custom weights for composite ranking, default {"text": 0.5, "geo": 0.3, "road": 0.2}.
            renormalize_missing: When rank_by='composite' and geo_scores is None, renormalize text+road weights.

        Returns:
            Ranked list of SearchResult objects with scores and explanations.
        """
        if not self.is_indexed:
            self.build_index()

        n_records = len(self.complaints_df)
        if n_records == 0 or top_k <= 0:
            return []

        active_alpha = self.alpha if alpha is None else float(alpha)
        if not (0.0 <= active_alpha <= 1.0):
            raise ValueError(f"Lexical weight alpha must be in [0.0, 1.0]; got {active_alpha}")

        # Assemble unified filter dict
        merged_filters: dict[str, Any] = dict(filters or {})
        if issue_type is not None:
            merged_filters["issue_type"] = issue_type
        if status is not None:
            merged_filters["status"] = status
        if department is not None:
            merged_filters["department"] = department
        if area is not None:
            merged_filters["area"] = area
        if road_importance is not None:
            merged_filters["road_importance"] = road_importance

        # 1. HARD FILTERS: identify valid indices
        survivor_indices: list[int] = []
        for idx, row in self.complaints_df.iterrows():
            passes = True
            for f_key, f_val in merged_filters.items():
                row_val = str(row.get(f_key, "")).strip()
                if isinstance(f_val, (list, tuple, set)):
                    allowed = {str(v).strip().lower() for v in f_val}
                    if row_val.lower() not in allowed:
                        passes = False
                        break
                else:
                    if row_val.lower() != str(f_val).strip().lower():
                        passes = False
                        break
            if passes:
                survivor_indices.append(idx)

        if not survivor_indices:
            return []

        # 2. LEXICAL RETRIEVAL (TF-IDF with domain query expansion)
        clean_query = query.strip()
        tfidf_scores = np.zeros(n_records, dtype=np.float32)
        if clean_query and self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            try:
                expanded_query = expand_query_for_lexical(clean_query)
                q_vec = self.tfidf_vectorizer.transform([expanded_query])
                if q_vec.nnz > 0:
                    raw_tfidf = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
                    tfidf_scores = np.clip(raw_tfidf, 0.0, 1.0).astype(np.float32)
            except Exception as e:
                logger.warning("TF-IDF transform failed: %s", e)

        # 3. DENSE SEMANTIC RETRIEVAL
        dense_scores: np.ndarray | None = None
        q_dense_emb = self._get_query_dense_embedding(clean_query)
        if q_dense_emb is not None and self.dense_embeddings is not None:
            raw_dense = np.dot(self.dense_embeddings, q_dense_emb)
            normalized_dense = [
                normalize_dense_score(s, mode=self.dense_norm_mode) for s in raw_dense
            ]
            dense_scores = np.array(normalized_dense, dtype=np.float32)
            current_method = "hybrid"
        else:
            dense_scores = None
            current_method = "lexical_fallback"

        # 4. HYBRID FUSION: H = alpha * L + (1 - alpha) * D
        hybrid_scores = np.zeros(n_records, dtype=np.float32)
        for i in survivor_indices:
            l_score = float(tfidf_scores[i])
            if dense_scores is not None:
                d_score = float(dense_scores[i])
                h_score = active_alpha * l_score + (1.0 - active_alpha) * d_score
            else:
                d_score = None
                h_score = l_score
            hybrid_scores[i] = float(np.clip(h_score, 0.0, 1.0))

        # 5. OVERALL RANKING (S = 0.5*T + 0.3*G + 0.2*R)
        default_weights = {"text": 0.5, "geo": 0.3, "road": 0.2}
        weights = dict(ranking_weights or default_weights)
        w_text = float(weights.get("text", 0.5))
        w_geo = float(weights.get("geo", 0.3))
        w_road = float(weights.get("road", 0.2))

        results: list[SearchResult] = []
        for i in survivor_indices:
            row = self.complaints_df.iloc[i]
            cid = str(row.get("complaint_id", ""))
            t_score = float(hybrid_scores[i])
            l_score = float(tfidf_scores[i])
            d_score = float(dense_scores[i]) if dense_scores is not None else None

            # Road importance score R
            road_imp_str = str(row.get("road_importance", "")).strip()
            if G is not None and kg is not None:
                try:
                    road_imp_str = kg.get_road_importance(G, cid)
                except Exception:
                    pass
            r_score = calculate_road_score(road_imp_str)

            # Geographic score G (from optional Member 3 input)
            g_score: float | None = None
            if geo_scores is not None:
                g_score = float(np.clip(geo_scores.get(cid, 0.0), 0.0, 1.0))

            # Calculate ranking_score S
            if rank_by == "composite":
                if g_score is not None:
                    s_score = w_text * t_score + w_geo * g_score + w_road * r_score
                else:
                    if renormalize_missing:
                        denom = w_text + w_road
                        s_score = (w_text * t_score + w_road * r_score) / denom if denom > 0 else t_score
                    else:
                        s_score = w_text * t_score + 0.0 + w_road * r_score
            else:
                s_score = t_score

            # Graph paths & reasoning if G is provided
            r_paths: list[str] = []
            if G is not None and kg is not None:
                try:
                    r_paths = kg.reasoning_paths(G, cid)
                except Exception:
                    pass

            result = SearchResult(
                complaint_id=cid,
                issue_type=str(row.get("issue_type", "")),
                description=str(row.get("description", "")),
                status=str(row.get("status", "")),
                area=str(row.get("area", "")),
                road_name=str(row.get("road_name", "")),
                road_importance=road_imp_str,
                department=str(row.get("department", "")),
                tfidf_score=round(l_score, 4),
                dense_score=round(d_score, 4) if d_score is not None else None,
                hybrid_score=round(t_score, 4),
                ranking_score=round(float(s_score), 4),
                retrieval_method=current_method,
                geo_score=round(g_score, 4) if g_score is not None else None,
                road_score=round(r_score, 4),
                matched_filters=merged_filters,
                reasoning_paths=r_paths,
                image_path=str(row.get("image_path", "")),
                created_date=str(row.get("created_date", "")),
                latitude=float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
                longitude=float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
                raw_data=row.to_dict(),
            )
            result.explanation = generate_explanation(
                result=result,
                alpha=active_alpha,
                rank_by=rank_by,
                G=G,
            )
            results.append(result)

        # 6. SORT RESULTS DESCENDING: rank_score, hybrid_score, tfidf_score, id
        results.sort(
            key=lambda r: (
                -r.ranking_score,
                -r.hybrid_score,
                -r.tfidf_score,
                r.complaint_id,
            )
        )

        return results[:top_k]

    def find_duplicates(
        self,
        threshold: float = 0.75,
        require_same_road: bool = True,
        require_same_area: bool = True,
        require_same_issue: bool = True,
        max_distance_meters: float | None = 250.0,
        min_confidence: str | None = "likely_duplicate",
        same_issue_only: bool = True,
        G: Any | None = None,
        top_k: int | None = None,
        metric: str = "dense",
    ) -> list[DuplicateCandidate]:
        """Detect duplicate complaints using multi-signal text, spatial, and landmark evidence.

        Args:
            threshold: Minimum similarity threshold (default 0.75).
            require_same_road: If True, candidate pairs must be on the same named road.
            require_same_area: If True, candidate pairs must be in the same geographic area.
            require_same_issue: If True, only compare complaints with matching issue_type.
            max_distance_meters: Maximum spatial distance in meters (default 250m) between GPS coordinates.
            min_confidence: 'likely_duplicate' (default), 'related_issue', or None (returns all with confidence tag).
            same_issue_only: Backward-compatible alias for require_same_issue.
            G: Optional NetworkX knowledge graph to fetch connecting graph paths.
            top_k: Maximum number of duplicate candidate pairs to return.
            metric: 'dense' (uses dense semantic embeddings) or 'hybrid' (uses hybrid similarity).

        Returns:
            List of DuplicateCandidate pairs sorted by descending similarity.
        """
        if not self.is_indexed:
            self.build_index()

        n = len(self.complaints_df)
        if n < 2:
            return []

        active_same_issue = require_same_issue and same_issue_only

        # Compute pairwise dense similarities
        if self.dense_embeddings is not None and len(self.dense_embeddings) == n:
            dense_sim_matrix = np.dot(self.dense_embeddings, self.dense_embeddings.T)
        else:
            dense_sim_matrix = None

        # Compute pairwise TF-IDF similarities
        if self.tfidf_matrix is not None:
            tfidf_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        else:
            tfidf_sim_matrix = np.zeros((n, n), dtype=np.float32)

        candidates: list[DuplicateCandidate] = []

        for i in range(n):
            row_i = self.complaints_df.iloc[i]
            cid_1 = str(row_i.get("complaint_id", ""))
            issue_1 = str(row_i.get("issue_type", ""))
            desc_1 = str(row_i.get("description", ""))
            road_1 = str(row_i.get("road_name", ""))
            area_1 = str(row_i.get("area", ""))
            lat_1 = float(row_i["latitude"]) if pd.notna(row_i.get("latitude")) else None
            lon_1 = float(row_i["longitude"]) if pd.notna(row_i.get("longitude")) else None

            for j in range(i + 1, n):
                row_j = self.complaints_df.iloc[j]
                cid_2 = str(row_j.get("complaint_id", ""))
                issue_2 = str(row_j.get("issue_type", ""))
                desc_2 = str(row_j.get("description", ""))
                road_2 = str(row_j.get("road_name", ""))
                area_2 = str(row_j.get("area", ""))
                lat_2 = float(row_j["latitude"]) if pd.notna(row_j.get("latitude")) else None
                lon_2 = float(row_j["longitude"]) if pd.notna(row_j.get("longitude")) else None

                same_issue = (issue_1.lower() == issue_2.lower())
                if active_same_issue and not same_issue:
                    continue

                same_road = bool(road_1 and road_2 and road_1.lower() == road_2.lower())
                same_area = bool(area_1 and area_2 and area_1.lower() == area_2.lower())

                # Location filters
                if require_same_road and not same_road:
                    continue
                if require_same_area and not same_area:
                    continue

                # Text similarity computation
                l_sim = float(np.clip(tfidf_sim_matrix[i, j], 0.0, 1.0))
                if dense_sim_matrix is not None:
                    d_raw = float(dense_sim_matrix[i, j])
                    d_sim = normalize_dense_score(d_raw, mode=self.dense_norm_mode)
                    h_sim = self.alpha * l_sim + (1.0 - self.alpha) * d_sim
                else:
                    d_sim = None
                    h_sim = l_sim

                chosen_sim = d_sim if metric == "dense" and d_sim is not None else h_sim

                if chosen_sim < threshold:
                    continue

                # Spatial distance calculation
                dist_m: float | None = None
                if lat_1 is not None and lon_1 is not None and lat_2 is not None and lon_2 is not None:
                    dist_m = haversine_distance(lat_1, lon_1, lat_2, lon_2)

                # Landmark evidence
                lm_1 = extract_landmarks(desc_1)
                lm_2 = extract_landmarks(desc_2)
                shared_lm = list(set(lm_1) & set(lm_2))
                conflicting_lm = bool((set(lm_1) - set(lm_2)) and (set(lm_2) - set(lm_1)))

                # Multi-signal confidence classification
                if same_issue and same_road and same_area:
                    if dist_m is not None and max_distance_meters is not None and dist_m > max_distance_meters:
                        confidence = "related_issue"
                    elif conflicting_lm:
                        confidence = "related_issue"  # Different landmarks along the same road
                    else:
                        confidence = "likely_duplicate"
                elif same_issue and (same_road or same_area):
                    confidence = "related_issue"
                else:
                    confidence = "low_confidence"

                # Filter by minimum confidence level if requested
                if min_confidence == "likely_duplicate" and confidence != "likely_duplicate":
                    continue
                elif min_confidence == "related_issue" and confidence not in ("likely_duplicate", "related_issue"):
                    continue

                # Graph connecting paths if available
                g_paths: list[str] = []
                if G is not None and kg is not None:
                    try:
                        rel_list = kg.related_complaints(G, cid_1, same_issue_only=False)
                        for r in rel_list:
                            if r["complaint_id"] == cid_2:
                                g_paths = r.get("paths", [])
                                break
                    except Exception:
                        pass

                # Synthesize explanation
                dist_str = f", ~{dist_m:.0f}m apart" if dist_m is not None else ""
                if confidence == "likely_duplicate":
                    lm_note = f" at shared landmark ({', '.join(shared_lm)})" if shared_lm else ""
                    explanation = (
                        f"Likely duplicate: High similarity ({chosen_sim:.3f}) and matching location "
                        f"('{road_1}', {area_1}{dist_str}{lm_note}). Both report {issue_1}."
                    )
                elif confidence == "related_issue":
                    explanation = (
                        f"Related incident: High similarity ({chosen_sim:.3f}) on same corridor ('{road_1}', {area_1}{dist_str}), "
                        f"but different spots or landmarks ('{', '.join(lm_1)}' vs '{', '.join(lm_2)}')."
                    )
                else:
                    explanation = (
                        f"Low-confidence match: High text similarity ({chosen_sim:.3f}) but distinct locations "
                        f"('{road_1}', {area_1} vs '{road_2}', {area_2}{dist_str}). Different incident."
                    )

                candidates.append(
                    DuplicateCandidate(
                        complaint_id_1=cid_1,
                        complaint_id_2=cid_2,
                        similarity=round(chosen_sim, 4),
                        dense_similarity=round(d_sim, 4) if d_sim is not None else None,
                        tfidf_similarity=round(l_sim, 4),
                        issue_type_1=issue_1,
                        issue_type_2=issue_2,
                        description_1=desc_1,
                        description_2=desc_2,
                        same_issue_type=same_issue,
                        same_road=same_road,
                        same_area=same_area,
                        confidence=confidence,
                        distance_meters=round(dist_m, 1) if dist_m is not None else None,
                        shared_landmarks=shared_lm,
                        status="potential_duplicate",
                        explanation=explanation,
                        graph_paths=g_paths,
                    )
                )

        candidates.sort(key=lambda c: (-c.similarity, c.complaint_id_1, c.complaint_id_2))
        return candidates[:top_k] if top_k is not None else candidates

    def find_duplicates_for_complaint(
        self,
        complaint_id: str,
        threshold: float = 0.75,
        require_same_road: bool = True,
        min_confidence: str | None = "likely_duplicate",
        same_issue_only: bool = True,
        G: Any | None = None,
        top_k: int = 5,
    ) -> list[DuplicateCandidate]:
        """Find candidate duplicates for a specific complaint ID."""
        all_candidates = self.find_duplicates(
            threshold=threshold,
            require_same_road=require_same_road,
            min_confidence=min_confidence,
            same_issue_only=same_issue_only,
            G=G,
            top_k=None,
        )
        matched = [
            c for c in all_candidates
            if c.complaint_id_1 == complaint_id or c.complaint_id_2 == complaint_id
        ]
        return matched[:top_k]


# ---------------------------------------------------------------------------
# Explainability Generator
# ---------------------------------------------------------------------------
def generate_explanation(
    result: SearchResult,
    alpha: float = 0.4,
    rank_by: str = "hybrid",
    G: Any | None = None,
) -> str:
    """Generate a factual, evidence-backed human-readable explanation for a retrieved result."""
    parts: list[str] = []

    # 1. Filters applied
    if result.matched_filters:
        filter_str = ", ".join(f"{k}='{v}'" for k, v in result.matched_filters.items())
        parts.append(f"Passed criteria ({filter_str}).")

    # 2. Lexical analysis
    l_score = result.tfidf_score
    if l_score >= 0.35:
        parts.append(f"Strong lexical overlap with query terms (TF-IDF: {l_score:.3f}).")
    elif l_score >= 0.05:
        parts.append(f"Partial lexical term overlap with query (TF-IDF: {l_score:.3f}).")
    elif l_score > 0.0:
        parts.append(f"Minor lexical term match (TF-IDF: {l_score:.3f}).")
    else:
        parts.append("No exact lexical term overlap with query.")

    # 3. Dense semantic analysis & fallback handling
    d_score = result.dense_score
    if result.retrieval_method == "hybrid" and d_score is not None:
        if d_score >= 0.70:
            parts.append(f"High semantic similarity to query intent (dense: {d_score:.3f}).")
        elif d_score >= 0.40:
            parts.append(f"Moderate semantic similarity to query intent (dense: {d_score:.3f}).")
        else:
            parts.append(f"Low semantic similarity to query intent (dense: {d_score:.3f}).")
        # 4. Hybrid score combination
        parts.append(
            f"Combined hybrid text score: {result.hybrid_score:.3f} "
            f"(lexical weight: {alpha:.2f}, dense weight: {1.0 - alpha:.2f})."
        )
    else:
        parts.append(
            f"Dense semantic search unavailable (retrieved via lexical-only fallback). "
            f"Text relevance score: {result.hybrid_score:.3f}."
        )

    # 5. Composite ranking factors
    if rank_by == "composite":
        factors = [f"text={result.hybrid_score:.2f}"]
        if result.road_score is not None:
            factors.append(f"road={result.road_score:.2f} ({result.road_importance})")
        if result.geo_score is not None:
            factors.append(f"geo={result.geo_score:.2f}")
        else:
            factors.append("geo=unmeasured (no geographic score provided)")
        parts.append(f"Composite ranking score: {result.ranking_score:.3f} ({', '.join(factors)}).")

    # 6. Knowledge graph context
    if G is not None and kg is not None:
        try:
            g_exp = kg.graph_explanation(G, result.complaint_id)
            parts.append(f"Graph context: {g_exp}")
        except Exception:
            pass

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Functional Public APIs
# ---------------------------------------------------------------------------
_DEFAULT_RETRIEVER: ComplaintRetriever | None = None
_DEFAULT_RETRIEVER_KEY: Any = None


def get_or_create_retriever(
    complaints: pd.DataFrame | list[dict[str, Any]] | str | Path | None = None,
    alpha: float = 0.4,
    encoder: Any | None = None,
) -> ComplaintRetriever:
    """Retrieve an existing ComplaintRetriever instance or create a new one."""
    global _DEFAULT_RETRIEVER, _DEFAULT_RETRIEVER_KEY

    if complaints is None:
        complaints = Path(__file__).resolve().with_name("complaints.csv")

    cache_key = (str(complaints) if isinstance(complaints, (str, Path)) else id(complaints), alpha)
    if _DEFAULT_RETRIEVER is not None and _DEFAULT_RETRIEVER_KEY == cache_key:
        return _DEFAULT_RETRIEVER

    retriever = ComplaintRetriever(complaints=complaints, alpha=alpha, encoder=encoder)
    _DEFAULT_RETRIEVER = retriever
    _DEFAULT_RETRIEVER_KEY = cache_key
    return retriever


def search_complaints(
    query: str,
    complaints: pd.DataFrame | list[dict[str, Any]] | str | Path | None = None,
    top_k: int = 10,
    issue_type: str | None = None,
    status: str | None = None,
    department: str | None = None,
    area: str | None = None,
    road_importance: str | None = None,
    geo_scores: dict[str, float] | None = None,
    G: Any | None = None,
    alpha: float = 0.4,
    rank_by: str = "hybrid",
    ranking_weights: dict[str, float] | None = None,
    retriever: ComplaintRetriever | None = None,
    encoder: Any | None = None,
) -> list[SearchResult]:
    """Search complaints using hybrid TF-IDF + dense semantic retrieval.

    This is the primary public interface for Member 2's retrieval module.

    Args:
        query: User's search query (e.g. 'unresolved pothole near bus stop').
        complaints: DataFrame, list of dicts, or CSV path (defaults to complaints.csv).
        top_k: Maximum number of results to return (default 10).
        issue_type: Hard filter for issue_type ('Pothole', 'Garbage', etc.).
        status: Hard filter for status ('Unresolved', 'In Progress', 'Resolved').
        department: Hard filter for department ('Roads', 'Sanitation', 'Streetlight').
        area: Hard filter for area ('Andheri', 'Kurla', etc.).
        road_importance: Hard filter for road importance ('Major', 'Minor').
        geo_scores: Optional dictionary of {complaint_id: proximity in [0, 1]}.
                    Integration contract with Member 3 (Geospatial module).
                    If None, geographic proximity is treated as unmeasured without fabricating values.
        G: Optional NetworkX knowledge graph.
        alpha: Lexical weight in [0.0, 1.0], dense weight is (1.0 - alpha). Default 0.4.
        rank_by: 'hybrid' (pure text relevance) or 'composite' (S = 0.5T + 0.3G + 0.2R).
        ranking_weights: Custom weights for composite ranking.
        retriever: Optional pre-built ComplaintRetriever instance.
        encoder: Optional injected encoder (e.g. MockDenseEncoder for testing).

    Returns:
        List of SearchResult objects sorted by descending relevance.
    """
    if retriever is None:
        retriever = get_or_create_retriever(complaints=complaints, alpha=alpha, encoder=encoder)

    return retriever.search(
        query=query,
        top_k=top_k,
        issue_type=issue_type,
        status=status,
        department=department,
        area=area,
        road_importance=road_importance,
        geo_scores=geo_scores,
        G=G,
        alpha=alpha,
        rank_by=rank_by,
        ranking_weights=ranking_weights,
    )


def find_duplicate_complaints(
    complaints: pd.DataFrame | list[dict[str, Any]] | str | Path | None = None,
    threshold: float = 0.75,
    require_same_road: bool = True,
    require_same_area: bool = True,
    require_same_issue: bool = True,
    max_distance_meters: float | None = 250.0,
    min_confidence: str | None = "likely_duplicate",
    same_issue_only: bool = True,
    G: Any | None = None,
    alpha: float = 0.4,
    retriever: ComplaintRetriever | None = None,
    encoder: Any | None = None,
    metric: str = "dense",
) -> list[DuplicateCandidate]:
    """Identify candidate duplicate complaints across the corpus with multi-signal evidence.

    Args:
        complaints: DataFrame, list of dicts, or CSV path.
        threshold: Minimum similarity threshold (default 0.75).
        require_same_road: Require identical road name (default True).
        require_same_area: Require identical area name (default True).
        require_same_issue: Require identical issue type (default True).
        max_distance_meters: Maximum GPS distance in meters (default 250m).
        min_confidence: 'likely_duplicate' (default), 'related_issue', or None (all candidates).
        same_issue_only: Alias for require_same_issue.
        G: Optional NetworkX knowledge graph.
        alpha: Weight for lexical term matching if hybrid metric is selected.
        retriever: Optional pre-built ComplaintRetriever.
        encoder: Optional injected encoder for offline testing.
        metric: 'dense' (semantic cosine similarity) or 'hybrid'.

    Returns:
        List of DuplicateCandidate pairs sorted by descending similarity.
    """
    if retriever is None:
        retriever = get_or_create_retriever(complaints=complaints, alpha=alpha, encoder=encoder)

    return retriever.find_duplicates(
        threshold=threshold,
        require_same_road=require_same_road,
        require_same_area=require_same_area,
        require_same_issue=require_same_issue,
        max_distance_meters=max_distance_meters,
        min_confidence=min_confidence,
        same_issue_only=same_issue_only,
        G=G,
        metric=metric,
    )


def find_duplicates_for_complaint(
    complaint_id: str,
    complaints: pd.DataFrame | list[dict[str, Any]] | str | Path | None = None,
    threshold: float = 0.75,
    require_same_road: bool = True,
    min_confidence: str | None = "likely_duplicate",
    same_issue_only: bool = True,
    G: Any | None = None,
    alpha: float = 0.4,
    retriever: ComplaintRetriever | None = None,
    top_k: int = 5,
) -> list[DuplicateCandidate]:
    """Find candidate duplicates for a single complaint."""
    if retriever is None:
        retriever = get_or_create_retriever(complaints=complaints, alpha=alpha)

    return retriever.find_duplicates_for_complaint(
        complaint_id=complaint_id,
        threshold=threshold,
        require_same_road=require_same_road,
        min_confidence=min_confidence,
        same_issue_only=same_issue_only,
        G=G,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Evaluation & Benchmarking
# ---------------------------------------------------------------------------
DEFAULT_EVALUATION_QUERIES = [
    {
        "query": "large pothole near bus stop",
        "expected_ids": ["C101", "C102", "C142", "C153"],
        "status": "Unresolved",
    },
    {
        "query": "overflowing garbage bin near station",
        "expected_ids": ["C122", "C131"],
        "status": "Unresolved",
    },
    {
        "query": "deep pothole near Kurla bus depot",
        "expected_ids": ["C113", "C114", "C170"],
        "status": "Unresolved",
    },
    {
        "query": "broken streetlight pole on highway",
        "expected_ids": ["C134", "C135"],
    },
    {
        "query": "asphalt peeling road damage",
        "expected_ids": ["C118", "C120"],
    },
    {
        "query": "dark stretch with broken streetlights near depot",
        "expected_ids": ["C138"],
    },
    {
        "query": "pothole near school gate on S.V. Road",
        "expected_ids": ["C110"],
    },
    {
        "query": "uncollected garbage pile on roadside",
        "expected_ids": ["C123", "C127"],
    },
]


def evaluate_retrieval(
    retriever: ComplaintRetriever,
    test_queries: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate retrieval performance across a test query suite.

    Computes Precision@k, Recall@k, MRR@k, and latency per query and overall.
    The benchmark is demonstration-oriented and uses verified ground truth from complaints.csv.
    """
    suite = test_queries or DEFAULT_EVALUATION_QUERIES
    metrics_list: list[EvaluationMetrics] = []

    for item in suite:
        q = item["query"]
        expected = item.get("expected_ids", [])
        status_filter = item.get("status")

        t0 = time.perf_counter()
        results = retriever.search(
            query=q,
            top_k=top_k,
            status=status_filter,
            **search_kwargs,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.complaint_id for r in results]
        expected_set = set(expected)
        retrieved_set = set(retrieved_ids)

        hits = len(expected_set & retrieved_set)
        p_at_k = hits / len(retrieved_ids) if retrieved_ids else 0.0
        r_at_k = hits / len(expected_set) if expected_set else 1.0

        # Mean Reciprocal Rank (MRR@k)
        rr = 0.0
        for rank_idx, cid in enumerate(retrieved_ids, 1):
            if cid in expected_set:
                rr = 1.0 / rank_idx
                break

        method = results[0].retrieval_method if results else retriever.retrieval_method

        metrics_list.append(
            EvaluationMetrics(
                query=q,
                expected_ids=expected,
                retrieved_ids=retrieved_ids,
                precision_at_k=round(p_at_k, 4),
                recall_at_k=round(r_at_k, 4),
                reciprocal_rank=round(rr, 4),
                latency_ms=round(latency, 2),
                retrieval_method=method,
            )
        )

    mean_p = float(np.mean([m.precision_at_k for m in metrics_list])) if metrics_list else 0.0
    mean_r = float(np.mean([m.recall_at_k for m in metrics_list])) if metrics_list else 0.0
    mean_mrr = float(np.mean([m.reciprocal_rank for m in metrics_list])) if metrics_list else 0.0
    mean_lat = float(np.mean([m.latency_ms for m in metrics_list])) if metrics_list else 0.0

    return {
        "top_k": top_k,
        "mean_precision_at_k": round(mean_p, 4),
        "mean_recall_at_k": round(mean_r, 4),
        "mean_mrr_at_k": round(mean_mrr, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "queries": [asdict(m) for m in metrics_list],
    }


def compare_alpha_weights(
    retriever: ComplaintRetriever,
    alphas: Sequence[float] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
    test_queries: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Compare retrieval performance across alternative lexical weights (alpha).

    Validates score fusion empirically rather than relying on intuition alone.
    """
    comparison: list[dict[str, Any]] = []
    for a in alphas:
        res = evaluate_retrieval(retriever, test_queries=test_queries, top_k=top_k, alpha=a)
        comparison.append({
            "alpha": round(float(a), 2),
            "lexical_weight": round(float(a), 2),
            "dense_weight": round(1.0 - float(a), 2),
            "mean_precision_at_5": res["mean_precision_at_k"],
            "mean_recall_at_5": res["mean_recall_at_k"],
            "mean_mrr_at_5": res["mean_mrr_at_k"],
            "mean_latency_ms": res["mean_latency_ms"],
        })
    return comparison


# ---------------------------------------------------------------------------
# Command Line Demonstration: python retrieval.py
# ---------------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser(description="CivicLens KG - Hybrid IR & Ranking Demo (Member 2)")
    p.add_argument("--csv", default=str(Path(__file__).resolve().with_name("complaints.csv")))
    p.add_argument("--query", default="unresolved pothole near bus stop", help="Search query")
    p.add_argument("--alpha", type=float, default=0.4, help="Lexical weight (0.0 to 1.0)")
    p.add_argument("--status", default=None, help="Filter by status (e.g. Unresolved)")
    p.add_argument("--issue", default=None, help="Filter by issue_type (e.g. Pothole)")
    p.add_argument("--rank-by", default="hybrid", choices=["hybrid", "composite"])
    p.add_argument("--top-k", type=int, default=5, help="Number of results")
    p.add_argument("--duplicates", action="store_true", help="Find potential duplicates")
    p.add_argument("--eval", action="store_true", help="Run evaluation benchmark suite")
    p.add_argument("--compare-alphas", action="store_true", help="Compare alternative alpha weights")
    args = p.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading complaints from {csv_path}...")
    retriever = ComplaintRetriever(complaints=csv_path, alpha=args.alpha)
    print(f"Index built ({len(retriever.complaints_df)} records, method={retriever.retrieval_method}).")

    G = None
    if kg is not None and csv_path.exists():
        try:
            G = kg.load_graph(csv_path)
            print("Knowledge graph linked successfully.")
        except Exception as e:
            print(f"Graph load note: {e}")

    # 1. Search Demo
    print(f"\n--- Search Demo: '{args.query}' (top_k={args.top_k}, alpha={args.alpha}) ---")
    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        status=args.status,
        issue_type=args.issue,
        G=G,
        rank_by=args.rank_by,
    )
    if not results:
        print("No complaints matched the query and filter criteria.")
    for idx, r in enumerate(results, 1):
        dense_str = f"{r.dense_score:.3f}" if r.dense_score is not None else "N/A"
        print(f"\n[{idx}] {r.complaint_id} ({r.issue_type} - {r.status}) on {r.road_name} ({r.road_importance})")
        print(f"    Description: {r.description}")
        print(f"    Scores: TF-IDF={r.tfidf_score:.3f} | Dense={dense_str} | Hybrid={r.hybrid_score:.3f} | Ranking={r.ranking_score:.3f}")
        print(f"    Explanation: {r.explanation}")

    # 2. Duplicate Detection Demo
    if args.duplicates:
        print("\n--- Multi-Signal Duplicate Detection Demo (threshold=0.75, likely duplicates) ---")
        dupes = retriever.find_duplicates(threshold=0.75, min_confidence="likely_duplicate", G=G)
        print(f"Found {len(dupes)} high-confidence candidate duplicate pair(s):")
        for d in dupes[:10]:
            dist_str = f", Dist: {d.distance_meters:.0f}m" if d.distance_meters is not None else ""
            dense_str = f"{d.dense_similarity:.3f}" if d.dense_similarity is not None else "N/A"
            print(f"  * {d.complaint_id_1} <-> {d.complaint_id_2} (Similarity: {d.similarity:.3f}, Dense: {dense_str}, TF-IDF: {d.tfidf_similarity:.3f}{dist_str})")
            print(f"    1: '{d.description_1}'")
            print(f"    2: '{d.description_2}'")
            print(f"    {d.explanation}")

    # 3. Evaluation Benchmark
    if args.eval:
        print("\n--- Evaluation Benchmark Suite (8 test queries, demonstration scope) ---")
        eval_res = evaluate_retrieval(retriever, top_k=5)
        print(f"Mean Precision@5: {eval_res['mean_precision_at_k']:.3f}")
        print(f"Mean Recall@5:    {eval_res['mean_recall_at_k']:.3f}")
        print(f"Mean MRR@5:       {eval_res['mean_mrr_at_k']:.3f}")
        print(f"Mean Latency:     {eval_res['mean_latency_ms']:.2f} ms")
        print("\nQuery Details:")
        for q in eval_res["queries"]:
            print(f"  Query: '{q['query']}'")
            print(f"    Expected:  {q['expected_ids']}")
            print(f"    Retrieved: {q['retrieved_ids']}")
            print(f"    P@5={q['precision_at_k']:.2f}, R@5={q['recall_at_k']:.2f}, MRR@5={q['reciprocal_rank']:.2f}, Latency={q['latency_ms']:.1f}ms")

    # 4. Compare Alpha Weights
    if args.compare_alphas:
        print("\n--- Empirical Comparison of Alpha Weights (TF-IDF vs Dense) ---")
        comparison = compare_alpha_weights(retriever)
        print(f"{'Alpha':<8}{'Lexical':<10}{'Dense':<10}{'P@5':<10}{'R@5':<10}{'MRR@5':<10}")
        print("-" * 58)
        for row in comparison:
            print(f"{row['alpha']:<8.2f}{row['lexical_weight']:<10.2f}{row['dense_weight']:<10.2f}{row['mean_precision_at_5']:<10.3f}{row['mean_recall_at_5']:<10.3f}{row['mean_mrr_at_5']:<10.3f}")


if __name__ == "__main__":
    _main()
