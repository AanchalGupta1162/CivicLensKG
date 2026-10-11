"""Tests for Member 2: Hybrid Information Retrieval, Ranking, Explainability, and Duplicates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import knowledge_graph as kg  # noqa: E402
import retrieval as ret  # noqa: E402

CSV = ROOT / "complaints.csv"


@pytest.fixture(scope="module")
def G():
    return kg.load_graph(CSV)


@pytest.fixture(scope="module")
def df():
    return kg.load_complaints(CSV)


@pytest.fixture
def mock_retriever():
    """Fast, deterministic retriever using MockDenseEncoder for offline testing."""
    sample_data = [
        {
            "complaint_id": "T001",
            "issue_type": "Pothole",
            "description": "Large dangerous pothole near the central bus stop",
            "area": "Andheri",
            "road_name": "Main Road",
            "road_importance": "Major",
            "department": "Roads",
            "status": "Unresolved",
        },
        {
            "complaint_id": "T002",
            "issue_type": "Pothole",
            "description": "Big crater close to the main bus station slowing traffic",
            "area": "Andheri",
            "road_name": "Main Road",
            "road_importance": "Major",
            "department": "Roads",
            "status": "Unresolved",
        },
        {
            "complaint_id": "T003",
            "issue_type": "Garbage",
            "description": "Overflowing trash and rotten waste near food market",
            "area": "Bandra",
            "road_name": "Market Road",
            "road_importance": "Minor",
            "department": "Sanitation",
            "status": "Unresolved",
        },
        {
            "complaint_id": "T004",
            "issue_type": "Streetlight",
            "description": "Dark streetlight pole not turning on at night",
            "area": "Kurla",
            "road_name": "Station Road",
            "road_importance": "Minor",
            "department": "Streetlight",
            "status": "Resolved",
        },
    ]
    encoder = ret.MockDenseEncoder(dim=16, seed=42)
    return ret.ComplaintRetriever(complaints=sample_data, alpha=0.4, encoder=encoder)


# ---------------------------------------------------------------------------
# 1. TF-IDF Lexical Retrieval
# ---------------------------------------------------------------------------
def test_tfidf_retrieval_sensible_matches(mock_retriever):
    """Test TF-IDF produces sensible lexical matches when keywords overlap."""
    results = mock_retriever.search("pothole bus stop", top_k=5, rank_by="hybrid")
    assert len(results) > 0
    top = results[0]
    assert top.complaint_id in ("T001", "T002")
    assert top.tfidf_score > 0.2
    assert "pothole" in top.description.lower() or "bus" in top.description.lower()


# ---------------------------------------------------------------------------
# 2. Dense Semantic Retrieval
# ---------------------------------------------------------------------------
def test_dense_retrieval_semantic_matches(mock_retriever):
    """Test dense retrieval yields scores when dense encoder is active."""
    results = mock_retriever.search("bus station crater", top_k=5)
    assert len(results) > 0
    for r in results:
        assert r.dense_score is not None
        assert 0.0 <= r.dense_score <= 1.0
        assert r.retrieval_method == "hybrid"


# ---------------------------------------------------------------------------
# 3. Hybrid Fusion Formula and Configured Weights
# ---------------------------------------------------------------------------
def test_hybrid_fusion_formula():
    """Verify H(q, d) = alpha * L(q, d) + (1 - alpha) * D(q, d)."""
    data = [{"complaint_id": "A1", "issue_type": "Pothole", "description": "pothole road"}]
    encoder = ret.MockDenseEncoder(dim=8, seed=1)
    retriever = ret.ComplaintRetriever(complaints=data, alpha=0.3, encoder=encoder)

    results = retriever.search("pothole", top_k=1, alpha=0.3)
    assert len(results) == 1
    r = results[0]
    expected_hybrid = 0.3 * r.tfidf_score + 0.7 * r.dense_score
    assert pytest.approx(r.hybrid_score, abs=1e-4) == round(expected_hybrid, 4)


def test_hybrid_fusion_extreme_weights():
    """Verify alpha=1.0 gives pure lexical and alpha=0.0 gives pure dense."""
    data = [{"complaint_id": "A1", "issue_type": "Pothole", "description": "pothole test"}]
    encoder = ret.MockDenseEncoder(dim=8, seed=1)
    retriever = ret.ComplaintRetriever(complaints=data, encoder=encoder)

    # Pure lexical
    res_lex = retriever.search("pothole", top_k=1, alpha=1.0)[0]
    assert pytest.approx(res_lex.hybrid_score, abs=1e-4) == res_lex.tfidf_score

    # Pure dense
    res_dense = retriever.search("pothole", top_k=1, alpha=0.0)[0]
    assert pytest.approx(res_dense.hybrid_score, abs=1e-4) == res_dense.dense_score


# ---------------------------------------------------------------------------
# 4. Weight Variation Changes Fused Score in Expected Direction
# ---------------------------------------------------------------------------
def test_weight_variation_moves_fused_score_in_expected_direction():
    """When TF-IDF > Dense, increasing alpha increases hybrid score."""
    data = [{"complaint_id": "A1", "issue_type": "Pothole", "description": "exact keyword match"}]
    encoder = ret.MockDenseEncoder(dim=16, seed=123)
    retriever = ret.ComplaintRetriever(complaints=data, encoder=encoder)

    res_low_alpha = retriever.search("exact keyword match", top_k=1, alpha=0.2)[0]
    res_high_alpha = retriever.search("exact keyword match", top_k=1, alpha=0.8)[0]

    # For exact query match, TF-IDF is close to 1.0
    if res_low_alpha.tfidf_score > res_low_alpha.dense_score:
        assert res_high_alpha.hybrid_score > res_low_alpha.hybrid_score
    elif res_low_alpha.dense_score > res_low_alpha.tfidf_score:
        assert res_high_alpha.hybrid_score < res_low_alpha.hybrid_score


# ---------------------------------------------------------------------------
# 5. Negative Dense Cosine Scores Handled Consistently
# ---------------------------------------------------------------------------
def test_negative_dense_cosine_normalization():
    """Test normalize_dense_score handles negative cosine values correctly."""
    # Clip mode
    assert ret.normalize_dense_score(-0.8, mode="clip") == 0.0
    assert ret.normalize_dense_score(-0.01, mode="clip") == 0.0
    assert ret.normalize_dense_score(0.0, mode="clip") == 0.0
    assert ret.normalize_dense_score(0.75, mode="clip") == 0.75
    assert ret.normalize_dense_score(1.5, mode="clip") == 1.0

    # Scale mode: (x + 1) / 2
    assert ret.normalize_dense_score(-1.0, mode="scale") == 0.0
    assert ret.normalize_dense_score(0.0, mode="scale") == 0.5
    assert ret.normalize_dense_score(1.0, mode="scale") == 1.0

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        ret.normalize_dense_score(0.5, mode="invalid_mode")


# ---------------------------------------------------------------------------
# 6. Results Returned in Descending Ranking Order
# ---------------------------------------------------------------------------
def test_results_descending_order(mock_retriever):
    """Verify search results are sorted in descending ranking_score order."""
    results = mock_retriever.search("pothole", top_k=10)
    for i in range(len(results) - 1):
        assert results[i].ranking_score >= results[i + 1].ranking_score


# ---------------------------------------------------------------------------
# 7. top_k is Respected
# ---------------------------------------------------------------------------
def test_top_k_respected(mock_retriever):
    """Verify top_k parameter strictly controls maximum returned items."""
    for k in (1, 2, 3, 10):
        results = mock_retriever.search("road", top_k=k)
        assert len(results) <= k
    assert mock_retriever.search("road", top_k=0) == []
    assert mock_retriever.search("road", top_k=-5) == []


# ---------------------------------------------------------------------------
# 8. Empty Queries Handled Appropriately
# ---------------------------------------------------------------------------
def test_empty_query_handling(mock_retriever):
    """Verify empty or whitespace-only queries do not crash and produce 0.0 scores."""
    for empty in ("", "   ", "\t\n"):
        results = mock_retriever.search(empty, top_k=5)
        assert len(results) <= 5
        for r in results:
            assert r.tfidf_score == 0.0
            assert r.hybrid_score == 0.0


# ---------------------------------------------------------------------------
# 9. Missing Descriptions and Optional Columns Do Not Crash
# ---------------------------------------------------------------------------
def test_missing_descriptions_and_optional_columns():
    """Verify handling of records with empty/missing descriptions or missing fields."""
    partial_data = [
        {"complaint_id": "P1", "issue_type": "Pothole", "description": ""},
        {"complaint_id": "P2", "issue_type": "", "description": "Some issue"},
        {"complaint_id": "P3"},  # completely empty fields
    ]
    encoder = ret.MockDenseEncoder(dim=8, seed=99)
    retriever = ret.ComplaintRetriever(complaints=partial_data, encoder=encoder)

    results = retriever.search("Pothole", top_k=5)
    assert len(results) == 3
    # P1 has issue_type "Pothole" so it matches
    p1 = [r for r in results if r.complaint_id == "P1"][0]
    assert p1.tfidf_score > 0.0


# ---------------------------------------------------------------------------
# 10. Empty Dataset Handled Correctly
# ---------------------------------------------------------------------------
def test_empty_dataset_handling():
    """Verify retriever handles completely empty complaints dataset."""
    empty_df = pd.DataFrame(columns=["complaint_id", "issue_type", "description"])
    encoder = ret.MockDenseEncoder(dim=8, seed=1)
    retriever = ret.ComplaintRetriever(complaints=empty_df, encoder=encoder)

    results = retriever.search("pothole", top_k=5)
    assert results == []

    dupes = retriever.find_duplicates(threshold=0.8)
    assert dupes == []


# ---------------------------------------------------------------------------
# 11. Explanation Fields Accurately Reflect Scores and Evidence
# ---------------------------------------------------------------------------
def test_explanation_reflects_actual_scores():
    """Verify explanation accurately explains the score components."""
    data = [
        {
            "complaint_id": "E1",
            "issue_type": "Pothole",
            "description": "deep crater in asphalt",
            "status": "Unresolved",
            "road_name": "Main Road",
            "road_importance": "Major",
        }
    ]
    encoder = ret.MockDenseEncoder(dim=16, seed=1)
    retriever = ret.ComplaintRetriever(complaints=data, encoder=encoder)

    # 1. Exact match query
    res = retriever.search("deep crater", top_k=1, status="Unresolved")[0]
    assert "Passed criteria" in res.explanation
    assert "status='Unresolved'" in res.explanation
    assert "lexical" in res.explanation.lower()
    assert "semantic" in res.explanation.lower()
    assert "hybrid text score" in res.explanation.lower()

    # 2. Query with zero lexical overlap
    res_no_lex = retriever.search("completely unrelated words xyz", top_k=1)[0]
    assert "No exact lexical term overlap" in res_no_lex.explanation


# ---------------------------------------------------------------------------
# 12. Retrieval Method Reported Correctly on Fallback
# ---------------------------------------------------------------------------
def test_lexical_fallback_when_dense_encoder_unavailable():
    """Verify fallback to lexical retrieval when encoder is None."""
    data = [{"complaint_id": "F1", "issue_type": "Garbage", "description": "uncollected trash"}]
    # Force encoder=None and prevent loading SentenceTransformer
    retriever = ret.ComplaintRetriever(
        complaints=data,
        encoder=None,
        model_name="nonexistent/model/path/xyz",
    )

    assert retriever.retrieval_method == "lexical_fallback"
    results = retriever.search("trash", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert r.retrieval_method == "lexical_fallback"
    assert r.dense_score is None
    assert r.hybrid_score == r.tfidf_score
    assert "lexical-only fallback" in r.explanation.lower()


# ---------------------------------------------------------------------------
# 13. Duplicate Detection Identifies Deliberately Similar Examples
# ---------------------------------------------------------------------------
def test_duplicate_detection_planted_examples(df, G):
    """Verify duplicate detection flags near-duplicate pairs (C101 & C102, C113 & C114, C122 & C131)."""
    retriever = ret.ComplaintRetriever(complaints=df, alpha=0.4)
    dupes = retriever.find_duplicates(threshold=0.80, G=G)

    pairs = {(d.complaint_id_1, d.complaint_id_2) for d in dupes}
    rev_pairs = {(d.complaint_id_2, d.complaint_id_1) for d in dupes}
    all_pairs = pairs | rev_pairs

    # C101 and C102 are planted duplicates
    assert ("C101", "C102") in all_pairs

    # Check candidate metadata
    c101_c102 = [d for d in dupes if {d.complaint_id_1, d.complaint_id_2} == {"C101", "C102"}][0]
    assert c101_c102.same_issue_type is True
    assert c101_c102.same_road is True
    assert c101_c102.confidence == "likely_duplicate"
    assert "duplicate" in c101_c102.explanation.lower()


def test_duplicate_detection_rejects_different_roads_and_landmarks(df):
    """Regression test: semantic similarity alone must not establish duplicates.
    
    C110 (Vile Parle, S.V. Road, near school) and C179 (Malad, Marve Road, near school)
    have high semantic similarity (~0.914) but are >10 km apart on different roads.
    They must NOT be flagged as likely duplicates.
    
    C109 (Station Road, station exit) and C158 (Station Road, market junction)
    describe different landmarks along Station Road and must be flagged as
    'related_issue', NOT 'likely_duplicate'.
    """
    retriever = ret.ComplaintRetriever(complaints=df)

    # 1. High text similarity on different roads (C110 vs C179)
    likely_dupes = retriever.find_duplicates(threshold=0.75, min_confidence="likely_duplicate")
    pairs = {(d.complaint_id_1, d.complaint_id_2) for d in likely_dupes}
    rev_pairs = {(d.complaint_id_2, d.complaint_id_1) for d in likely_dupes}
    all_likely = pairs | rev_pairs

    assert ("C110", "C179") not in all_likely

    # When inspecting all candidates without location filtering, C110/C179 is marked low_confidence
    all_candidates = retriever.find_duplicates(
        threshold=0.75,
        require_same_road=False,
        require_same_area=False,
        min_confidence=None,
    )
    c110_c179 = [
        d for d in all_candidates
        if {d.complaint_id_1, d.complaint_id_2} == {"C110", "C179"}
    ]
    assert len(c110_c179) == 1
    assert c110_c179[0].confidence == "low_confidence"
    assert "distinct locations" in c110_c179[0].explanation.lower()

    # 2. Same road but distinct landmarks (C109 station exit vs C158 market junction)
    assert ("C109", "C158") not in all_likely
    station_road_candidates = retriever.find_duplicates(
        threshold=0.75,
        require_same_road=True,
        min_confidence=None,
    )
    c109_c158 = [
        d for d in station_road_candidates
        if {d.complaint_id_1, d.complaint_id_2} == {"C109", "C158"}
    ]
    assert len(c109_c158) == 1
    assert c109_c158[0].confidence == "related_issue"
    assert "different spots or landmarks" in c109_c158[0].explanation.lower()


def test_haversine_distance_calculation():
    """Verify haversine distance calculation in meters."""
    # C101 (19.1197, 72.8468) and C102 (19.1199, 72.8470) are ~30.6 meters apart
    d = ret.haversine_distance(19.1197, 72.8468, 19.1199, 72.8470)
    assert 25.0 <= d <= 35.0

    # Same point distance is 0.0
    assert ret.haversine_distance(19.12, 72.84, 19.12, 72.84) == 0.0


def test_duplicate_detection_avoids_unrelated_complaints():
    """Verify clearly unrelated complaints are not flagged as duplicates."""
    data = [
        {
            "complaint_id": "U1",
            "issue_type": "Pothole",
            "description": "deep pothole on highway",
            "road_name": "Highway",
            "area": "Andheri",
        },
        {
            "complaint_id": "U2",
            "issue_type": "Streetlight",
            "description": "broken lamp fixture on dark street",
            "road_name": "Lane 4",
            "area": "Kurla",
        },
    ]
    encoder = ret.MockDenseEncoder(dim=16, seed=10)
    retriever = ret.ComplaintRetriever(complaints=data, encoder=encoder)
    dupes = retriever.find_duplicates(threshold=0.85)
    assert len(dupes) == 0


def test_find_duplicates_for_single_complaint(df):
    """Verify finding duplicates for a single complaint ID."""
    retriever = ret.ComplaintRetriever(complaints=df)
    c101_dupes = retriever.find_duplicates_for_complaint("C101", threshold=0.80)
    assert len(c101_dupes) > 0
    first_pair = {c101_dupes[0].complaint_id_1, c101_dupes[0].complaint_id_2}
    assert "C101" in first_pair


# ---------------------------------------------------------------------------
# 14. Missing Geographic Inputs Do Not Prevent Text Retrieval
# ---------------------------------------------------------------------------
def test_missing_geographic_inputs(mock_retriever):
    """Verify search succeeds when geo_scores is None, and uses geo when provided."""
    # 1. Without geo_scores (text-only)
    res_no_geo = mock_retriever.search("pothole", top_k=2, rank_by="hybrid")
    assert len(res_no_geo) > 0
    assert res_no_geo[0].geo_score is None

    # 2. Composite without geo_scores
    res_comp_no_geo = mock_retriever.search("pothole", top_k=2, rank_by="composite")
    assert res_comp_no_geo[0].geo_score is None
    # S = 0.5 * T + 0.0 + 0.2 * R
    expected_s = 0.5 * res_comp_no_geo[0].hybrid_score + 0.2 * res_comp_no_geo[0].road_score
    assert pytest.approx(res_comp_no_geo[0].ranking_score, abs=1e-4) == round(expected_s, 4)

    # 3. Composite with geo_scores provided (Member 3 simulation)
    geo_scores = {"T001": 0.95, "T002": 0.10}
    res_with_geo = mock_retriever.search(
        "pothole",
        top_k=2,
        rank_by="composite",
        geo_scores=geo_scores,
    )
    t001 = [r for r in res_with_geo if r.complaint_id == "T001"][0]
    assert t001.geo_score == 0.95
    # S = 0.5 * T + 0.3 * 0.95 + 0.2 * R
    expected_full = 0.5 * t001.hybrid_score + 0.3 * 0.95 + 0.2 * t001.road_score
    assert pytest.approx(t001.ranking_score, abs=1e-4) == round(expected_full, 4)


# ---------------------------------------------------------------------------
# 15. Hard Filters Strictly Exclude Non-Matching Complaints
# ---------------------------------------------------------------------------
def test_hard_filters_strictly_exclude(mock_retriever):
    """Verify complaints failing status or issue_type filter are never returned."""
    # Only Unresolved
    res_unresolved = mock_retriever.search("streetlight", status="Unresolved")
    for r in res_unresolved:
        assert r.status == "Unresolved"
    assert "T004" not in [r.complaint_id for r in res_unresolved]  # T004 is Resolved

    # Only Garbage
    res_garbage = mock_retriever.search("pothole", issue_type="Garbage")
    for r in res_garbage:
        assert r.issue_type == "Garbage"


# ---------------------------------------------------------------------------
# 16. Road Importance Ranking Factor (Major vs Minor)
# ---------------------------------------------------------------------------
def test_road_importance_composite_ranking(df, G):
    """Verify that on similar text relevance, Major roads outrank Minor roads under composite ranking."""
    retriever = ret.ComplaintRetriever(complaints=df, alpha=0.4)
    # C101 is Major road pothole in Andheri; C104 is Minor road pothole in Andheri
    results = retriever.search(
        "pothole",
        top_k=20,
        area="Andheri",
        status="Unresolved",
        rank_by="composite",
        G=G,
    )
    c101 = [r for r in results if r.complaint_id == "C101"][0]
    c104 = [r for r in results if r.complaint_id == "C104"][0]

    assert c101.road_score == 1.0  # Major
    assert c104.road_score == 0.5  # Minor
    # C101 should outrank C104 due to road importance
    assert c101.ranking_score > c104.ranking_score


# ---------------------------------------------------------------------------
# 17. Evaluation Benchmark Suite Runs Successfully
# ---------------------------------------------------------------------------
def test_evaluation_benchmark_suite(df):
    """Verify evaluation benchmark computes Precision@5, Recall@5, and latency."""
    retriever = ret.ComplaintRetriever(complaints=df, alpha=0.4)
    bench_results = ret.evaluate_retrieval(retriever, top_k=5)

    assert "mean_precision_at_k" in bench_results
    assert "mean_recall_at_k" in bench_results
    assert "mean_latency_ms" in bench_results
    assert bench_results["mean_recall_at_k"] > 0.70
    assert len(bench_results["queries"]) == len(ret.DEFAULT_EVALUATION_QUERIES)


# ---------------------------------------------------------------------------
# 18. Functional search_complaints and find_duplicate_complaints APIs
# ---------------------------------------------------------------------------
def test_functional_apis(df):
    """Test top-level search_complaints and find_duplicate_complaints functions."""
    # Search
    results = ret.search_complaints(
        query="garbage overflow near station",
        complaints=df,
        top_k=3,
        status="Unresolved",
    )
    assert len(results) <= 3
    assert all(r.status == "Unresolved" for r in results)

    # Duplicates
    dupes = ret.find_duplicate_complaints(complaints=df, threshold=0.75)
    assert isinstance(dupes, list)
    assert len(dupes) > 0
    assert all(d.status == "potential_duplicate" for d in dupes)


# ---------------------------------------------------------------------------
# 19. MRR@5 and Alpha Comparison Benchmark
# ---------------------------------------------------------------------------
def test_mrr_metric_and_alpha_comparison(df):
    """Verify MRR@5 calculation and compare_alpha_weights grid tool."""
    retriever = ret.ComplaintRetriever(complaints=df, alpha=0.4)
    bench = ret.evaluate_retrieval(retriever, top_k=5)

    assert "mean_mrr_at_k" in bench
    assert 0.0 <= bench["mean_mrr_at_k"] <= 1.0
    for q_m in bench["queries"]:
        assert "reciprocal_rank" in q_m
        assert 0.0 <= q_m["reciprocal_rank"] <= 1.0

    comparison = ret.compare_alpha_weights(retriever, alphas=[0.2, 0.4, 0.8])
    assert len(comparison) == 3
    assert all("mean_mrr_at_5" in row for row in comparison)


# ---------------------------------------------------------------------------
# 20. Road Name Aliases and Issue Synonym Expansion
# ---------------------------------------------------------------------------
def test_query_expansion_road_aliases_and_synonyms():
    """Verify query expansion converts abbreviations and synonyms into formal terms."""
    # WEH -> Western Express Highway
    exp_weh = ret.expand_query_for_lexical("WEH pothole")
    assert "Western Express Highway" in exp_weh

    # trash -> garbage
    exp_trash = ret.expand_query_for_lexical("trash near station")
    assert "garbage" in exp_trash

    # street light -> streetlight
    exp_sl = ret.expand_query_for_lexical("broken street light on road")
    assert "streetlight" in exp_sl


# ---------------------------------------------------------------------------
# 21. Fallback Explanation Integrity
# ---------------------------------------------------------------------------
def test_fallback_explanation_does_not_claim_hybrid_weights():
    """Verify that in fallback mode, explanation does not report hybrid weights."""
    data = [{"complaint_id": "FB1", "issue_type": "Pothole", "description": "pothole on road"}]
    retriever = ret.ComplaintRetriever(
        complaints=data,
        encoder=None,
        model_name="nonexistent/model",
    )
    assert retriever.retrieval_method == "lexical_fallback"
    results = retriever.search("pothole", top_k=1)
    res = results[0]
    assert res.retrieval_method == "lexical_fallback"
    assert "lexical-only fallback" in res.explanation
    assert "lexical weight:" not in res.explanation
