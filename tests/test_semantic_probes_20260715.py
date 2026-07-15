"""Deterministic tests for preregistered semantic probes."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from vfe3.viz import semantic_probes


def _toy_manifest() -> dict:
    return {
        "name": "toy_english_v1",
        "schema_version": 1,
        "concepts": [
            {"name": "king", "forms": [" king", "king"]},
            {"name": "queen", "forms": [" queen"]},
            {"name": "father", "forms": [" father", "father"]},
            {"name": "mother", "forms": [" mother", "mother"]},
            {"name": "dog", "forms": [" dog", "dog"]},
            {"name": "cat", "forms": [" cat", "cat"]},
        ],
        "fields": {
            "royalty": ["king", "queen"],
            "kinship": ["father", "mother"],
            "animals": ["dog", "cat"],
        },
        "pairs": [
            {
                "name": "king_queen",
                "concept_a": "king",
                "concept_b": "queen",
                "expectation": "close",
            },
            {
                "name": "father_mother",
                "concept_a": "father",
                "concept_b": "mother",
                "expectation": "close",
            },
            {
                "name": "king_dog",
                "concept_a": "king",
                "concept_b": "dog",
                "expectation": "control",
            },
            {
                "name": "queen_cat",
                "concept_a": "queen",
                "concept_b": "cat",
                "expectation": "control",
            },
            {
                "name": "king_father",
                "concept_a": "king",
                "concept_b": "father",
                "expectation": "descriptive",
            },
        ],
    }


def _decoder(mapping: dict[int, str]):
    def decode(ids: list[int]) -> str:
        return mapping[int(ids[0])]

    return decode


def _metric_value(record: dict, key: str) -> float:
    value = record[key]["value"]
    assert value is not None
    return float(value)


def test_default_manifest_identity_and_required_examples():
    manifest = semantic_probes.DEFAULT_SEMANTIC_MANIFEST

    semantic_probes.validate_manifest(manifest)

    assert manifest["name"] == "english_gpt2_semantic_probes"
    assert manifest["schema_version"] == 1
    pairs = {pair["name"]: pair for pair in manifest["pairs"]}
    assert pairs["king_queen"]["expectation"] == "close"
    assert pairs["father_mother"]["expectation"] == "close"
    assert pairs["king_jump"]["expectation"] == "control"
    assert pairs["king_father"]["expectation"] == "descriptive"


def test_default_manifest_forms_are_exact_single_gpt2_tokens():
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("gpt2")

    for concept in semantic_probes.DEFAULT_SEMANTIC_MANIFEST["concepts"]:
        for form in concept["forms"]:
            token_ids = encoding.encode(form)
            assert len(token_ids) == 1, (concept["name"], form, token_ids)
            assert encoding.decode(token_ids) == form


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_concept", "duplicate concept name"),
        ("duplicate_field_membership", "appears in multiple fields"),
        ("unknown_endpoint", "unknown concept"),
        ("identical_endpoint", "distinct concepts"),
        ("invalid_expectation", "invalid expectation"),
    ],
)
def test_manifest_validation_rejects_malformed_contracts(mutation: str, message: str):
    manifest = _toy_manifest()
    if mutation == "duplicate_concept":
        manifest["concepts"].append(copy.deepcopy(manifest["concepts"][0]))
    elif mutation == "duplicate_field_membership":
        manifest["fields"]["animals"].append("king")
    elif mutation == "unknown_endpoint":
        manifest["pairs"][0]["concept_b"] = "emperor"
    elif mutation == "identical_endpoint":
        manifest["pairs"][0]["concept_b"] = "king"
    else:
        manifest["pairs"][0]["expectation"] = "maybe"

    with pytest.raises(ValueError, match=message):
        semantic_probes.validate_manifest(manifest)


def test_resolve_concepts_accepts_multiple_exact_single_token_forms():
    token_ids = np.asarray([10, 11, 10, 12, 13, 13], dtype=np.int64)
    decode = _decoder({10: " king", 11: "king", 12: " queen", 13: "other"})

    resolution = semantic_probes.resolve_concepts(
        token_ids,
        decode,
        manifest=_toy_manifest(),
    )

    king = resolution["concepts"]["king"]
    queen = resolution["concepts"]["queen"]
    father = resolution["concepts"]["father"]
    assert king["resolved"] is True
    assert king["token_ids"] == [10, 11]
    assert king["decoded_forms"] == [" king", "king"]
    assert king["occurrence_indices"] == [0, 1, 2]
    assert king["occurrence_count"] == 3
    assert queen["token_ids"] == [12]
    assert father["resolved"] is False
    assert father["reason"] == "no accepted token form occurs in the bank"
    assert resolution["resolved_concept_count"] == 2


def test_unavailable_record_has_stable_null_schema():
    record = semantic_probes.unavailable_record(
        "token decoder unavailable",
        manifest=_toy_manifest(),
    )

    assert record["available"] is False
    assert record["reason"] == "token decoder unavailable"
    assert record["manifest"] == {"name": "toy_english_v1", "schema_version": 1}
    assert record["native_space"]["semantic_field_silhouette"] == {
        "value": None,
        "reason": "token decoder unavailable",
    }
    assert record["clusters"]["semantic_field_adjusted_mutual_information"]["value"] is None
    assert record["aggregate"]["control_to_close_distance_ratio"]["value"] is None


def _toy_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    token_ids = np.repeat(np.arange(1, 7, dtype=np.int64), 2)
    features = np.asarray(
        [
            [0.0, 0.0], [0.2, 0.0],
            [0.3, 0.0], [0.5, 0.0],
            [5.0, 0.0], [5.2, 0.0],
            [5.3, 0.0], [5.5, 0.0],
            [0.0, 10.0], [0.2, 10.0],
            [5.0, 10.0], [5.2, 10.0],
        ],
        dtype=float,
    )
    cluster_labels = np.asarray([0, -1, 0, 0, 1, 1, 1, -1, -1, -1, 2, 2], dtype=int)
    decode = _decoder(
        {
            1: " king",
            2: " queen",
            3: " father",
            4: " mother",
            5: " dog",
            6: " cat",
        }
    )
    return features, token_ids, cluster_labels, decode


def test_evaluator_uses_contextual_centroids_and_native_pair_distances():
    features, token_ids, cluster_labels, decode = _toy_bank()

    record = semantic_probes.evaluate_semantic_probes(
        features,
        token_ids,
        cluster_labels,
        decode,
        manifest=_toy_manifest(),
    )

    assert record["available"] is True
    assert record["native_space"]["concept_centroids"]["king"] == pytest.approx([0.1, 0.0])
    king_queen = record["pairs"]["king_queen"]
    assert _metric_value(king_queen, "centroid_distance") == pytest.approx(0.3)
    assert _metric_value(king_queen, "rank_a_to_b") == 1.0
    assert _metric_value(king_queen, "rank_b_to_a") == 1.0
    assert _metric_value(king_queen, "mean_reciprocal_rank") == 1.0
    assert _metric_value(king_queen, "hit_at_5") == 1.0
    assert _metric_value(king_queen, "hdbscan_co_membership") == pytest.approx(0.5)


def test_evaluator_reports_field_and_aggregate_metrics():
    features, token_ids, cluster_labels, decode = _toy_bank()

    record = semantic_probes.evaluate_semantic_probes(
        features,
        token_ids,
        cluster_labels,
        decode,
        manifest=_toy_manifest(),
    )

    royalty = record["native_space"]["fields"]["royalty"]
    assert royalty["resolved_concept_count"] == 2
    assert royalty["occurrence_count"] == 4
    assert _metric_value(royalty, "mean_within_distance") == pytest.approx(0.3)
    assert _metric_value(royalty, "mean_between_distance") > 1.0
    assert record["native_space"]["semantic_field_silhouette"]["value"] is not None
    assert record["clusters"]["semantic_field_adjusted_mutual_information"]["value"] is not None
    assert record["aggregate"]["expectations"]["close"]["resolved_pair_count"] == 2
    assert record["aggregate"]["expectations"]["control"]["resolved_pair_count"] == 2
    assert _metric_value(record["aggregate"], "control_to_close_distance_ratio") > 1.0
    json.dumps(record, allow_nan=False)


def test_missing_pair_endpoint_produces_explicit_null_metrics():
    features, token_ids, cluster_labels, decode = _toy_bank()
    decode = _decoder({1: " king", 2: "not queen", 3: " father", 4: " mother", 5: " dog", 6: " cat"})

    record = semantic_probes.evaluate_semantic_probes(
        features,
        token_ids,
        cluster_labels,
        decode,
        manifest=_toy_manifest(),
    )

    pair = record["pairs"]["king_queen"]
    assert pair["resolved"] is False
    assert pair["centroid_distance"] == {
        "value": None,
        "reason": "unresolved concept: queen",
    }
    assert pair["hdbscan_co_membership"]["value"] is None


def test_descriptive_pairs_do_not_change_confirmatory_aggregates():
    features, token_ids, cluster_labels, decode = _toy_bank()
    with_descriptive = _toy_manifest()
    without_descriptive = copy.deepcopy(with_descriptive)
    without_descriptive["pairs"] = [
        pair for pair in without_descriptive["pairs"]
        if pair["expectation"] != "descriptive"
    ]

    record_a = semantic_probes.evaluate_semantic_probes(
        features,
        token_ids,
        cluster_labels,
        decode,
        manifest=with_descriptive,
    )
    record_b = semantic_probes.evaluate_semantic_probes(
        features,
        token_ids,
        cluster_labels,
        decode,
        manifest=without_descriptive,
    )

    assert record_a["aggregate"]["expectations"]["close"] == record_b["aggregate"]["expectations"]["close"]
    assert record_a["aggregate"]["expectations"]["control"] == record_b["aggregate"]["expectations"]["control"]
    assert record_a["aggregate"]["control_to_close_distance_ratio"] == record_b["aggregate"]["control_to_close_distance_ratio"]
