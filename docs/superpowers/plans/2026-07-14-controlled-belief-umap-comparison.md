# Controlled Belief UMAP Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make finalized belief/model UMAP artifacts comparable across sequence lengths while preserving the current adaptive plot as an explicitly exploratory view.

**Architecture:** The extraction layer will select an exact deterministic token population and attach within-sequence positions. A focused `embedding_comparison` module will own comparison contracts, deterministic PCA clustering coordinates, multi-seed diagnostics, sidecar validation, and cross-run synthesis. `figures.py` will remain responsible for rendering, while `report.py` will make the controlled contract the finalization default and expose the cross-run sidecar driver.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn PCA/HDBSCAN/manifold and clustering metrics, umap-learn through the existing isolated `UMAPWorker`, Matplotlib, pytest, JSON with atomic same-directory replacement.

## Global Constraints

The default controlled bank contains exactly 16,384 validation tokens, with fixed display `n_neighbors=32`, `min_dist=0.1`, PCA initialization, and seeds `(0, 1, 2, 3, 4)`.

Controlled cluster discovery uses native features at dimension ten or below and deterministic PCA to ten dimensions otherwise; it never uses UMAP coordinates.

The existing `max_sequences` API and adaptive exploratory rendering remain available. Supplying `max_tokens` and `max_sequences` together raises `ValueError`.

Every controlled PNG has a same-stem JSON sidecar. Cross-run synthesis fails closed when token fingerprints or comparison-contract fields differ and never overlays independent UMAP coordinates.

Raw belief channels are labeled as gauge-fixed coordinate diagnostics, not gauge-invariant observables.

No training configuration, model objective, checkpoint, or existing run artifact is modified.

---

### Task 1: Exact-token belief banks

**Files:**
- Modify: `vfe3/viz/extract.py:284-364`
- Modify: `vfe3/viz/extract.py:922-990`
- Modify: `vfe3/viz/report.py:100-210`
- Create: `tests/test_controlled_umap_comparison_20260714.py`

**Interfaces:**
- Produces: `belief_bank(..., max_tokens: Optional[int] = None, max_sequences: Optional[int] = None)` and `model_channel_bank(...)` returning aligned `pos_idx` with every flattened bank tensor.
- Produces: `generate_figures(..., max_tokens: Optional[int] = None, max_sequences: Optional[int] = None)`, where both omitted selects 16,384 tokens.
- Consumes: existing unshuffled validation loaders and existing belief/model extraction paths without changing model execution.

- [ ] **Step 1: Write failing extractor tests.** Add tests that run the real tiny extraction fixtures, request a token cap that cuts through the final batch, require exact and aligned lengths, verify `pos_idx` repeats `0..N-1` for each sequence, and require conflicting or nonpositive caps to raise.

```python
def test_belief_bank_max_tokens_slices_every_aligned_field():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_tokens=11)
    assert {bank[k].shape[0] for k in ("mu", "sigma", "phi", "token_ids", "seq_idx", "pos_idx")} == {11}
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]


def test_belief_bank_rejects_ambiguous_population_caps():
    with pytest.raises(ValueError, match="max_tokens.*max_sequences"):
        extract.belief_bank(_tiny_model(), _token_batches(), max_tokens=8, max_sequences=2)
```

- [ ] **Step 2: Run the extractor tests and verify RED.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py -k "max_tokens or population_caps"`. Expected: failures because `max_tokens` and `pos_idx` do not exist.

- [ ] **Step 3: Implement exact token slicing.** Add a shared private cap validator and append positions beside the existing sequence IDs. Stop after collecting enough whole batches, concatenate once, and slice every bank field with a leading token dimension.

```python
def _validate_bank_caps(
    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> None:
    if max_tokens is not None and max_sequences is not None:
        raise ValueError("max_tokens and max_sequences are mutually exclusive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")
```

After concatenation, use `limit = min(max_tokens, total)` when a token cap is active and return `{key: value[:limit] for key, value in bank.items()}`. Build positions with `torch.arange(n, device=device).repeat(b)` so they align with row-major `(B, N, ...)` flattening.

- [ ] **Step 4: Implement report token budgeting.** When both caps are omitted, set `max_tokens=16_384`. Compute `n_batches = max(1, ceil(max_tokens / (cfg.batch_size * cfg.max_seq_len)))`; preserve the existing sequence calculation when `max_sequences` is explicitly supplied. Pass the selected cap into both bank extractors.

- [ ] **Step 5: Run tests and verify GREEN.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_model_channel_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-umap-task1-20260714.xml`. Expected XML: zero failures and zero errors.

- [ ] **Step 6: Commit Task 1.** Stage only the extractor, report, test, and dated edit files; inspect `git diff --cached`; commit `fix(viz): use exact token budgets for belief banks`.

### Task 2: Controlled clustering and machine-readable diagnostics

**Files:**
- Create: `vfe3/viz/embedding_comparison.py`
- Modify: `vfe3/viz/figures.py:684-712`
- Modify: `vfe3/viz/figures.py:1714-1749`
- Modify: `vfe3/viz/figures.py:1864-2023`
- Modify: `tests/test_controlled_umap_comparison_20260714.py`

**Interfaces:**
- Consumes: exact-token banks with `token_ids`, `seq_idx`, and `pos_idx`; channel features returned by `_belief_channel_features`; seeded embeddings returned by the existing `UMAPWorker`.
- Produces: `cluster_coordinates(features: np.ndarray, max_components: int = 10) -> tuple[np.ndarray, str]`.
- Produces: `controlled_embedding_record(...) -> Dict[str, object]` and `write_json_atomic(record, path) -> Path`.
- Produces: `plot_belief_umap(..., controlled: bool = False, seeds: Optional[Sequence[int]] = None, sidecar_path: Optional[str] = None)` while preserving the returned Matplotlib figure.

- [ ] **Step 1: Write failing pure-helper tests.** Require native coordinates at `D <= 10`, deterministic PCA at `D > 10`, stable SHA-256 token fingerprints, finite five-seed trustworthiness/neighborhood summaries on synthetic data, relative position quartiles, adjusted-mutual-information confound scores, and atomic JSON serialization.

```python
def test_cluster_coordinates_use_pca_not_umap_above_ten_dimensions():
    X = np.random.default_rng(0).normal(size=(80, 20))
    coords, desc = embedding_comparison.cluster_coordinates(X)
    assert coords.shape == (80, 10)
    assert desc == "PCA 10-D"
    assert np.array_equal(coords, embedding_comparison.cluster_coordinates(X)[0])


def test_controlled_record_reports_projection_and_confound_metrics(tmp_path):
    record = embedding_comparison.controlled_embedding_record(
        features=X,
        coords_by_seed={seed: coords[seed] for seed in range(5)},
        cluster_labels=cluster_labels,
        token_ids=token_ids,
        bpe_labels=bpe_labels,
        function_content_labels=fc_labels,
        seq_idx=seq_idx,
        pos_idx=pos_idx,
        seq_len=8,
        contract=contract,
    )
    assert record["projection"]["trustworthiness"]["count"] == 5
    assert np.isfinite(record["projection"]["neighbor_overlap"]["mean"])
    assert "position_quartile" in record["clusters"]["adjusted_mutual_information"]
```

- [ ] **Step 2: Run helper tests and verify RED.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py -k "cluster_coordinates or controlled_record or token_fingerprint"`. Expected: import or attribute failures because `embedding_comparison.py` does not exist.

- [ ] **Step 3: Implement the comparison helper module.** Use `sklearn.decomposition.PCA(svd_solver="full")` for deterministic reduction. Compute trustworthiness and embedding-neighbor overlap on one seed-zero deterministic subset of at most 2,000 points. Use `adjusted_mutual_info_score` after removing HDBSCAN noise labels. Return explicit `{value: None, reason: ...}` records for degenerate labels rather than dropping keys. Write JSON through a same-directory temporary file followed by `os.replace`.

```python
CONTROLLED_MAX_TOKENS       = 16_384
CONTROLLED_N_NEIGHBORS      = 32
CONTROLLED_MIN_DIST         = 0.1
CONTROLLED_SEEDS            = (0, 1, 2, 3, 4)
CONTROLLED_MIN_CLUSTER_SIZE = 128
CONTROLLED_MIN_SAMPLES      = 10


def cluster_coordinates(features: np.ndarray, max_components: int = 10) -> tuple[np.ndarray, str]:
    dim = min(max_components, features.shape[1], max(1, features.shape[0] - 1))
    if dim >= features.shape[1]:
        return features.copy(), f"native {features.shape[1]}-D"
    coords = PCA(n_components=dim, svd_solver="full").fit_transform(features)
    return coords, f"PCA {dim}-D"
```

- [ ] **Step 4: Extend `_cluster_embedding` without changing exploratory defaults.** Add optional explicit `min_cluster_size`, `min_samples`, and `cluster_selection_epsilon` arguments. When omitted, retain the current adaptive formulas byte-for-byte. Controlled mode passes `128`, `10`, and `0.0` against the equal 16,384-token bank.

- [ ] **Step 5: Add controlled rendering.** In controlled mode, use fixed display parameters, obtain seed-zero through seed-four UMAP embeddings, cluster native/PCA coordinates, compute the record, write the same-stem JSON, and label the title `controlled clusters`. Exploratory mode retains its current one-seed adaptive display and UMAP clustering route, with the title changed to `exploratory clusters`. Add `gauge-fixed coordinate diagnostic` to the controlled footer.

- [ ] **Step 6: Run tests and verify GREEN.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_viz.py tests/test_july13_root_fixes.py --junitxml=C:\tmp\vfe3-umap-task2-20260714.xml`. Expected XML: zero failures and zero errors.

- [ ] **Step 7: Commit Task 2.** Inspect the staged diff, then commit `feat(viz): add controlled UMAP diagnostics`.

### Task 3: Fail-closed cross-run synthesis

**Files:**
- Modify: `vfe3/viz/embedding_comparison.py`
- Modify: `vfe3/viz/report.py`
- Modify: `vfe3/viz/figures.py`
- Modify: `tests/test_controlled_umap_comparison_20260714.py`

**Interfaces:**
- Consumes: two or more controlled sidecar paths and one label per sidecar.
- Produces: `validate_comparison_records(records: Sequence[Mapping[str, object]]) -> None`.
- Produces: `compare_belief_umap_sidecars(sidecars, labels, *, json_path, figure_path) -> tuple[Path, Path]`.
- Produces: a comparison JSON and one metric-panel PNG; no raw UMAP coordinates enter either artifact.

- [ ] **Step 1: Write failing contract and rendering tests.** Build small sidecar fixtures. Require matching records to synthesize, token hash and display-setting mismatches to raise messages naming the exact field, and the resulting figure to contain metric panels but no UMAP-coordinate axes.

```python
def test_comparator_rejects_token_population_mismatch(tmp_path):
    left, right = _matching_records()
    right["sample"]["token_sha256"] = "different"
    with pytest.raises(ValueError, match="sample.token_sha256"):
        embedding_comparison.validate_comparison_records([left, right])


def test_cross_run_figure_contains_metrics_not_independent_coordinates(tmp_path):
    json_path, figure_path = report.compare_belief_umap_sidecars(
        sidecars=[left_path, right_path], labels=["N=256", "N=512"],
        json_path=tmp_path / "comparison.json", figure_path=tmp_path / "comparison.png")
    assert json_path.is_file() and figure_path.is_file()
    assert "coordinates" not in json.loads(json_path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run comparator tests and verify RED.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py -k "comparator or cross_run"`. Expected: attribute failures because the comparator is absent.

- [ ] **Step 3: Implement fail-closed record validation.** Compare `schema_version`, `mode`, token count and hash, channel, feature chart and dimension, UMAP settings, clustering settings, and seed list. Accumulate all mismatches as dotted field names and raise one `ValueError` containing the full set.

- [ ] **Step 4: Implement the cross-run metric artifact.** Persist labels and selected scalar metrics in JSON. Render grouped panels for native BPE/function-content silhouettes, trustworthiness, cluster noise, and adjusted mutual information with BPE category, function/content category, relative position, and sequence identity. Use no scatter coordinates.

- [ ] **Step 5: Wire controlled mode into production finalization.** `generate_figures` passes `controlled=True`, the five fixed seeds, and explicit sidecar paths for every belief and active model channel. Direct `plot_belief_umap` callers remain exploratory unless they opt in.

- [ ] **Step 6: Run tests and verify GREEN.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_viz.py tests/test_model_channel_diagnostics_2026_06_13.py --junitxml=C:\tmp\vfe3-umap-task3-20260714.xml`. Expected XML: zero failures and zero errors.

- [ ] **Step 7: Commit Task 3.** Inspect the staged diff, then commit `feat(viz): compare controlled belief geometry across runs`.

### Task 4: Documentation, regression verification, and repository closeout

**Files:**
- Modify: `docs/2026-07-14-edits.md`
- Modify as needed from test findings: files touched by Tasks 1 through 3 only

**Interfaces:**
- Consumes: the completed controlled figure path and the repository's pytest configuration.
- Produces: verified implementation record, pushed task branch, merged `origin/main`, and safe local-main handling.

- [ ] **Step 1: Update the dated edit record.** Add the final public APIs, controlled constants, sidecar schema, fail-closed behavior, regression commands, and machine-readable results to `docs/2026-07-14-edits.md`. Do not claim counts until reading the JUnit XML.

- [ ] **Step 2: Run focused verification.** Run `python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_viz.py tests/test_model_channel_diagnostics_2026_06_13.py tests/test_july13_root_fixes.py --junitxml=C:\tmp\vfe3-controlled-umap-focused-20260714.xml`. Read the XML `tests`, `failures`, and `errors` attributes.

- [ ] **Step 3: Run the full suite.** Run `python -m pytest --junitxml=C:\tmp\vfe3-controlled-umap-full-20260714.xml`. Read the XML attributes and retain the exact machine-reported counts.

- [ ] **Step 4: Verify the original defect contract.** Run the targeted test that constructs N=128 and N=512 report populations and read its assertions proving equal 16,384-token banks, fixed display settings, and non-UMAP clustering coordinates. Inspect one generated sidecar and comparison JSON directly.

- [ ] **Step 5: Inspect and commit.** Run `git diff --check`, inspect `git status --short` and the complete staged diff, then commit the documentation and any test-derived corrections with `docs: record controlled UMAP verification`.

- [ ] **Step 6: Push and merge.** Fetch and recheck `origin/main`, push `codex/umap-comparison-20260714`, merge it into `main`, push `main`, and fetch again to verify the resulting remote SHA. Do not alter the dirty live checkout.

- [ ] **Step 7: Safe local handling and cleanup.** Fast-forward the user's local `main` only if doing so cannot overwrite WIP. Since the live checkout was dirty at task start, leave it untouched unless Git proves a fast-forward is safe with every dirty path preserved. Remove the temporary worktree and local task branch after confirming the remote merge. Report the final `origin/main` SHA, commits, JUnit counts, worktree removal, and the live checkout's remaining user-owned status.
