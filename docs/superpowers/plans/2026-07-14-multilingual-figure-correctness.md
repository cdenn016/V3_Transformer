# Multilingual Figure Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate zero-width trajectory warnings, bind Arabic runs to their cl100k caches, and render honest native Japanese and shaped right-to-left Arabic text without applying English-only linguistic diagnostics.

**Architecture:** Tokenizer provenance remains centralized in `vfe3.data.datasets`. A new focused `vfe3.viz.text` module owns dataset-language policy, Unicode font fallbacks, token-label sanitation, and final-boundary Arabic shaping; `figures.py` consumes those services while `report.py` supplies the dataset-aware policy. Controlled sidecars retain their schema but serialize explicit unavailable reasons for English-only taxonomies.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, tiktoken, arabic-reshaper, python-bidi, pytest, JUnit XML.

## Global Constraints

Do not change training configurations, cached token streams, model weights, objectives, checkpoint formats, or historical run artifacts.

Keep `wikitext-2`, `wikitext-103`, and `wiki-en` behavior compatible; disable English-only taxonomies only for `wiki-ja` and `wiki-ar`.

Transform Arabic only at the Matplotlib display boundary. Keep decoded logical text unchanged for classification, sorting, provenance, and data processing.

Use the shared cl100k registry for Arabic cache paths, vocabulary bounds, decoders, bits-per-character decoding, and artifact provenance.

Every test run must write JUnit XML, and reported counts must be read from that XML.

Update the existing `docs/2026-07-14-edits.md`; do not create a second dated edit file.

---

### Task 1: Correct Arabic tokenizer provenance

**Files:**
- Modify: `vfe3/data/datasets.py:21-46,277-294`
- Modify: `tests/test_data.py:1-65`
- Modify: `tests/test_fixes_20260709_data.py:329-343`

**Interfaces:**
- Consumes: dataset names passed to `_tokenizer_tag`, `cache_path`, `tokenizer_vocab_size`, and `get_tiktoken_decoder`.
- Produces: `_CL100K_DATASETS = ("wiki-ja", "wiki-en", "wiki-ar")`; Arabic cache paths tagged `tiktoken_cl100k`; Arabic vocabulary size `100277`; Arabic decoder encoding `cl100k_base`.

- [ ] **Step 1: Write failing provenance tests**

Add assertions that pin both multilingual datasets and a mocked decoder-selection test that does not download tokenizer tables:

```python
def test_wiki_ar_uses_cl100k_cache_and_vocab(tmp_path):
    assert _tokenizer_tag("wiki-ar") == "tiktoken_cl100k"
    assert tokenizer_vocab_size("wiki-ar") == 100277
    assert cache_path("wiki-ar", "validation", cache_dir=tmp_path).name == (
        "wiki-ar_validation_tiktoken_cl100k_tokens.pt"
    )


def test_multilingual_decoders_select_cl100k(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        get_encoding=lambda name: calls.append(name) or SimpleNamespace(
            decode=lambda ids: ",".join(str(i) for i in ids)
        )
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    assert get_tiktoken_decoder("wiki-ja")([1, 2]) == "1,2"
    assert get_tiktoken_decoder("wiki-ar")([3]) == "3"
    assert calls == ["cl100k_base", "cl100k_base"]
```

Extend `test_tokenizer_vocab_size_per_dataset` with `wiki-ar == 100277`.

- [ ] **Step 2: Run tests and verify the Arabic cases fail**

Run:

```powershell
python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py --junitxml=.pytest-task1-red.xml
```

Expected: the XML records failures showing `wiki-ar` resolves to `tiktoken`, `50257`, or a GPT-2-tagged cache.

- [ ] **Step 3: Add Arabic to the central cl100k registry**

Implement the minimal source change and update its decoder/cache docstrings:

```python
_CL100K_DATASETS = ("wiki-ja", "wiki-en", "wiki-ar")


def _tokenizer_tag(dataset: str) -> str:
    """Cache tokenizer tag: cl100k for multilingual wiki caches, GPT-2 otherwise."""
    return "tiktoken_cl100k" if dataset in _CL100K_DATASETS else "tiktoken"
```

- [ ] **Step 4: Run the focused tests and inspect XML**

Run:

```powershell
python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py --junitxml=.pytest-task1-green.xml
```

Expected: zero failures and zero errors in `.pytest-task1-green.xml`.

- [ ] **Step 5: Commit tokenizer provenance**

```powershell
git add vfe3/data/datasets.py tests/test_data.py tests/test_fixes_20260709_data.py
git commit -m "fix(data): bind Arabic wiki cache to cl100k"
```

### Task 2: Define finite step axes and font-safe loss notation

**Files:**
- Modify: `vfe3/viz/figures.py:530-545,609,796,918,962,989-995,1140,1488,1540`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Produces: `_set_finite_xlim(ax: object, values: object) -> bool`, returning whether finite limits were applied.
- Consumes: every step-indexed trajectory's x values before `_step_xaxis(ax)` is called.

- [ ] **Step 1: Write failing one-point and endpoint tests**

Import `_history_dashboard`, `_set_finite_xlim`, `plot_kappa_history`, `plot_kappa_block_trajectory`, and `plot_model_channel_terms`. Add:

```python
def test_finite_xlim_pads_one_point_without_warning():
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert _set_finite_xlim(ax, np.array([15000.0]))
    lo, hi = ax.get_xlim()
    assert lo < 15000.0 < hi
    plt.close(fig)


def test_finite_xlim_preserves_distinct_endpoints():
    fig, ax = plt.subplots()
    assert _set_finite_xlim(ax, np.array([100.0, 200.0]))
    assert ax.get_xlim() == pytest.approx((100.0, 200.0))
    plt.close(fig)
```

Add a parameterized warning-as-error test covering `plot_trajectory`, `plot_model_channel_terms`, `plot_grad_norm_decomposition`, `_history_dashboard`, `plot_kappa_history`, and `plot_kappa_block_trajectory` with one recorded step.

- [ ] **Step 2: Run the tests and verify the helper is missing and current plots warn**

Run:

```powershell
python -m pytest tests/test_viz.py -k "finite_xlim or one_point_step" --junitxml=.pytest-task2-red.xml
```

Expected: collection or assertion failures identify the missing helper and identical-limit warnings.

- [ ] **Step 3: Implement one shared finite-limit helper**

Add after `_step_xaxis`:

```python
def _set_finite_xlim(ax: object, values: object) -> bool:
    """Set finite, nondegenerate x limits; pad a singleton by at least one step."""
    finite = _np(values).reshape(-1).astype(float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return False
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        pad = max(1.0, abs(lo) * 0.01)
        lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo, hi)
    return True
```

Replace all seven data-derived step `set_xlim(min, max)` calls with `_set_finite_xlim(ax, values)` and retain `_step_xaxis(ax)` only when the helper returns true.

- [ ] **Step 4: Replace the calligraphic loss glyph**

Use plain mathematical `L` in the four M-step gradient labels:

```python
("grad_norm_mu",    r"$\|\nabla_\mu L\|_2$",    _CB[0], "mu"),
("grad_norm_sigma", r"$\|\nabla_\Sigma L\|_2$", _CB[1], "sigma"),
("grad_norm_phi",   r"$\|\nabla_\phi L\|_2$",   _CB[2], "phi"),
```

Set the y label to `r"pre-clip $\|\nabla_\theta L\|_2$ (per role)"`.

- [ ] **Step 5: Run visualization tests and inspect XML**

Run:

```powershell
python -m pytest tests/test_viz.py tests/test_reporting_additions.py --junitxml=.pytest-task2-green.xml
```

Expected: zero failures/errors and no identical-limit warning in pytest's warning summary.

- [ ] **Step 6: Commit axis and notation fixes**

```powershell
git add vfe3/viz/figures.py tests/test_viz.py
git commit -m "fix(viz): handle one-point step axes"
```

### Task 3: Add native-script text rendering

**Files:**
- Create: `vfe3/viz/text.py`
- Create: `tests/test_viz_text.py`
- Modify: `vfe3/viz/figures.py:20-58,1785-1802,2066-2073,3106-3117`
- Modify: `pyproject.toml:15-21`

**Interfaces:**
- Produces: `MULTILINGUAL_SANS_SERIF: tuple[str, ...]`; `supports_english_linguistic_taxonomies(dataset: str) -> bool`; `display_text(text: str) -> str`; `token_label(tid: int, max_chars: int = 12, decode: Optional[Callable] = None) -> str`.
- Consumes: logical decoded strings and token IDs. Arabic shaping is never used for classifier input.

- [ ] **Step 1: Write failing native-script tests**

Create `tests/test_viz_text.py`:

```python
import matplotlib.pyplot as plt

from vfe3.viz.text import (
    MULTILINGUAL_SANS_SERIF,
    display_text,
    supports_english_linguistic_taxonomies,
    token_label,
)


def test_japanese_display_text_is_unchanged():
    assert display_text("日本語の表現") == "日本語の表現"


def test_arabic_display_text_is_shaped_for_matplotlib():
    logical = "العربية"
    visual = display_text(logical)
    assert visual != logical
    assert any("\ufb50" <= ch <= "\ufeff" for ch in visual)


def test_token_label_rejects_invalid_decoded_fragment():
    assert token_label(17, decode=lambda ids: "�") == "17"


def test_font_fallbacks_and_dataset_policy_are_explicit():
    assert "Yu Gothic" in MULTILINGUAL_SANS_SERIF
    assert "Noto Sans Arabic" in MULTILINGUAL_SANS_SERIF
    assert supports_english_linguistic_taxonomies("wiki-en")
    assert not supports_english_linguistic_taxonomies("wiki-ja")
    assert not supports_english_linguistic_taxonomies("wiki-ar")
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run:

```powershell
python -m pytest tests/test_viz_text.py --junitxml=.pytest-task3-red.xml
```

Expected: collection fails because `vfe3.viz.text` does not exist.

- [ ] **Step 3: Declare and install shaping dependencies**

Extend the visualization extra:

```toml
viz = [
    "umap-learn",
    "scikit-learn>=1.3",
    "scipy",
    "arabic-reshaper",
    "python-bidi",
]
```

Install the edited project for the isolated worktree:

```powershell
python -m pip install -e ".[viz]"
```

- [ ] **Step 4: Implement the focused Unicode display module**

Create `vfe3/viz/text.py` with typed helpers. Use Arabic Unicode ranges `0600-06FF`, `0750-077F`, `08A0-08FF`, `FB50-FDFF`, and `FE70-FEFF`. `display_text` must return non-Arabic input unchanged, otherwise call `arabic_reshaper.reshape(text)` and `bidi.algorithm.get_display(...)`; missing packages raise `RuntimeError("Arabic figure text requires the 'viz' dependencies arabic-reshaper and python-bidi")`. `token_label` must sanitize whitespace, replace embedded newline/tab markers, fall back to the numeric ID for replacement or nonprintable fragments, truncate logical text, and then call `display_text`.

- [ ] **Step 5: Wire fonts and decoded labels into figures**

Update `set_publication_style` with:

```python
"font.family":     "sans-serif",
"font.sans-serif": list(MULTILINGUAL_SANS_SERIF),
```

Make `_tok_label` delegate to `token_label`. Keep `_cluster_lift_labels` scoring and sorting on logical text, but pass each complete rendered legend row through `display_text` at `fig.text`. Add a `mark_subword_boundary` boolean to `_lift_label_display` so non-English callers can suppress the English middle-dot convention.

- [ ] **Step 6: Run native-text and vocabulary-figure tests**

Run:

```powershell
python -m pytest tests/test_viz_text.py tests/test_viz.py -k "viz_text or vocab or lift_label or publication_style" --junitxml=.pytest-task3-green.xml
```

Expected: zero failures/errors. A separate render probe on this machine must emit no missing-glyph warning for Japanese.

- [ ] **Step 7: Commit native-script rendering**

```powershell
git add pyproject.toml vfe3/viz/text.py vfe3/viz/figures.py tests/test_viz_text.py tests/test_viz.py
git commit -m "feat(viz): render Japanese and Arabic labels"
```

### Task 4: Disable English-only diagnostics for Japanese and Arabic

**Files:**
- Modify: `vfe3/viz/figures.py:1805-2117`
- Modify: `vfe3/viz/embedding_comparison.py:277-339`
- Modify: `vfe3/viz/report.py:151-177,251-258,291-299,378-410,487-588`
- Modify: `tests/test_controlled_umap_comparison_20260714.py`
- Modify: `tests/test_report.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Extends: `plot_belief_umap(..., english_linguistic_diagnostics: bool = True, ...)`.
- Extends: `controlled_embedding_record(..., taxonomy_unavailable_reason: Optional[str] = None) -> dict`.
- Consumes: `supports_english_linguistic_taxonomies(dataset)` from Task 3.

- [ ] **Step 1: Write failing sidecar-reason and report-policy tests**

Add a controlled-record test that passes no BPE/function labels plus `taxonomy_unavailable_reason="English-only linguistic taxonomies disabled for wiki-ar"` and asserts that all four taxonomy silhouette/AMI values are null with that exact reason while position and sequence metrics remain populated.

Add a live report test with a small model/loader and a `RunArtifacts` config naming `wiki-ja`; monkeypatch `get_tiktoken_decoder` to a native-text fake decoder. Assert that `belief_category_separation.png` and `vocab_confusion.png` are absent while vocabulary heatmaps/readouts remain present. Retain the existing English figure-set assertion.

- [ ] **Step 2: Run the new tests and verify current code emits English diagnostics**

Run:

```powershell
python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_report.py -k "taxonomy or multilingual or wiki_ja or wiki_ar" --junitxml=.pytest-task4-red.xml
```

Expected: failures show the missing reason argument and unwanted category figures.

- [ ] **Step 3: Serialize explicit non-English taxonomy reasons**

Extend `controlled_embedding_record` with the optional reason. When either taxonomy label array is `None` and the reason is present, store `_null_metric(taxonomy_unavailable_reason)` for the corresponding native silhouette and adjusted mutual information. Do not change trustworthiness, neighbor overlap, cluster count/noise, position-quartile AMI, or sequence-identity AMI.

- [ ] **Step 4: Make UMAP labeling and footers language-aware**

Add `english_linguistic_diagnostics: bool = True` to `plot_belief_umap`. Only build BPE/function labels and the function/content silhouette when it is true. Pass `mark_subword_boundary=english_linguistic_diagnostics` to cluster-label rendering. When false, append `English linguistic taxonomies disabled` to the footer and pass the explicit unavailability reason into the controlled record.

- [ ] **Step 5: Wire the dataset policy through single-run and comparison reports**

In `generate_figures`, compute:

```python
english_linguistic_diagnostics = supports_english_linguistic_taxonomies(dataset)
```

Pass it to belief/model UMAP figures. Require it in the availability condition for `belief_category_separation` and `vocab_confusion`. In `vocab_comparison_figures`, generate category confusion only when every prepared dataset supports English linguistic taxonomies. Log that the figure was skipped because the taxonomy is unavailable, not because an input failed.

- [ ] **Step 6: Run controlled-UMAP, report, and visualization suites**

Run:

```powershell
python -m pytest tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_viz.py --junitxml=.pytest-task4-green.xml
```

Expected: zero failures/errors; English tests retain category figures; Japanese/Arabic tests omit them deliberately.

- [ ] **Step 7: Commit dataset-aware diagnostics**

```powershell
git add vfe3/viz/embedding_comparison.py vfe3/viz/figures.py vfe3/viz/report.py tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_viz.py
git commit -m "fix(viz): gate English taxonomies by dataset"
```

### Task 5: Verify artifacts, document results, and close the branch

**Files:**
- Modify: `docs/2026-07-14-edits.md`
- Test artifacts: task-owned `.pytest-*.xml` files, deleted before the final commit.

**Interfaces:**
- Consumes: all source and test changes from Tasks 1-4.
- Produces: exact verification counts in the dated edit record, a clean task branch, and the requested pushed/merged/fast-forwarded repository state.

- [ ] **Step 1: Run the focused regression suite**

Run:

```powershell
python -m pytest tests/test_data.py tests/test_fixes_20260709_data.py tests/test_viz_text.py tests/test_viz.py tests/test_controlled_umap_comparison_20260714.py tests/test_report.py tests/test_reporting_additions.py --junitxml=.pytest-multilingual-focused.xml
```

Expected: zero failures and zero errors in the XML.

- [ ] **Step 2: Render Japanese and Arabic artifact probes**

Generate one Japanese and one Arabic vocabulary-label PNG with `set_publication_style`. Capture warnings as errors. Confirm the Japanese glyphs render through the configured CJK fallback and inspect the Arabic image to confirm joined right-to-left glyphs. Delete probe images after inspection.

- [ ] **Step 3: Run the full repository suite**

Run:

```powershell
python -m pytest --junitxml=.pytest-multilingual-full.xml
```

Expected: zero failures and zero errors in the XML.

- [ ] **Step 4: Update the dated edit record**

Append implementation sections to `docs/2026-07-14-edits.md` covering tokenizer provenance, finite axes and loss notation, native-script rendering/dependencies, language-valid diagnostics, focused XML counts, full-suite XML counts, and visual artifact inspection. Use only counts read from the XML files.

- [ ] **Step 5: Remove task-owned verification artifacts and commit documentation**

Delete `.pytest-*.xml` and render probes after recording their counts. Then run `git status --short`, `git diff --check`, inspect the staged diff, and commit:

```powershell
git add docs/2026-07-14-edits.md
git commit -m "docs: record multilingual figure verification"
```

- [ ] **Step 6: Request review and complete the mandatory git lifecycle**

Run the repository review workflow, address actionable findings, fetch `origin`, and verify the branch still merges cleanly into the current `origin/main`. Push `codex/multilingual-figure-fixes-20260714`, merge it into `main`, push `main`, and fast-forward the live checkout only if its uncommitted files are not overwritten. Remove the temporary worktree and local task branch after confirming the remote refs. Report the task commit SHA, resulting `origin/main` SHA, XML counts, remote branch, merge, worktree removal, and the live checkout's exact final `git status --short`.
