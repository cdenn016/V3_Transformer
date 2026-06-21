# Remove VFE_2.0 Requirements from the V3 Repo

Date: 2026-06-21
Branch: `worktree-remove-vfe2-requirements` (worktree, fresh from `origin/main` @ 33f046f)

## Goal

The V3 repo stands on its own and surpasses VFE_2.0. Remove every framing that
positions V3 as *pinned to*, *a continuation of*, or *in parity with* VFE_2.0,
so the codebase reads as a self-standing system whose correctness standard is
its own golden regression suite — not equivalence to a predecessor.

## What VFE_2.0 actually is in this repo (findings)

There is **no live VFE_2.0 dependency**: no `import vfe2`, no pinned-checkout
path, no env var, nothing in `tests/conftest.py` but a device fixture. The
golden/parity tests carry their expected values **baked in as snapshots** — they
are already self-contained and never needed a VFE_2.0 checkout at runtime.

"VFE_2.0 requirements" are therefore entirely **documentary**: 45 mention sites
across ~25 files (comments, docstrings, and three Markdown files).

## Scope (locked with user)

- **Depth: "Framing + all comments."** Rewrite the three framing Markdown files
  AND scrub every in-code/test `VFE_2.0` provenance comment/docstring. **No**
  config toggle removed, **no** test deleted, **no** test value changed, **no**
  identifier/string-literal/executable line touched. Zero behavior change.
- **Files: "Core only."** In scope: `CLAUDE.md`, `README.md`, `AGENTS.md`,
  `vfe3/**/*.py`, `tests/**/*.py`, and top-level `train_vfe3.py` + `ablation.py`.
  Out of scope (left untouched): `Manuscripts-Theory/*.tex` (canonical copies live
  in the Research vault), `references.bib` (no VFE_2.0 in it), and
  `docs/audits` + `docs/edits` + `docs/reviews` (dated historical record).

## Rewrite style guide (the contract every edit follows)

1. Remove the `VFE_2.0` name and any "parity / mirrors / matches / ported from /
   continuation of VFE_2.0" framing.
2. **Preserve** every independent technical fact in the same comment (apply
   order, denominator definition, no-bias, no-decay, fp32, byte-identical-to-pure-F,
   etc.). Restate as V3's own design intent — it *is* V3's design now.
3. Where "VFE_2.0 parity" was the *whole* content of a comment, replace with the
   plain statement of what the code does, not a deletion that orphans rationale.
4. The framing files shift from "pinned to VFE_2.0 / continuation of VFE_2.0" to
   "self-standing, pinned by its own golden regression tests." The CLAUDE.md
   Testing line "Golden equivalence vs a pinned VFE_2.0 checkout for every ported
   kernel" becomes "Golden regression tests pin every kernel to its reference
   values."
5. **`VFE_3.0` is the current project name and stays.** `head_mixer.py:1` carries
   both `VFE_3.0` (keep) and `VFE_2.0` (remove) on one line — edit surgically.

## Inventory (51 sites)

The `VFE_2`/`VFE2` pattern found 45 sites. A follow-up sweep for the bare `V2`
predecessor shorthand found 6 more (the pattern missed them), and correctly
EXCLUDED 5 `V2` hits that are internal audit-finding labels ("Audit finding V2",
"audit V2", "V6 ... V2 close_basis"), not the predecessor transformer. Total
scrubbed: 51.

- `CLAUDE.md` (4): the pinned-to-VFE_2.0 mandate, the continuation framing, the
  use_prior_bank ablation label, the Golden-equivalence Testing line.
- `README.md` (3): the pinned-reference framing, the ablation label, the
  golden-equivalence Conventions line.
- `AGENTS.md` (2): the pinned-to-VFE_2.0 line, the Golden-equivalence Testing line.
- `vfe3/` (25): config.py(5 incl. one `V2`), prior_bank.py(4), train.py(4),
  model.py(2), e_step.py(2), metrics.py, numerics.py(1 `V2`), run_artifacts.py,
  datasets.py, block.py, transport.py, head_mixer.py
- `tests/` (14): test_config.py(2), test_use_prior_bank.py(2),
  test_phi_weight_decay.py(2 incl. one `V2`), test_mu_trust_region.py(2 incl. one
  `V2`), test_head_mixer_per_block.py(1 `V2`), test_decode_bias.py, test_report.py,
  test_head_mixer.py, test_lambda_beta.py
- top-level (3): train_vfe3.py(2), ablation.py(1)

KEPT (not VFE_2.0): audit-finding labels `V2`/`V6` in test_fix_gauge_audit.py,
test_fix_model_audit.py, and model.py (`audit V2`). `VFE_3.0`/`V3` (the current
project name) kept everywhere.

## Verification

1. `git grep -nE 'VFE_2|VFE 2|VFE2'` over in-scope files returns **zero** hits
   (`VFE_3.0` allowed).
2. Full test suite green at the **same count** as baseline (no executable line
   changed, so any delta is a bug in an edit).
3. `git diff` touches only comment / docstring / Markdown lines — no code lines.
4. Independent adversarial verification pass: confirm no technical fact was lost
   and no non-comment line was altered.

## Out of scope / non-goals

- Removing or renaming any config toggle (`use_prior_bank`, `e_mu_q_trust`,
  `lambda_beta`, head mixer, etc.) — these stay; the pure-path rule is preserved.
- Editing manuscripts, the bibliography, or historical docs.
- Any change that alters a forward-pass value or a test's pinned value.
