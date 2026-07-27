# Base investigator 1 — code-reviewer (quality + security)

Returned 2026-07-27 ~08:16 CDT. Verbatim findings; NOT yet verified.

## Negative results (useful)

No critical findings. Checkpoint/artifact IO is genuinely well hardened: every `torch.load` in
`vfe3/` passes `weights_only=True` (the one `weights_only=False` fallback at
`run_artifacts.py:1844` is gated behind an explicit `trust_resume_checkpoint` toggle and raises
otherwise); no `eval`/`exec`/`yaml.load`/`pickle`; no `shell=True`; no bare `except:` anywhere in
the package; no CLI arg parsing; no UK spellings; no undocumented ninth NN (`nn.Linear`/MLP/
activation grep empty); no credentials in first-party code. Path handling (`path_utils.py`,
`datasets.py:cache_path`, `RunArtifacts.save_json`) validates single-component names, rejects
reparse points, and re-checks containment after `mkdir`.

---

### Böhning emission surrogate never detaches the expansion-point probabilities, contradicting its own MM contract and tripling peak memory
**Location:** vfe3/emission.py:110-140
**Severity:** high
**Evidence:** Only `mu_p` is detached; `weight` (and `bias`) stay live through every logit tile, so `p_0` is not frozen:
```python
    expansion = mu_p.detach()                                      # MM: coefficients frozen at z_0
...
        logit_tile = expansion @ tile.transpose(-1, -2)            # (..., N, C)
        if bias is not None:
            logit_tile = logit_tile + bias[start:start + vocab_chunk]
...
        prob_tile = (logit_tile - log_norm.unsqueeze(-1)).exp()    # (..., N, C) detached by construction
        g = g - prob_tile @ tile                                   # -sum_v p_v W_v, W live
```
Measured under `C:/anaconda/python.exe` (torch CUDA build). Gradient of `g` w.r.t. `W` versus the
frozen-`p_0` reference `-(softmax(mu_p.detach() @ W.detach().T) @ W) + W[ids]`:
`max |grad_live - grad_frozen| = 1.786`, relative `1.503` — a 150% deviation, not a rounding
artifact. Because all three vocab loops are retained in the autograd graph, peak CUDA memory for
one call at `V=50257, K=64, B=8, N=128` is `669.6 MiB` with autograd vs `148.9 MiB` under
`no_grad` and `413.3 MiB` for the naive one-shot dense `(B,N,V)` softmax — i.e. the streaming loop
costs 1.6x MORE than the dense materialization it exists to avoid. No test in
`tests/test_emission_factor_20260726.py` pins detachment (grep for `detach`/`requires_grad` in that
file returns nothing). Reachable via `emission_mode='shared'|'separate'` with `emission_weight>0`.
**Fix:** Compute `running_max`, `exp_sum`, `log_norm` and `prob_tile` inside `torch.no_grad()` (or
detach `tile`/`bias` in those three loops) so only the final `g = g - prob_tile @ tile` and
`weight[token_ids]` carry `W`'s gradient, matching the stated MM convention and restoring the
streaming memory bound.

> ORCHESTRATOR NOTE: this is code added yesterday (commit 2b7a96d). The design intent was
> "expansion point probabilities DETACHED (MM convention); W stays live so the readout table
> receives an E-step gradient." The finding is that the *probabilities* retain a W-path, so the
> stated contract and the code disagree. Whether the live path is a defect or a deliberate
> extra gradient route is exactly the question for the challenge tier. ESCALATE.

### Three top-level drivers spawn children with raw `subprocess.run`, no timeout and no process-tree containment
**Location:** scaling_analysis.py:47, multiseed_analysis.py:48, compare_vocab_figures.py:36
**Severity:** medium
**Evidence:** All three are byte-for-byte the same launch, and none imports `vfe3.process_utils`
(`grep -c process_utils` returns `0` for each):
```python
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(script.parent),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
```
The repo already owns the hardened helper: `vfe3/process_utils.py:176 run_process_tree`, which
assigns Windows descendants to a Job Object with `_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and reaps on
timeout, and which `ablation.py:5525` and `make_figures.py:188` do use with an explicit `timeout=`.
With `capture_output=True` and no timeout, a child that hangs inside matplotlib/numba/UMAP blocks
the parent forever on the pipe, and its grandchildren survive the parent.
**Fix:** Replace the three `subprocess.run` calls with `run_process_tree(..., timeout=<bounded
seconds>)` and handle `subprocess.TimeoutExpired`, matching `make_figures.py:188`.

### Sample-text decoder dispatches on hardcoded `vocab_size` ranges, bypassing the dataset-keyed tokenizer lookup
**Location:** vfe3/train.py:494-499
**Severity:** low
**Evidence:**
```python
    if 40_000 <= cfg.vocab_size <= 60_000:
        enc = tiktoken.get_encoding("gpt2")
    elif 90_000 <= cfg.vocab_size <= 110_000:
        enc = tiktoken.get_encoding("cl100k_base")
    else:
        return None
```
`vfe3/data/datasets.py:47 tiktoken_encoding_name(dataset)` and `datasets.py:429
get_tiktoken_decoder(dataset)` already resolve the encoding from the dataset name through
`_TIKTOKEN_ENCODING_NAME[_tokenizer_tag(dataset)]` — the same mapping used to build the cache
filename (`datasets.py:84`) and validate provenance (`datasets.py:190`). `cfg.vocab_size` is an
independent user-set field; padding a GPT-2 corpus's vocab to a GPU-friendly multiple above 90,000
silently selects `cl100k_base` and prints wrong sample text with no warning. Called at
`vfe3/train.py:1642`.
**Fix:** Thread the dataset name into `train()` and call
`vfe3.data.datasets.get_tiktoken_decoder(dataset)` instead of re-deriving from `cfg.vocab_size`.

### 43 functions violate the mandatory keyword-only argument ordering, including core inference and IO seams
**Location:** vfe3/emission.py:93-95 (plus vfe3/inference/e_step.py:818, vfe3/model/block.py:73, vfe3/model/stack.py:31, vfe3/train.py:503, vfe3/run_artifacts.py:1764, and 37 more)
**Severity:** low
**Evidence:** CLAUDE.md mandates "defined floats, defined ints, defined bools, then Optional". An
AST pass over `vfe3/**/*.py` found 43 functions where an `Optional`-annotated keyword-only
parameter precedes a defaulted scalar. `cat -A` of the newest instance also shows the `=` columns
are not aligned (column 41 vs 23):
```
    bias:        Optional[torch.Tensor] = None,   # (V,) additive logit bias, or None
    vocab_chunk: int   = VOCAB_CHUNK,             # vocabulary tile size
    eps:         float = 1e-12,
```
Representative core-path cases: `e_step_iteration()` — Optional `e_mu_q_trust` precedes defined
scalar `mu_trust_mode`; `train_step()` — Optional `grad_clip` precedes `grad_accum_steps`;
`load_checkpoint()` — Optional `map_location` precedes `restore_rng`; `vfe_block()` / `vfe_stack()`
— Optional `log_prior` precedes `e_step_gradient`.
**Fix:** Reorder keyword-only parameters so defined scalars precede `Optional` ones and re-align
the `:`/`=`/`#` columns, starting with `vfe3/emission.py`.

### `except Exception: pass` silently drops the silhouette diagnostic with no log line
**Location:** vfe3/viz/figures.py:2604-2612
**Severity:** low
**Evidence:**
```python
    if decode is not None and english_linguistic_diagnostics:
        try:
            cats, _ = _token_category_labels(token_ids, decode, "function_content")
            sil = clustering_metrics(X, cats, sample_size=sil_sample)["silhouette"]
            fig.text(0.02, 0.038, f"function/content category silhouette {sil:+.2f} in native space "
...
        except Exception:
            pass
```
Every other best-effort handler logs (`run_artifacts.py:2901`, `run_artifacts.py:2919`,
`train.py:1089`). This one yields a figure silently missing its annotation, so a broken
`_token_category_labels` or `clustering_metrics` is indistinguishable from the feature being off.
**Fix:** Catch the specific expected failure and emit a `logger.warning` naming the exception.

### `UMAPWorker.close()` can block indefinitely on an unbounded `proc.wait()` during cleanup
**Location:** vfe3/viz/figures.py:368-372
**Severity:** low
**Evidence:**
```python
            try:
                proc.wait(timeout=5.0)
            except Exception:
                proc.kill()
                proc.wait()
```
The recovery path drops the timeout, so a worker that does not die on `kill()` hangs `close()` —
which runs from `__exit__` (figures.py:391) and therefore from the figure worker's finalize path.
The module's own containment helper bounds every reap (`process_utils.py:139`, `:165`). The
handler is also broader than needed: `proc.wait(timeout=...)` raises only
`subprocess.TimeoutExpired`.
**Fix:** Narrow to `except subprocess.TimeoutExpired` and bound the post-kill reap.
