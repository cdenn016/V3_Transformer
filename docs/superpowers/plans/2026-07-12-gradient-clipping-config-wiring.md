# Gradient Clipping Config Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing gradient-clipping threshold a validated, persisted VFE3Config field and propagate it through every click-to-run training entry point while preserving the current 1.0 default and global/per-role clipping semantics.

**Architecture:** VFE3Config becomes the single source for the threshold. The train and train_step APIs retain their current default for direct callers but accept Optional[float], with None and 0.0 both disabling clipping. Each click-to-run driver passes cfg.grad_clip explicitly. Existing asdict-based run artifacts persist the value without a second serialization path.

**Tech Stack:** Python 3, PyTorch AdamW and clip_grad_norm_, dataclasses, AST-based script-wiring tests, pytest, and JUnit XML.

## Global Constraints

- Preserve the default numerical path: grad_clip=1.0 must invoke the same clip_grad_norm_ call at the same point as the current literal default.
- Preserve grad_clip_per_role=False as one global clip and True as one clip for each nonempty optimizer role. Do not change optimizer groups, learning rates, unscale timing, finite-gradient gates, or step ordering.
- Define None and 0.0 as explicit clipping-off values. The optimizer and scheduler still step when gradients are finite.
- Reject booleans, strings, negative values, NaN, and positive or negative infinity during VFE3Config construction. Accept finite positive real values and zero.
- Keep direct train and train_step callers source-compatible by retaining a default of 1.0.
- Add no CLI parser or alternate environment-variable override. The three production drivers and the CUDA check script remain click-to-run entry points.
- New model tests run on CPU with embed_dim below 6.
- Run pytest without an additional -q and obtain counts from JUnit XML.
- Append implementation notes to docs/2026-07-12-edits.md; update the same daily document if it already exists.

---

### Task 1: Add and validate VFE3Config.grad_clip

**Files:**

- Modify: vfe3/config.py
- Modify: tests/test_config.py

**Interface:**

Add next to grad_clip_per_role:

    grad_clip:          Optional[float] = 1.0
    grad_clip_per_role: bool            = False

- [ ] Add these parameterized config tests:

    @pytest.mark.parametrize("value", [None, 0.0, 0, 0.25, 1.0])
    def test_grad_clip_accepts_none_or_finite_nonnegative_real(value):
        assert VFE3Config(grad_clip=value).grad_clip == value

    @pytest.mark.parametrize(
        "value", [True, False, "1.0", -0.1, float("nan"), float("inf"), float("-inf")]
    )
    def test_grad_clip_rejects_invalid_domain(value):
        with pytest.raises(ValueError, match=r"grad_clip.*finite real value >= 0"):
            VFE3Config(grad_clip=value)

- [ ] Run the config tests and confirm that VFE3Config currently rejects the unknown grad_clip keyword.

    python -m pytest tests/test_config.py --junitxml=C:\tmp\vfe3-grad-clip-config-red.xml

Expected result: the new tests fail because VFE3Config has no grad_clip field.

- [ ] Add the Optional field and validate it in __post_init__ without coercing None:

    if self.grad_clip is not None:
        if (
            isinstance(self.grad_clip, bool)
            or not isinstance(self.grad_clip, (int, float))
            or not math.isfinite(float(self.grad_clip))
            or float(self.grad_clip) < 0.0
        ):
            raise ValueError(
                "grad_clip must be None or a finite real value >= 0; "
                f"got {self.grad_clip!r}"
            )

- [ ] Re-run the focused tests.

    python -m pytest tests/test_config.py --junitxml=C:\tmp\vfe3-grad-clip-config-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add vfe3/config.py tests/test_config.py
    git commit -m "feat: validate gradient clipping threshold"

### Task 2: Pin None, zero, global, and per-role runtime behavior

**Files:**

- Modify: vfe3/train.py
- Modify: tests/test_train.py

**Interfaces:**

    def train_step(
        model:     VFEModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        tokens:    torch.Tensor,
        targets:   torch.Tensor,

        *,
        grad_accum_steps: int                                = 1,
        grad_clip:        Optional[float]                    = 1.0,
        scaler:           Optional["torch.amp.GradScaler"]   = None,
        metrics_out:      Optional[dict]                     = None,
        status_out:       Optional[dict]                     = None,
    ) -> float:

    def train(
        model:  VFEModel,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        cfg:    VFE3Config,

        *,
        tokens_per_char: float = 1.0,

        n_steps:           int = 100,
        sample_new_tokens: int = 40,
        sample_prompt_len: int = 6,

        generate_samples: bool = True,

        grad_clip:        Optional[float]                           = 1.0,
        log_interval:     Optional[int]                             = None,
        eval_interval:    Optional[int]                             = None,
        val_loader:       Optional[Iterable]                        = None,
        device:           Optional[torch.device]                    = None,
        logger:           Optional[logging.Logger]                  = None,
        artifacts:        Optional["RunArtifacts"]                  = None,
        resume_from:      "Optional[str | Path]"                    = None,
        sample_decode:     Optional[Callable[[Sequence[int]], str]] = None,
        terminal_callback: Optional[Callable[["TrainingTerminalState", List[float]], None]] = None,
    ) -> List[float]:

The interface is cumulative with the earlier artifact-integrity plan: retain every extant Optional keyword even when the clipping patch does not consume it.

- [ ] Add one exact CPU fixture and clip spy:

    def _clip_case(*, per_role=False):
        cfg = VFE3Config(
            vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4,
            grad_clip_per_role=per_role,
        )
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_lambda(step, cfg)
        )
        tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        return model, optimizer, scheduler, tokens, targets

    def _install_clip_spy(monkeypatch):
        calls = []

        def spy(parameters, max_norm, *args, **kwargs):
            params = list(parameters)
            calls.append(({id(p) for p in params}, float(max_norm)))
            return torch.tensor(0.0)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
        return calls

- [ ] Implement the global and per-role assertions without prose-only placeholders:

    def test_grad_clip_global_calls_once_with_all_parameters(monkeypatch):
        model, optimizer, scheduler, tokens, targets = _clip_case(per_role=False)
        calls = _install_clip_spy(monkeypatch)
        train_step(model, optimizer, scheduler, tokens, targets, grad_clip=0.25)
        assert calls == [({id(p) for p in model.parameters()}, 0.25)]

    def test_grad_clip_per_role_calls_once_per_nonempty_role(monkeypatch):
        model, optimizer, scheduler, tokens, targets = _clip_case(per_role=True)
        expected = {}
        for group in optimizer.param_groups:
            expected.setdefault(group.get("role", "other"), set()).update(
                id(p) for p in group["params"]
            )
        calls = _install_clip_spy(monkeypatch)
        train_step(model, optimizer, scheduler, tokens, targets, grad_clip=0.25)
        assert {frozenset(ids) for ids, _ in calls} == {
            frozenset(ids) for ids in expected.values() if ids
        }
        assert all(max_norm == 0.25 for _, max_norm in calls)
        flattened = [pid for ids, _ in calls for pid in ids]
        assert len(flattened) == len(set(flattened))

- [ ] Parameterize `None` and `0.0`; patch both clipping and `optimizer.step`, then assert clipping is absent while optimizer and scheduler each advance once:

    @pytest.mark.parametrize("grad_clip", [None, 0.0])
    def test_grad_clip_off_still_steps(monkeypatch, grad_clip):
        model, optimizer, scheduler, tokens, targets = _clip_case()
        calls = _install_clip_spy(monkeypatch)
        status = {}
        before_epoch = scheduler.last_epoch
        train_step(
            model, optimizer, scheduler, tokens, targets,
            grad_clip=grad_clip, status_out=status,
        )
        assert calls == []
        assert status["did_step"] is True
        assert scheduler.last_epoch == before_epoch + 1

- [ ] Add the omitted-versus-explicit regression:

    def test_omitted_grad_clip_matches_explicit_one(monkeypatch):
        left, left_opt, left_sched, tokens, targets = _clip_case()
        right, right_opt, right_sched, _, _ = _clip_case()
        right.load_state_dict(left.state_dict())
        right_opt.load_state_dict(left_opt.state_dict())
        right_sched.load_state_dict(left_sched.state_dict())
        original = torch.nn.utils.clip_grad_norm_
        calls = []

        def tracked(parameters, max_norm, *args, **kwargs):
            params = list(parameters)
            calls.append((len(params), float(max_norm)))
            return original(params, max_norm, *args, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", tracked)
        train_step(left, left_opt, left_sched, tokens, targets)
        split = len(calls)
        train_step(right, right_opt, right_sched, tokens, targets, grad_clip=1.0)
        assert calls[:split] == calls[split:]
        for key, value in left.state_dict().items():
            assert torch.equal(value, right.state_dict()[key])

- [ ] Add `test_grad_clip_signature_preserves_terminal_callback`. Pass a counter callback through `train(..., terminal_callback=callback, grad_clip=cfg.grad_clip)` on a one-step CPU run and require exactly one callback with a `TrainingTerminalState`; this prevents the later clipping signature/wiring edit from erasing the PB-02 terminal-artifact seam.

- [ ] Run the focused tests before changing type hints. The current runtime should already pass the behavioral cases; only the config-to-train integration is absent. Record that distinction in the JUnit result rather than calling the entire feature broken.

    python -m pytest tests/test_train.py --junitxml=C:\tmp\vfe3-grad-clip-runtime-baseline.xml

- [ ] Change the train_step and train annotations to Optional[float]. Keep these existing guards structurally unchanged:

    need_unscale = (
        grad_clip is not None and grad_clip > 0
    ) or metrics_out is not None

    if grad_clip is not None and grad_clip > 0 and not skip_step:
        if getattr(model.cfg, "grad_clip_per_role", False):
            role_params: dict = {}
            for group in optimizer.param_groups:
                role = group.get("role", "other")
                role_params.setdefault(role, []).extend(group["params"])
            for params in role_params.values():
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

- [ ] Update the train_step docstring to state that None and zero disable clipping, positive values clip once after accumulation and unscale, and per-role mode applies the same threshold independently to each role.

- [ ] Re-run the focused tests.

    python -m pytest tests/test_train.py --junitxml=C:\tmp\vfe3-grad-clip-runtime-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add vfe3/train.py tests/test_train.py
    git commit -m "test: pin gradient clipping modes"

### Task 3: Propagate the config through every click-to-run path

**Files:**

- Modify: train_vfe3.py
- Modify: ablation.py
- Modify: scaling.py
- Modify: check_gpu_tests.py
- Modify: vfe3/train.py
- Create: tests/test_gradient_clipping_config_wiring.py

**Required call shape in every driver:**

    losses = train(
        model,
        train_loader,
        cfg,
        n_steps=cfg.max_steps,
        grad_clip=cfg.grad_clip,
    )

- [ ] Write an AST-based test that parses train_vfe3.py, ablation.py, scaling.py, and check_gpu_tests.py without importing them. For each file, find calls whose callee is train and assert that the grad_clip keyword is the attribute expression cfg.grad_clip.

    def _train_calls(path: Path) -> list[ast.Call]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "train"
        ]

- [ ] In the same test, collect keyword names from the self-contained config dictionaries in train_vfe3.py, ablation.py, and scaling.py and assert that each exposes grad_clip. Also assert scaling.py exposes grad_clip_per_role because its baseline currently relies on the dataclass default instead of showing the click-to-run toggle. `check_gpu_tests.py` obtains `cfg` from `_structured_cfg()` and therefore has no duplicate config field to add; its call still must pass `cfg.grad_clip`.

- [ ] Add a source test for run_training in vfe3/train.py asserting its train call passes grad_clip=cfg.grad_clip.

- [ ] Run the wiring test and confirm all five call sites fail the new assertion.

    python -m pytest tests/test_gradient_clipping_config_wiring.py --junitxml=C:\tmp\vfe3-grad-clip-wiring-red.xml

Expected result: JUnit reports assertion failures naming train_vfe3.py, ablation.py, scaling.py, check_gpu_tests.py, and run_training.

- [ ] Add grad_clip=1.0 adjacent to grad_clip_per_role in train_vfe3.py and ablation.py. Add both grad_clip=1.0 and grad_clip_per_role=False to the training-mechanics section of scaling.py. Do not alter any existing local values.

- [ ] Pass grad_clip=cfg.grad_clip from all four scripts and from run_training.

- [ ] Re-run the wiring test.

    python -m pytest tests/test_gradient_clipping_config_wiring.py --junitxml=C:\tmp\vfe3-grad-clip-wiring-green.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Commit the task.

    git add train_vfe3.py ablation.py scaling.py check_gpu_tests.py vfe3/train.py tests/test_gradient_clipping_config_wiring.py
    git commit -m "feat: wire gradient clipping config"

### Task 4: Prove run-artifact and resume persistence

**Files:**

- Modify: tests/test_run_artifacts.py
- Modify: tests/test_checkpoint_resume.py

- [ ] Add a parameterized config.json test for grad_clip=None, grad_clip=0.0, and grad_clip=0.25:

    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert meta["config"]["grad_clip"] == expected

- [ ] Extend the best-model bundle test to assert bundle["config"]["grad_clip"] equals the config value and that semantic_config_fingerprint changes when only grad_clip changes.

- [ ] Extend the resumable checkpoint test to assert checkpoint["config"]["grad_clip"] equals the config value. Add a resume-drift test showing that changing only grad_clip is reported by the existing config-drift check.

- [ ] Add the cross-plan selection-migration regression after the artifact-integrity plan is present. Build a raw legacy config mapping from `asdict(VFE3Config())`, remove only `grad_clip`, and require `_selection_semantic_config(legacy)` to equal the live default projection with `grad_clip=1.0`. Then set an explicit `grad_clip=0.25` and require a different projection. Preserve the raw legacy mapping's own full fingerprint check before projection, and add an unknown-key case that fails closed rather than being ignored by `config_from_serialized`.

- [ ] Run the focused artifact tests.

    python -m pytest tests/test_run_artifacts.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-grad-clip-artifacts-green.xml

Expected result: JUnit failures=0 and errors=0. No production serialization change should be necessary because RunArtifacts already writes asdict(cfg).

- [ ] Commit the task.

    git add tests/test_run_artifacts.py tests/test_checkpoint_resume.py
    git commit -m "test: persist gradient clipping config"

### Task 5: Document and verify the completed wiring

**Files:**

- Modify: docs/2026-07-12-edits.md

- [ ] Record the config field, accepted domain, None/zero semantics, global/per-role behavior, four propagation call sites, and artifact persistence.

- [ ] Compile the touched modules and scripts.

    python -m compileall vfe3 train_vfe3.py ablation.py scaling.py check_gpu_tests.py

Expected result: exit code 0.

- [ ] Run the complete focused suite.

    python -m pytest tests/test_config.py tests/test_train.py tests/test_gradient_clipping_config_wiring.py tests/test_run_artifacts.py tests/test_checkpoint_resume.py --junitxml=C:\tmp\vfe3-grad-clip-final.xml

Expected result: JUnit failures=0 and errors=0.

- [ ] Run the full suite once.

    python -m pytest --junitxml=C:\tmp\vfe3-grad-clip-full.xml

Expected result: exit code 0 and JUnit failures=0 and errors=0. Report the tests count only from the JUnit tests attribute.

- [ ] Inspect the diff and status. Confirm that the existing default remains 1.0 at the config, train, and train_step layers and that no config value unrelated to clipping changed.

    git diff origin/main...HEAD -- vfe3/config.py vfe3/train.py train_vfe3.py ablation.py scaling.py check_gpu_tests.py tests
    git status --short

- [ ] Commit the dated document, then follow the repository lifecycle: push the task branch, merge into main after verification, push main, safely fast-forward the live checkout only if user WIP is untouched, remove the temporary worktree, and show final git status.
