# VFE 4.0 Modular Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new sibling repository at `C:\Users\chris and christine\Desktop\VFE_4.0` whose normalized causal latent-variable model, structured recognition laws, information-geometric distribution families, gauge geometry, ordinary ELBO, and scalable numerical backends remain type-separated and continuously checked against bounded dense float64 references. Objective identity, objective evaluation, linear algebra, arithmetic, variational-family relation, and optimizer guarantees remain separate claims.

**Architecture:** A typed model graph owns probability semantics, reference measures, factor dependencies, parameter ownership, and the complete objective inventory. Frozen built-in catalogs bind typed kernels during composition, while inference backends own numerical realization only; the dense float64 implementation remains reachable as the semantic oracle after every scalable backend is added. Training recognition, prior-predictive evaluation, and generation use separate typed entry points so an observation-conditioned posterior cannot be substituted for a target-blind predictive law.

**Design authority:** [2026-07-17-vfe4-modular-architecture-design.md](../specs/2026-07-17-vfe4-modular-architecture-design.md), the local VFE 4.0 Gauge-Causal ELBO white paper, and [2026-07-17-global-elbo-language-model-observations-investigation.md](../../2026-07-17-global-elbo-language-model-observations-investigation.md).

**Tech Stack:** A CPython version supported by the selected PyTorch build, a PyTorch CUDA build validated on the RTX 5090, NumPy and SciPy for CPU references, safetensors for pickle-free tensor artifacts, pytest with JUnit XML for every gate, Ruff and mypy for static verification, and standard-library dataclasses, enums, protocols, JSON, and hashing for immutable contracts and manifests. Task 0.1 records the actually tested CPU and CUDA versions and then freezes the environment; this plan does not predeclare a CUDA toolkit or library minimum that has not been verified on the target machine.

## Execution Preflight and Path Interpretation

Every absolute path below names the final destination in the sibling repository. Workers execute the relative commands inside a dedicated temporary VFE 4.0 task worktree, never inside the V3 checkout and never in a VFE 4.0 checkout containing user work. Before Task 0.1, the coordinator ensures that the new remote repository has an `origin/main` bootstrap commit with no implementation files. If the remote or bootstrap commit has not been provisioned, stop and record that prerequisite rather than substituting a local-only repository.

- [ ] Fetch the remote and inspect the authoritative base.

```powershell
git -C "C:\Users\chris and christine\Desktop\VFE_4.0" fetch origin
git -C "C:\Users\chris and christine\Desktop\VFE_4.0" log -5 --oneline origin/main
git -C "C:\Users\chris and christine\Desktop\VFE_4.0" status --short
```

- [ ] Create the isolated implementation worktree and task branch from `origin/main`.

```powershell
git -C "C:\Users\chris and christine\Desktop\VFE_4.0" worktree add "C:\tmp\VFE4_modular_build_20260718" -b codex/vfe4-modular-build origin/main
git -C "C:\tmp\VFE4_modular_build_20260718" status --short
```

All later relative paths and commands run with `C:\tmp\VFE4_modular_build_20260718` as the working directory. A worker does not change, stash, reset, clean, restore, or overwrite either the V3 checkout or pre-existing VFE 4.0 files.

## Global Constraints

- All implementation work occurs in the new repository `C:\Users\chris and christine\Desktop\VFE_4.0`; this V3 worktree contains only the design and implementation-plan artifacts.
- The installed package is `vfe4` under `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4`.
- V4 has no runtime import, package dependency, path lookup, or configuration fallback to `vfe3`; V3 is migration evidence and an optional one-way checkpoint input only.
- The normalized generative joint, recognition law, base/reference measures, supports, and complete factor inventory are declared before any ELBO evaluator or update exists.
- The dense float64 CPU path remains callable and tested after every later phase. No scalable backend may replace or weaken it.
- Float32 CUDA is the production path. Autocast, TF32, float16, and bfloat16 are disabled inside SPD assembly, factorization, solves, log determinants, matrix exponentials, and oracle comparisons.
- No implicit approximation, silent jitter, pseudo-inverse fallback, moment projection, source truncation, sparse fill drop, or invalid-value saturation is permitted. A changed candidate or approximation produces a typed ledger event and a fresh complete-objective evaluation.
- Exactness is not a boolean. Every run records objective contract, family relation, evaluation status, and optimization status separately.
- `exact_coordinate_argmax`, `conjugate_closed_form`, `accepted_mm`, `accepted_generalized_em`, `natural_gradient_step`, and `unverified_finite_step` remain distinct optimization labels.
- Natural coordinates do not by themselves make an optimizer a natural-gradient method. A natural-gradient label requires a registered inverse-Fisher action or a proved dual-coordinate implementation algebraically equivalent to that action on the declared chart domain.
- Categorical source KL terms in the ordinary ELBO have coefficient one. Temperatures parameterize normalized source priors and do not independently reweight source entropy or KL.
- The initial core conditions on deterministic geometry. Latent geometry, noncompact-group priors, gauge fixing, and quotient measures are outside the initial build.
- No neural recognition network is implemented initially. Recognition uses explicit per-example structured variational state and registered iterative or coordinate backends.
- No dynamic third-party plugins, Python entry points, runtime module discovery, or configuration-selected imports. Built-ins are registered explicitly in one composition root and catalogs freeze before graph compilation.
- No call site branches on family, group, divergence, graph profile, device backend, or approximation mode. The composition root resolves a typed implementation once and passes the resulting object through protocols.
- Every registered capability fails closed when unsupported. An all-invalid categorical support, out-of-domain chart, non-SPD precision, missing closure certificate, or unavailable backend operation raises a typed error.
- Every phase is test-first. Red and green commands produce machine-readable JUnit XML; pass counts and failure claims come from that XML, never memory or console inference.
- Tests are deterministic unless their stochastic error budget, seed, sample count, and interval rule are part of the fixture and approximation ledger.
- The code uses American English spelling, focused files, complete type hints, immutable specifications, and no speculative abstraction beyond the phase that first consumes it.

## Exactness and Gate Vocabulary

The following independent records are created in Phase 0 and used unchanged by all later phases:

```python
class ObjectiveContract(Enum):
    ORDINARY_CONDITIONAL_ELBO = "ordinary_conditional_elbo"

class FamilyRelation(Enum):
    EXACT_DECLARED_FAMILY = "exact_declared_family"
    RESTRICTED_VARIATIONAL_FAMILY = "restricted_variational_family"
    PROJECTED_FAMILY = "projected_family"

class EvaluationStatus(Enum):
    ANALYTIC_CLOSED_FORM = "analytic_closed_form"
    EXACT_FINITE_ENUMERATION = "exact_finite_enumeration"
    DETERMINISTIC_QUADRATURE = "deterministic_quadrature"
    STOCHASTIC_UNBIASED = "stochastic_unbiased"
    STOCHASTIC_BIASED = "stochastic_biased"
    FAILED = "failed"

class LinearAlgebraStatus(Enum):
    DIRECT_FACTORIZATION = "direct_factorization"
    STRUCTURE_EQUIVALENT_DIRECT = "structure_equivalent_direct"
    ITERATIVE_CERTIFIED = "iterative_certified"
    STOCHASTIC_ESTIMATE = "stochastic_estimate"
    REGULARIZED_CANDIDATE = "regularized_candidate"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class ArithmeticReport:
    dtype: str
    device: str
    autocast_enabled: bool
    tf32_enabled: bool
    residual: float | None
    tolerance: float | None

class OptimizationStatus(Enum):
    EXACT_COORDINATE_ARGMAX = "exact_coordinate_argmax"
    CONJUGATE_CLOSED_FORM = "conjugate_closed_form"
    ACCEPTED_MM = "accepted_mm"
    ACCEPTED_GENERALIZED_EM = "accepted_generalized_em"
    NATURAL_GRADIENT_STEP = "natural_gradient_step"
    UNVERIFIED_FINITE_STEP = "unverified_finite_step"
```

Each phase gate checks these properties where applicable:

1. Normalization of every probability law and kernel on its declared support.
2. Identity of monolithic and decomposed objective ledgers.
3. Equality of natural, expectation, and moment evaluations of the same Gaussian law.
4. Equality of dense and structured backend values inside the declared linear-algebra and arithmetic budgets.
5. Independent reporting of objective evaluation, linear algebra, arithmetic, variational-family relation, and optimization status.
6. Causal no-leakage and complete gauge covariance of the paths that claim those properties.

## Future Repository Map

The implementation creates and maintains these focused package areas:

```text
C:\Users\chris and christine\Desktop\VFE_4.0\
  pyproject.toml
  README.md
  src\vfe4\
    core\          # spaces, measures, supports, identities, exactness, errors
    catalog\       # frozen built-in registrations and composition manifests
    families\      # categorical, Gaussian, charts, mixtures, divergences
    geometry\      # groups, representations, bundles, fibers, morphisms, transports
    factors\       # normalized initial, transition, source, emission, hierarchy factors
    graph\         # typed nodes, factor graph, profiles, validation, compiler
    objectives\    # complete ELBO, local decomposition, update acceptance, ledger
    inference\     # recognition laws, predictive filtering, projections, backend protocols
    numerics\      # dense, block-banded, sparse, quadrature, SPD, CUDA kernels
    language\      # token spaces, causal contexts, emissions, metrics, data windows
    config\        # authored, canonical, and resolved immutable experiment specs
    runtime\       # explicit composition, training, evaluation, generation, device policy
    artifacts\     # checkpoints, manifests, JUnit summaries, V3 import
    experiments\   # frozen protocols, matched baselines, matched ablations
  tests\
    contracts\
    oracles\
    integration\
    cuda\
    migration\
  tools\v3_import\ # standalone one-way initialization converter
  experiments\     # click-run experiment files with no command-line parser
```

## Phase 0: Contracts and Oracles

### Task 0.1: Scaffold the sibling repository and machine-readable test lane

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\pyproject.toml`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\README.md`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\build-handoff.md`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_package_contract.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\artifacts\junit\.gitkeep`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\artifacts\environment\cpu-bootstrap.json`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\requirements.lock` from the tested resolution

**Interfaces:**

- Consumes: no earlier V4 code.
- Produces: importable package `vfe4`, `vfe4.__version__`, pytest markers `cuda`, `oracle`, `migration`, the machine-readable directory `artifacts/junit`, a tested CPU environment record, a frozen dependency resolution, and the continuously maintained execution handoff.

- [ ] **Step 1: Scaffold the isolated task worktree and write the failing package contract**

```python
from importlib.metadata import version

import vfe4


def test_installed_name_and_public_version_match() -> None:
    assert vfe4.__version__ == version("vfe4")
    assert vfe4.__all__ == ["__version__"]
```

- [ ] **Step 2: Run the red test and preserve its JUnit output**

Run from the isolated task worktree established in the execution preflight:

```powershell
python -m pytest tests\contracts\test_package_contract.py --junitxml=artifacts\junit\phase0-package-red.xml
```

Expected: nonzero exit code because `vfe4` does not exist; retain any emitted XML as red-phase evidence.

- [ ] **Step 3: Add the minimal package and tooling configuration**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "vfe4"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["torch", "numpy", "scipy", "safetensors"]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-config --strict-markers"
markers = [
    "oracle: deterministic float64 or higher precision identity test",
    "cuda: requires CUDA and a supported GPU",
    "migration: consumes an external V3 checkpoint fixture",
]
```

These are bootstrap names, not the reproducibility record. Before the first green commit, resolve them in the CPU environment, run the package smoke test, record Python, PyTorch, NumPy, SciPy, safetensors, pytest, Ruff, and mypy versions in `artifacts/environment/cpu-bootstrap.json`, and freeze the tested resolution with the repository's selected lock mechanism. The first CUDA gate records its separate PyTorch build, CUDA runtime, driver, and RTX 5090 identity. Do not infer GPU support from a version string alone.

```python
from importlib.metadata import version

__version__ = version("vfe4")
__all__ = ["__version__"]
```

- [ ] **Step 4: Install editable development dependencies and run the green test**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests\contracts\test_package_contract.py --junitxml=artifacts\junit\phase0-package.xml
```

Expected: both commands exit 0; `phase0-package.xml` has zero failures and zero errors.

- [ ] **Step 5: Commit the independently runnable scaffold**

```powershell
git add pyproject.toml requirements.lock README.md docs\build-handoff.md src\vfe4\__init__.py tests\contracts\test_package_contract.py artifacts\junit\.gitkeep artifacts\environment
git commit -m "chore: scaffold the vfe4 package and test lane"
```

### Task 0.2: Define spaces, measures, supports, exactness, and the approximation ledger

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\ids.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\spaces.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\measures.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\supports.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\exactness.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\core\errors.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\ledger.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_core_contracts.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_approximation_ledger.py`

**Interfaces:**

- Consumes: `torch.Tensor` and standard-library immutable records.
- Produces: `SpaceId`, `NodeId`, `FactorId`, `FiniteSpace`, `VectorSpace`, `ProductSpace`, `CountingMeasure`, `LebesgueMeasure`, `ProductMeasure`, `FiniteSupport`, `ObjectiveContract`, `FamilyRelation`, `EvaluationStatus`, `LinearAlgebraStatus`, `ArithmeticReport`, `OptimizationStatus`, `ExecutionEvent`, and `ApproximationLedger`.

- [ ] **Step 1: Write failing equality, support, and identity-ledger tests**

```python
from vfe4.core.exactness import ArithmeticReport, LinearAlgebraStatus
from vfe4.core.spaces import FiniteSpace, ProductSpace, VectorSpace
from vfe4.core.supports import FiniteSupport
from vfe4.objectives.ledger import ApproximationLedger


def test_spaces_and_supports_are_immutable_and_typed() -> None:
    tokens = FiniteSpace(name="token", cardinality=3)
    latent = VectorSpace(name="z", dimension=2)
    joint = ProductSpace(name="token_z", factors=(tokens, latent))
    support = FiniteSupport(space=tokens, allowed=(0, 2))
    assert len(joint.factors) == 2
    assert joint.continuous_dimension == 2
    assert support.contains(2)
    assert not support.contains(1)


def test_direct_solve_does_not_claim_analytic_objective_evaluation() -> None:
    ledger = ApproximationLedger()
    ledger.record_linear_algebra(
        operation_id="dense_f64.solve",
        status=LinearAlgebraStatus.DIRECT_FACTORIZATION,
        arithmetic=ArithmeticReport.float64_cpu(residual=2.0e-14),
    )
    report = ledger.freeze()
    assert report.events[0].objective_evaluation is None
    assert report.events[0].linear_algebra is LinearAlgebraStatus.DIRECT_FACTORIZATION
    assert report.events[0].arithmetic.dtype == "float64"
    assert report.events[0].changed_candidate is False
```

- [ ] **Step 2: Run the red contract tests**

Run:

```powershell
python -m pytest tests\contracts\test_core_contracts.py tests\contracts\test_approximation_ledger.py --junitxml=artifacts\junit\phase0-core-red.xml
```

Expected: nonzero exit code because `vfe4.core` and `vfe4.objectives.ledger` do not exist.

- [ ] **Step 3: Implement immutable contracts and fail-closed validation**

Use frozen dataclasses. The load-bearing public forms are:

```python
@dataclass(frozen=True, slots=True)
class FiniteSpace:
    name: str
    cardinality: int

    def __post_init__(self) -> None:
        if self.cardinality < 1:
            raise ContractError("finite-space cardinality must be positive")


@dataclass(frozen=True, slots=True)
class VectorSpace:
    name: str
    dimension: int


@dataclass(frozen=True, slots=True)
class ProductSpace:
    name: str
    factors: tuple[FiniteSpace | VectorSpace, ...]

    @property
    def continuous_dimension(self) -> int:
        return sum(
            factor.dimension
            for factor in self.factors
            if isinstance(factor, VectorSpace)
        )


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    operation_id: str
    objective_evaluation: EvaluationStatus | None
    linear_algebra: LinearAlgebraStatus | None
    arithmetic: ArithmeticReport | None
    reason: str
    changed_candidate: bool
    residual: float | None = None
    tolerance: float | None = None
    seed: int | None = None
```

`ApproximationLedger.freeze()` returns an immutable report and rejects additional writes. Objective-integral evaluators, linear-algebra operations, arithmetic realizations, family changes, and optimizer steps populate separate fields. `record_linear_algebra()` never assigns an objective-evaluation status. An approximation-free run still records the operations it performed; direct float64 and float32 factorizations differ in their arithmetic reports even when both implement the same algebraic formula.

- [ ] **Step 4: Run the green contract tests and type checker**

Run:

```powershell
python -m pytest tests\contracts\test_core_contracts.py tests\contracts\test_approximation_ledger.py --junitxml=artifacts\junit\phase0-core.xml
python -m mypy src\vfe4\core src\vfe4\objectives\ledger.py
```

Expected: both commands exit 0; the JUnit XML has zero failures and errors.

- [ ] **Step 5: Commit the semantic foundation**

```powershell
git add src\vfe4\core src\vfe4\objectives\ledger.py tests\contracts\test_core_contracts.py tests\contracts\test_approximation_ledger.py
git commit -m "feat: define probability contracts and exactness ledger"
```

### Task 0.3: Add frozen catalogs and the typed factor graph

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\catalog\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\catalog\registry.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\catalog\builtins.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\nodes.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\model.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\recognition.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\validation.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_frozen_catalog.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_typed_graph.py`

**Interfaces:**

- Consumes: IDs, spaces, measures, supports, and exactness enums from Task 0.2.
- Produces: generic `CatalogBuilder[K, V]`, immutable `CatalogSnapshot[K, V]`, `ComponentDescriptor`, typed capability records, `CompatibilityPredicateRecord`, `CatalogManifest`, `RandomNode`, `ObservedNode`, `DeterministicNode`, `FactorSpec`, `RecognitionFactorSpec`, `RecognitionDisintegrationSpec`, `ParameterOwner`, `ModelGraph`, `RecognitionGraph`, and validators returning `ValidatedModelGraph` and `ValidatedRecognitionGraph`.

- [ ] **Step 1: Write failing catalog and graph tests**

```python
import pytest

from vfe4.catalog.registry import CatalogBuilder
from vfe4.core.errors import CatalogFrozenError, DuplicateRegistrationError
from vfe4.graph.model import FactorSpec, ModelGraph
from vfe4.graph.recognition import RecognitionGraph
from vfe4.graph.validation import validate_model_graph, validate_recognition_graph


def test_catalog_fails_closed_after_freeze() -> None:
    catalog: CatalogBuilder[str, int] = CatalogBuilder(name="test")
    catalog.register("one", 1)
    with pytest.raises(DuplicateRegistrationError):
        catalog.register("one", 2)
    snapshot = catalog.freeze()
    assert snapshot.resolve("one") == 1
    with pytest.raises(CatalogFrozenError):
        catalog.register("two", 2)


def test_graph_rejects_target_dependency_in_predictive_factor() -> None:
    graph = ModelGraph.for_test_with_target_dependent_transition()
    with pytest.raises(ValueError, match="predictive factor depends on current observation"):
        validate_model_graph(graph)


@pytest.mark.parametrize("defect", ["missing_target", "overlap", "backward_edge", "cycle"])
def test_recognition_disintegration_must_define_one_normalized_global_q(defect: str) -> None:
    recognition = RecognitionGraph.for_test_with_defect(defect)
    with pytest.raises(ValueError):
        validate_recognition_graph(recognition)
```

- [ ] **Step 2: Run the red catalog and graph tests**

Run:

```powershell
python -m pytest tests\contracts\test_frozen_catalog.py tests\contracts\test_typed_graph.py --junitxml=artifacts\junit\phase0-catalog-graph-red.xml
```

Expected: nonzero exit code because the catalog and graph modules do not exist.

- [ ] **Step 3: Implement one-time composition and graph validation**

```python
@dataclass(slots=True)
class CatalogBuilder(Generic[K, V]):
    name: str
    _entries: dict[K, V] = field(default_factory=dict)
    _frozen: bool = False

    def register(self, key: K, value: V) -> None:
        if self._frozen:
            raise CatalogFrozenError(self.name)
        if key in self._entries:
            raise DuplicateRegistrationError(f"{self.name}:{key}")
        self._entries[key] = value

    def freeze(self) -> CatalogSnapshot[K, V]:
        self._frozen = True
        return CatalogSnapshot.from_entries(self.name, self._entries)
```

Only the builder mutates. Duplicate registration fails. An explicit replacement operation requires the expected previous descriptor digest. A component descriptor records stable key, catalog API version, implementation version, immutable local settings schema, typed capabilities, factory, compatibility predicates, and source provenance. Freezing returns an immutable snapshot plus a canonical manifest and digest; each predicate identifier, input digest, result, and diagnostic is recorded during resolution.

`FactorSpec` declares inputs, output, factor kind, reference measure, normalized status, causal visibility, and parameter owners. `validate_model_graph` checks unique IDs, acyclicity, compatible spaces, complete measures, normalized generative factors, nonempty supports, topological causality, and parameter ownership. `RecognitionGraph` is either one normalized joint law over the full latent event or an ordered disintegration. In the disintegrated form, normalized target blocks are pairwise disjoint, cover every latent exactly once, and may condition only on earlier latent blocks and permitted observations. Validation rejects omissions, overlap, backward dependencies, and recognition cycles. Runtime code accepts neither model nor recognition graphs before these separate validators succeed.

- [ ] **Step 4: Run green tests and serialize a deterministic catalog manifest**

Run:

```powershell
python -m pytest tests\contracts\test_frozen_catalog.py tests\contracts\test_typed_graph.py --junitxml=artifacts\junit\phase0-catalog-graph.xml
```

Expected: exit code 0; the XML has zero failures and errors; registering entries in a different order produces the same sorted manifest hash.

- [ ] **Step 5: Commit the static composition boundary**

```powershell
git add src\vfe4\catalog src\vfe4\graph tests\contracts\test_frozen_catalog.py tests\contracts\test_typed_graph.py
git commit -m "feat: add frozen catalogs and typed model graph"
```

### Task 0.4: Implement the bounded dense float64 precision oracle

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\backend.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\spd.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\dense.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_dense_f64_precision.py`

**Interfaces:**

- Consumes: exactness ledger and validated tensor shapes.
- Produces: `PrecisionBackend`, `PrecisionSystem`, `FactorizationHandle`, `OperationReport[T]`, `ConditioningReport`, and `DenseF64Oracle`. The public handle exposes `solve`, `quadratic`, `logdet`, `selected_inverse`, `sample`, and `condition_estimate`; it exposes no full inverse method.

- [ ] **Step 1: Write failing solve, log-determinant, and API-surface tests**

```python
import torch

from vfe4.numerics.dense import DenseF64Oracle


def test_dense_f64_oracle_matches_direct_residual_and_logdet() -> None:
    precision = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    handle = DenseF64Oracle(max_dimension=32).factorize(precision)
    solution = handle.solve(rhs).value
    assert torch.linalg.vector_norm(precision @ solution - rhs) < 1e-13
    assert torch.allclose(handle.logdet().value, torch.linalg.slogdet(precision).logabsdet)
    assert not hasattr(handle, "inverse")
```

- [ ] **Step 2: Run the red dense-oracle test**

Run:

```powershell
python -m pytest tests\oracles\test_dense_f64_precision.py --junitxml=artifacts\junit\phase0-dense-red.xml
```

Expected: nonzero exit code because `DenseF64Oracle` does not exist.

- [ ] **Step 3: Implement strict Cholesky factorization without repair fallbacks**

```python
class DenseF64Oracle:
    def __init__(self, *, max_dimension: int) -> None:
        self.max_dimension = max_dimension

    def factorize(self, precision: torch.Tensor) -> DenseFactorization:
        matrix = precision.to(device="cpu", dtype=torch.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("precision must be one square matrix")
        if not bool(torch.isfinite(matrix).all()):
            raise NonSPDPrecisionError("precision contains nonfinite entries")
        if matrix.shape[-1] > self.max_dimension:
            raise ValueError("dense float64 oracle dimension exceeds its declared bound")
        if not torch.equal(matrix, matrix.transpose(-1, -2)):
            raise NonSPDPrecisionError("precision is not exactly symmetric")
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info.max()) != 0:
            raise NonSPDPrecisionError("precision Cholesky failed")
        return DenseFactorization(precision=matrix, cholesky=chol)
```

Each operation returns its value plus residual, tolerance, dtype, device, and identity ledger event. `selected_inverse` accepts an explicit tuple of block selections and uses solves; no production helper materializes a complete covariance.

- [ ] **Step 4: Run the green oracle test and the complete Phase 0 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles\test_dense_f64_precision.py --junitxml=artifacts\junit\phase0-gate.xml
```

Expected: exit code 0; `phase0-gate.xml` has zero failures and errors. Inspect the XML rather than reporting a remembered count.

- [ ] **Step 5: Commit the exact numerical oracle**

```powershell
git add src\vfe4\inference src\vfe4\numerics tests\oracles\test_dense_f64_precision.py
git commit -m "feat: add the bounded dense float64 precision oracle"
```

### Task 0.5: Establish the strict authored and canonical configuration envelope

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\schema.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\codec.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\migrations.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\resolved.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_config_codec.py`

**Interfaces:**

- Consumes: immutable core identifiers and component keys; later phases extend section-specific settings without changing the envelope.
- Produces: frozen `ExperimentSpec`, section specs, typed `ComponentRef`, `AuthoredSpec`, `CanonicalSpec`, `ResolvedSpec`, and pure sequential schema migrations.

- [ ] **Step 1: Write failing strict-decoding and immutability tests**

```python
def test_unknown_field_and_ambiguous_boolean_fail_closed() -> None:
    with pytest.raises(UnknownConfigFieldError):
        decode_experiment_spec({"schema_version": 1, "model": {}, "mdoel": {}})
    with pytest.raises(ConfigTypeError):
        decode_experiment_spec({"schema_version": 1, "training": {"deterministic": 1}})


def test_canonicalization_does_not_mutate_authored_spec() -> None:
    authored = decode_experiment_spec(minimal_authored_mapping())
    snapshot = canonical_json(authored)
    canonical = canonicalize(authored)
    assert canonical_json(authored) == snapshot
    assert canonical.authored_digest == digest(authored)
```

- [ ] **Step 2: Run the red configuration-envelope tests**

Run:

```powershell
python -m pytest tests\contracts\test_config_codec.py --junitxml=artifacts\junit\phase0-config-red.xml
```

Expected: nonzero exit code because the strict configuration envelope does not exist.

- [ ] **Step 3: Implement frozen schemas, an explicit codec, and pure migrations**

Define the root section structure now and add only settings consumed by completed phases. `ComponentRef` contains a stable key and a locally validated settings object. Reject unknown fields, wrong scalar types, unsupported newer versions, ambiguous booleans, duplicate semantic identifiers, and incompatible references. Preserve authored input exactly, materialize defaults only in the canonical form, and reserve the resolved form for compiler output. Compilation and later migrations never mutate an earlier form.

- [ ] **Step 4: Run the green Phase 0 configuration gate**

Run:

```powershell
python -m pytest tests\contracts\test_config_codec.py tests\contracts\test_frozen_catalog.py tests\contracts\test_typed_graph.py --junitxml=artifacts\junit\phase0-config.xml
```

Expected: exit code 0; strict invalid configurations fail before tensor allocation and canonical digests are deterministic.

- [ ] **Step 5: Commit the configuration envelope**

```powershell
git add src\vfe4\config tests\contracts\test_config_codec.py
git commit -m "feat: add strict immutable experiment configuration"
```

## Phase 1: Categorical and Full-Gaussian Families

### Task 1.1: Implement normalized categorical laws and the exact discrete oracle

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\categorical.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\divergences.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_categorical_family.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_discrete_normalization.py`

**Interfaces:**

- Consumes: `FiniteSpace`, `FiniteSupport`, `CountingMeasure`, and exactness records.
- Produces: `CategoricalLogits`, `NormalizedCategorical`, `KLDivergence`, `entropy(law)`, and `kl(q, p)` with orientation encoded as `KLDivergence(source=q, target=p)`.

- [ ] **Step 1: Write failing normalization, mask, self-KL, and enumeration tests**

```python
import torch

from vfe4.families.categorical import NormalizedCategorical
from vfe4.families.divergences import KLDivergence


def test_mask_precedes_normalization_and_self_kl_is_zero() -> None:
    law = NormalizedCategorical.from_logits(
        torch.tensor([0.2, 9.0, -0.4], dtype=torch.float64),
        allowed=torch.tensor([True, False, True]),
    )
    assert torch.isneginf(law.log_probs[1])
    assert torch.allclose(torch.logsumexp(law.log_probs, dim=-1), torch.zeros((), dtype=torch.float64))
    assert KLDivergence().evaluate(law, law) == 0.0
```

- [ ] **Step 2: Run the red categorical tests**

Run:

```powershell
python -m pytest tests\contracts\test_categorical_family.py tests\oracles\test_discrete_normalization.py --junitxml=artifacts\junit\phase1-categorical-red.xml
```

Expected: nonzero exit code because the categorical family is absent.

- [ ] **Step 3: Implement stable categorical normalization and exact enumeration**

`NormalizedCategorical.from_logits` validates finite allowed logits, applies `-inf` to forbidden entries before `torch.log_softmax`, and raises `EmptySupportError` if no entry is allowed. Entropy uses `torch.special.xlogy` so zero mass contributes zero. KL checks `q << p`; positive `q` mass at zero `p` mass returns positive infinity rather than a capped value.

```python
@dataclass(frozen=True, slots=True)
class NormalizedCategorical:
    log_probs: torch.Tensor
    support: FiniteSupport

    @classmethod
    def from_logits(cls, logits: torch.Tensor, allowed: torch.Tensor) -> "NormalizedCategorical":
        if not bool(allowed.any()):
            raise EmptySupportError("categorical support is empty")
        masked = logits.masked_fill(~allowed, float("-inf"))
        if not bool(torch.isfinite(masked[allowed]).all()):
            raise NonFiniteProbabilityError("allowed categorical logits must be finite")
        return cls(torch.log_softmax(masked, dim=-1), FiniteSupport.from_mask(allowed))
```

- [ ] **Step 4: Run the green categorical tests**

Run:

```powershell
python -m pytest tests\contracts\test_categorical_family.py tests\oracles\test_discrete_normalization.py --junitxml=artifacts\junit\phase1-categorical.xml
```

Expected: exit code 0; the XML has zero failures and errors.

- [ ] **Step 5: Commit the first normalized family**

```powershell
git add src\vfe4\families tests\contracts\test_categorical_family.py tests\oracles\test_discrete_normalization.py
git commit -m "feat: add normalized categorical laws and discrete oracle"
```

### Task 1.2: Implement full-Gaussian natural, expectation, and moment charts

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\charts.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\gaussian.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_gaussian_charts.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_gaussian_chart_equality.py`

**Interfaces:**

- Consumes: `VectorSpace` and strict SPD operations from `numerics.spd`.
- Produces: `GaussianInformation(information_vector, precision)`, `GaussianNatural(linear_natural, quadratic_natural)`, `GaussianExpectation(mean, second_moment)`, `GaussianMoments(mean, covariance)`, `FullGaussian`, and explicit conversions among the certified charts.

- [ ] **Step 1: Write failing round-trip and chart-domain tests**

```python
import torch

from vfe4.families.charts import GaussianMoments, information_to_natural, moments_to_information, natural_to_expectation


def test_full_gaussian_chart_round_trip_uses_second_moment() -> None:
    mean = torch.tensor([0.5, -0.25], dtype=torch.float64)
    covariance = torch.tensor([[1.2, 0.3], [0.3, 0.8]], dtype=torch.float64)
    moments = GaussianMoments(mean=mean, covariance=covariance)
    information = moments_to_information(moments)
    natural = information_to_natural(information)
    expectation = natural_to_expectation(natural)
    assert torch.allclose(expectation.mean, mean, atol=1e-13, rtol=1e-13)
    assert torch.allclose(
        expectation.second_moment,
        covariance + torch.outer(mean, mean),
        atol=1e-13,
        rtol=1e-13,
    )
```

- [ ] **Step 2: Run the red chart tests**

Run:

```powershell
python -m pytest tests\contracts\test_gaussian_charts.py tests\oracles\test_gaussian_chart_equality.py --junitxml=artifacts\junit\phase1-charts-red.xml
```

Expected: nonzero exit code because the Gaussian chart types do not exist.

- [ ] **Step 3: Implement named charts with solves and strict domains**

Use the canonical coordinates

```python
GaussianInformation(information_vector=h, precision=J)       # inference storage
GaussianNatural(linear_natural=h, quadratic_natural=-J / 2) # eta = (h, -J/2)
GaussianExpectation(mean=mu, second_moment=M)                # xi = (mu, M)
GaussianMoments(mean=mu, covariance=Sigma)                   # interpretation view
```

Conversions use `torch.cholesky_solve` or triangular solves and never call a dense inverse. `GaussianInformation` validates positive-definite precision; `GaussianNatural` validates negative-definite quadratic natural coordinates; and `GaussianExpectation` validates `M - mu mu^T` as SPD. The inference backend stores `GaussianInformation`; the law exposes the mathematical natural chart through an explicit conversion. `FullGaussian` never infers chart meaning from tensor rank.

- [ ] **Step 4: Run green chart tests under float64**

Run:

```powershell
python -m pytest tests\contracts\test_gaussian_charts.py tests\oracles\test_gaussian_chart_equality.py --junitxml=artifacts\junit\phase1-charts.xml
```

Expected: exit code 0; natural, expectation, and moment evaluations agree inside the fixture-derived float64 budget.

- [ ] **Step 5: Commit the full-Gaussian chart contract**

```powershell
git add src\vfe4\families\charts.py src\vfe4\families\gaussian.py tests\contracts\test_gaussian_charts.py tests\oracles\test_gaussian_chart_equality.py
git commit -m "feat: add full Gaussian natural and expectation charts"
```

### Task 1.3: Add Gaussian entropy, log normalizer, KL, sampling, and linear-Gaussian oracles

**Files:**

- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\gaussian.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\divergences.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\information.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\linear_gaussian.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_gaussian_identities.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_linear_gaussian_evidence.py`

**Interfaces:**

- Consumes: full-Gaussian charts and `DenseF64Oracle`.
- Produces: `FullGaussian.log_normalizer()`, `FullGaussian.entropy()`, `FullGaussian.sample()`, oriented analytic `KLDivergence.evaluate(q, p)`, `SquareRootInformationFactor`, `LinearGaussianFactor`, and `dense_linear_gaussian_evidence(model, observation)`.

- [ ] **Step 1: Write failing analytic-identity tests**

```python
import torch

from vfe4.families.divergences import KLDivergence
from vfe4.families.gaussian import FullGaussian


def test_gaussian_kl_orientation_and_self_divergence() -> None:
    q = FullGaussian.from_moments(
        mean=torch.tensor([0.0, 1.0], dtype=torch.float64),
        covariance=torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64),
    )
    p = FullGaussian.from_moments(
        mean=torch.tensor([0.5, -0.5], dtype=torch.float64),
        covariance=torch.tensor([[2.0, 0.1], [0.1, 0.7]], dtype=torch.float64),
    )
    divergence = KLDivergence()
    assert abs(float(divergence.evaluate(q, q))) < 1e-13
    assert not torch.allclose(divergence.evaluate(q, p), divergence.evaluate(p, q))
```

The linear-Gaussian evidence test assembles a two-variable model twice: once by direct marginal covariance and once through square-root information factors. It asserts equal log evidence and posterior moments.

- [ ] **Step 2: Run the red Gaussian-identity tests**

Run:

```powershell
python -m pytest tests\oracles\test_gaussian_identities.py tests\oracles\test_linear_gaussian_evidence.py --junitxml=artifacts\junit\phase1-gaussian-oracles-red.xml
```

Expected: nonzero exit code because the analytic Gaussian operations and factor compiler are absent.

- [ ] **Step 3: Implement analytic operations and square-root information assembly**

For (q=N(\mu_q,\Sigma_q)) and (p=N(\mu_p,\Sigma_p)), implement the oriented KL with solves and Cholesky log determinants. `SquareRootInformationFactor` stores `whitener @ design` and `whitener @ offset`; assembly adds (A^T P A) and the corresponding information vector. A strictly SPD anchor is mandatory. Sampling uses the Cholesky factor of precision and a triangular solve, not `solve(J, epsilon)`.

- [ ] **Step 4: Run the Phase 1 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles --junitxml=artifacts\junit\phase1-gate.xml
```

Expected: exit code 0; the XML has zero failures and errors; chart equality, normalization, evidence, self-KL, non-negativity, and sampling tests all run through float64.

- [ ] **Step 5: Commit the analytic Gaussian oracle slice**

```powershell
git add src\vfe4\families src\vfe4\factors tests\oracles\test_gaussian_identities.py tests\oracles\test_linear_gaussian_evidence.py
git commit -m "feat: add analytic Gaussian and linear evidence oracles"
```

### Task 1.4: Implement truthful Fisher-metric capability and natural-gradient updates

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\protocols.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\exponential.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\optimization_charts.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\fisher.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\categorical.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\gaussian.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\natural_gradient.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_family_capabilities.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_fisher_metric.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_natural_expectation_duality.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_natural_gradient_label.py`

**Interfaces:**

- Consumes: categorical and full-Gaussian normalized laws, exact chart conversions, automatic differentiation of log densities, and strict linear solves.
- Produces: compositional `Law`, `DiscreteLaw`, `ContinuousLaw`, `MixedLaw`, `ExponentialFamily`, `LegendreDualCapable`, `FisherMetricCapable`, `FisherOperator`, `CategoricalFisher`, `FullGaussianFisher`, `OptimizationChart`, and `NaturalGradientUpdate`.

- [ ] **Step 1: Write failing capability, Fisher, and dual-chart tests**

```python
def test_categorical_fisher_matches_score_covariance_on_tangent_space() -> None:
    law = categorical_law([0.2, 0.3, 0.5], dtype=torch.float64)
    tangent = torch.tensor([0.4, -0.1, -0.3], dtype=torch.float64)
    expected = (torch.diag(law.probs) - torch.outer(law.probs, law.probs)) @ tangent
    assert torch.allclose(law.fisher().apply(tangent), expected, atol=1e-13, rtol=1e-13)


def test_gaussian_natural_and_expectation_jacobians_are_inverse_duals() -> None:
    law = full_gaussian_fixture(dtype=torch.float64)
    natural_to_expectation_jacobian = jacobian_natural_to_expectation(law)
    expectation_to_natural_jacobian = jacobian_expectation_to_natural(law)
    identity = natural_to_expectation_jacobian @ expectation_to_natural_jacobian
    assert torch.allclose(identity, torch.eye(identity.shape[0], dtype=torch.float64), atol=1e-10, rtol=1e-10)


def test_redundant_categorical_logits_do_not_claim_bijective_dual_chart() -> None:
    law = categorical_with_full_logits([0.0, 1.0, 2.0], dtype=torch.float64)
    assert law.has_capability(ExponentialFamily)
    assert not law.has_capability(LegendreDualCapable)
    assert categorical_with_reference_logit(law.probs).has_capability(LegendreDualCapable)


def test_natural_gradient_label_requires_inverse_fisher_application() -> None:
    with pytest.raises(MissingFisherMetricError):
        NaturalGradientUpdate(metric=None).apply(euclidean_gradient_fixture())
```

- [ ] **Step 2: Run the red information-geometry tests**

Run:

```powershell
python -m pytest tests\contracts\test_family_capabilities.py tests\oracles\test_fisher_metric.py tests\oracles\test_natural_expectation_duality.py tests\integration\test_natural_gradient_label.py --junitxml=artifacts\junit\phase1-fisher-red.xml
```

Expected: nonzero exit code because compositional family and Fisher capabilities do not exist.

- [ ] **Step 3: Implement minimal compositional capabilities and exact initial metrics**

`Law` declares event space, support, reference measure, batch and event shapes, normalized log density or mass, and available sampling operations. Entropy, moments, pushforward, exponential-family charts, and Fisher geometry are optional typed capabilities, never universal Gaussian-shaped fields. `ExponentialFamily` supplies sufficient statistics, natural parameters, and log partition. `LegendreDualCapable` additionally requires a minimal regular representation or an explicit quotient or gauge fixing with a proved bijection on its chart domain. Full categorical logits therefore lack this capability until common-shift redundancy is removed. An `OptimizationChart` such as a Cholesky precision parameterization remains distinct from natural, expectation, moment, and information-storage coordinates.

For categorical laws, operate on an explicitly identified simplex tangent space rather than pseudo-inverting the singular ambient Fisher matrix. For full Gaussians, implement Fisher actions for mean and symmetric covariance or equivalent natural coordinates with solve-based operators. Verify analytic actions against score covariance or Hessians of the log partition on deterministic float64 fixtures. `NaturalGradientUpdate` earns `NATURAL_GRADIENT_STEP` only after applying the registered inverse Fisher operator and recording its solve residual; merely storing `(information_vector, precision)` is insufficient.

- [ ] **Step 4: Run the complete green information-geometry gate**

Run:

```powershell
python -m pytest tests\contracts\test_family_capabilities.py tests\oracles\test_fisher_metric.py tests\oracles\test_natural_expectation_duality.py tests\integration\test_natural_gradient_label.py --junitxml=artifacts\junit\phase1-fisher.xml
```

Expected: exit code 0; analytic Fisher actions match independent float64 references, dual chart Jacobians compose to identity within tolerance, and Euclidean stand-ins cannot receive a natural-gradient label.

- [ ] **Step 5: Commit the information-geometry capability slice**

```powershell
git add src\vfe4\families src\vfe4\inference\natural_gradient.py tests\contracts\test_family_capabilities.py tests\oracles\test_fisher_metric.py tests\oracles\test_natural_expectation_duality.py tests\integration\test_natural_gradient_label.py
git commit -m "feat: add Fisher metric and natural gradient contracts"
```

## Phase 2: Causal Language-Model Oracle and Shared ELBO

### Task 2.1: Compile the normalized causal model from explicit factors

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\initial.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\source.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\transition.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\emission.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\compiler.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\tokens.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\causal.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_causal_factor_graph.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\tiny_model_factory.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_causal_joint_normalization.py`

**Interfaces:**

- Consumes: validated model graph, normalized categorical laws, full Gaussians, and linear-Gaussian factors.
- Produces: `Token`, `TokenPrefix`, `ParentSet[Channel]`, `InitialFactor`, `SourcePriorFactor[Channel]`, `GaussianTransitionFactor[Channel]`, `LinearCategoricalEmission`, `CausalModelSpec`, and `compile_causal_model(spec) -> CompiledCausalModel`.

- [ ] **Step 1: Write failing factor-normalization and causal-dependency tests**

```python
from vfe4.graph.compiler import compile_causal_model
from tests.oracles.tiny_model_factory import make_immediate_predecessor_model


def test_compiled_causal_joint_normalizes_and_has_no_target_edge() -> None:
    model = compile_causal_model(make_immediate_predecessor_model(vocabulary_size=3, length=2))
    assert model.graph.generative_factor_kinds == (
        "initial",
        "model_source",
        "model_transition",
        "state_source",
        "state_transition",
        "emission",
    )
    assert model.graph.current_observation_not_in_predictive_ancestors()
    assert abs(model.enumerate_token_evidence().sum() - 1.0) < 1e-13
```

- [ ] **Step 2: Run the red causal-factor tests**

Run:

```powershell
python -m pytest tests\contracts\test_causal_factor_graph.py tests\oracles\test_causal_joint_normalization.py --junitxml=artifacts\junit\phase2-causal-graph-red.xml
```

Expected: nonzero exit code because the causal factors and compiler do not exist.

- [ ] **Step 3: Implement the fixed normalized joint in generative order**

`CompiledCausalModel.log_joint` evaluates exactly one factor of each kind per time step:

```python
def log_joint(self, trace: CompleteTrace, observations: TokenSequence) -> torch.Tensor:
    total = self.initial.log_prob(trace.z[0], trace.m[0])
    for time in range(1, self.length + 1):
        prefix = observations.before(time)
        source_context = self.source_contexts[time]
        total = total + self.model_sources[time].log_prob(trace.b[time], source_context)
        total = total + self.model_transitions[time].log_prob(trace.m[time], trace, prefix)
        total = total + self.state_sources[time].log_prob(trace.a[time], source_context)
        total = total + self.state_transitions[time].log_prob(trace.z[time], trace, prefix)
        total = total + self.emissions[time].log_prob(observations[time], trace.z[time], trace.m[time])
    return total
```

For V1, `source_context` contains only the declared causal parent set, source and receiver positions, mask, and deterministic geometry. It contains no token values or recognition state. History-conditioned generative source priors are a later separately normalized model extension, not an implicit field on this base context. The compiler validates nonempty causal parent sets, normalized source priors, SPD receiver covariances, receiver-fiber dimensions, and a normalized categorical emission. No transition coefficient receives the current target token.

- [ ] **Step 4: Run green normalization tests**

Run:

```powershell
python -m pytest tests\contracts\test_causal_factor_graph.py tests\oracles\test_causal_joint_normalization.py --junitxml=artifacts\junit\phase2-causal-graph.xml
```

Expected: exit code 0; exact discrete enumeration sums to one within the fixture-derived float64 tolerance.

- [ ] **Step 5: Commit the normalized causal model**

```powershell
git add src\vfe4\factors src\vfe4\graph\compiler.py src\vfe4\language tests\contracts\test_causal_factor_graph.py tests\oracles\tiny_model_factory.py tests\oracles\test_causal_joint_normalization.py
git commit -m "feat: compile the normalized causal generative model"
```

### Task 2.2: Separate target-aware recognition from target-blind prediction

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\recognition.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\predictive.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\evaluate.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\generate.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_inference_role_types.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_causal_no_leakage.py`

**Interfaces:**

- Consumes: `CompiledCausalModel`, token types, and normalized laws.
- Produces: `PrefixFilterState`, `PredictiveState`, `ObservedStep`, `RecognitionContext`, `FilteringRecognitionLaw`, `PriorPredictiveBackend`, `TrainingRecognitionBackend`, `evaluate_next_token`, and `generate_next_token`.

- [ ] **Step 1: Write failing target-mutation and suffix-mutation tests**

```python
import torch

from vfe4.inference.predictive import ExactPrefixBackend
from tests.oracles.tiny_model_factory import compiled_tiny_model


def test_predictive_law_depends_on_shared_prefix_but_not_target_or_suffix() -> None:
    backend = ExactPrefixBackend(compiled_tiny_model())
    sequence_one = (0, 1, 2)
    sequence_two = (0, 2, 1)
    predictive_one = backend.predict_before_target(sequence_one, time=1)
    predictive_two = backend.predict_before_target(sequence_two, time=1)
    assert torch.equal(
        backend.token_law(predictive_one).log_probs,
        backend.token_law(predictive_two).log_probs,
    )
    assert predictive_one.dependencies.includes("normalized_prefix_filter_state")
    assert predictive_one.dependencies.excludes("current_target")
    assert predictive_one.dependencies.excludes("future_suffix")


def test_recognition_is_allowed_to_use_the_observation() -> None:
    backend = ExactPrefixBackend(compiled_tiny_model())
    predicted = backend.predict(backend.filter_observed_prefix(tokens=(0,)))
    q_one = backend.training_recognition(predicted, observed_token=1)
    q_two = backend.training_recognition(predicted, observed_token=2)
    assert not q_one.value_equal(q_two)
```

- [ ] **Step 2: Run the red inference-role tests**

Run:

```powershell
python -m pytest tests\contracts\test_inference_role_types.py tests\integration\test_causal_no_leakage.py --junitxml=artifacts\junit\phase2-inference-roles-red.xml
```

Expected: nonzero exit code because the predictive and recognition role types do not exist.

- [ ] **Step 3: Implement role-specific APIs with no shared catch-all forward method**

```python
@dataclass(frozen=True, slots=True)
class PrefixFilterState:
    time: int
    prefix: TokenPrefix
    filtered_law: Law


@dataclass(frozen=True, slots=True)
class PredictiveState:
    time: int
    prefix: TokenPrefix
    predicted_law: Law


@dataclass(frozen=True, slots=True)
class ObservedStep:
    predictive: PredictiveState
    token: Token


class PriorPredictiveBackend(Protocol):
    def predict(self, state: PrefixFilterState) -> PredictiveState: ...
    def token_law(self, state: PredictiveState) -> NormalizedCategorical: ...
    def assimilate(self, step: ObservedStep) -> PrefixFilterState: ...
```

`runtime.evaluate` scores `token_law(predict(state))` before constructing `ObservedStep`. `runtime.generate` calls the same two methods, samples from the returned law, and assimilates the sampled token. `FilteringRecognitionLaw` is accepted by ELBO training APIs and rejected by primary predictive-metric APIs.

- [ ] **Step 4: Run green no-leakage tests, including stepwise versus batched prefix equality**

Run:

```powershell
python -m pytest tests\contracts\test_inference_role_types.py tests\integration\test_causal_no_leakage.py --junitxml=artifacts\junit\phase2-inference-roles.xml
```

Expected: exit code 0; target and suffix mutations leave pre-assimilation token laws unchanged, while recognition laws may change.

- [ ] **Step 5: Commit the causal role boundary**

```powershell
git add src\vfe4\inference\recognition.py src\vfe4\inference\predictive.py src\vfe4\runtime tests\contracts\test_inference_role_types.py tests\integration\test_causal_no_leakage.py
git commit -m "feat: separate posterior recognition from prior prediction"
```

### Task 2.3: Assemble the complete ELBO and enforce M-step parameter ownership

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\elbo.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\decomposition.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\updates.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\emission.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_elbo_inventory.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_emission_cross_entropy.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_mstep_accounting.py`

**Interfaces:**

- Consumes: compiled model, recognition law, categorical emission, exactness ledger, and graph parameter ownership.
- Produces: `ELBOResult`, `ELBOTerm`, `evaluate_monolithic_elbo`, `evaluate_local_elbo`, `DecoderMCoordinate`, `GradientProposal`, and `accept_generalized_em`.

- [ ] **Step 1: Write failing complete-inventory and decoder-ownership tests**

```python
from vfe4.objectives.elbo import evaluate_local_elbo
from vfe4.objectives.updates import DecoderMCoordinate
from tests.oracles.tiny_model_factory import tiny_model_and_recognition


def test_local_elbo_counts_each_factor_once_and_source_kl_once() -> None:
    model, recognition, observations = tiny_model_and_recognition()
    result = evaluate_local_elbo(model, recognition, observations)
    assert result.inventory.names() == (
        "expected_emission",
        "initial_kl",
        "model_source_kl",
        "model_transition_kl",
        "state_source_kl",
        "state_transition_kl",
    )
    assert result.inventory.coefficient("model_source_kl") == 1.0
    assert result.inventory.coefficient("state_source_kl") == 1.0


def test_decoder_coordinate_rejects_parameters_owned_by_other_factors() -> None:
    model, recognition, observations = tiny_model_and_recognition(shared_decoder_transition=True)
    with pytest.raises(ValueError, match="decoder parameter has non-emission owners"):
        DecoderMCoordinate(model.graph).objective(recognition, observations)
```

- [ ] **Step 2: Run the red ELBO and M-step tests**

Run:

```powershell
python -m pytest tests\contracts\test_elbo_inventory.py tests\oracles\test_emission_cross_entropy.py tests\integration\test_mstep_accounting.py --junitxml=artifacts\junit\phase2-elbo-red.xml
```

Expected: nonzero exit code because the ELBO assembler and update labels do not exist.

- [ ] **Step 3: Implement stable expected cross-entropy and complete-objective accounting**

`LinearCategoricalEmission` returns normalized log probabilities from

```python
logits = state_readout @ z + model_readout @ m + bias
log_probs = torch.log_softmax(logits, dim=-1)
```

The expected negative log emission is categorical cross-entropy only. The ELBO additionally contains initial, transition, source, and recognition-entropy terms. `ELBOResult` stores the scalar, ordered term inventory, evaluation status per term, stochastic error budget, and approximation-ledger hash.

`DecoderMCoordinate` is legal only when all candidate parameters are owned solely by emission factors. A finite optimizer step returns `UNVERIFIED_FINITE_STEP`. `accept_generalized_em` reevaluates the complete ELBO and returns `ACCEPTED_GENERALIZED_EM` only when a deterministic bound, or a predeclared confidence procedure, proves that the lower bound on the improvement is nonnegative after accounting for both evaluations. Successive numerical estimates without such a bound remain stochastic or unverified updates.

- [ ] **Step 4: Run green ELBO tests and verify dense versus fused CE equality**

Run:

```powershell
python -m pytest tests\contracts\test_elbo_inventory.py tests\oracles\test_emission_cross_entropy.py tests\integration\test_mstep_accounting.py --junitxml=artifacts\junit\phase2-elbo.xml
```

Expected: exit code 0; dense and chunked full-vocabulary CE agree inside the declared float64 budget, and no update is mislabeled as generalized EM.

- [ ] **Step 5: Commit the shared objective and update boundary**

```powershell
git add src\vfe4\objectives src\vfe4\language\emission.py tests\contracts\test_elbo_inventory.py tests\oracles\test_emission_cross_entropy.py tests\integration\test_mstep_accounting.py
git commit -m "feat: assemble the complete ELBO and M-step accounting"
```

### Task 2.4: Build the semantically exact tiny-language reference and Phase 2 promotion gate

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\quadrature.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\tiny_model_factory.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_tiny_causal_lm.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_evaluation_generation_identity.py`

**Interfaces:**

- Consumes: every Phase 0 through Phase 2 contract.
- Produces: `TinyLanguageOracle`, `DeterministicQuadratureResult`, `QuadratureCalibration`, and a fixed fixture with `T=2`, `d_z=d_m=1`, `V=3`, positive source priors, parents `{0}` at time 1 and `{0,1}` at time 2, and deterministic flat geometry.

- [ ] **Step 1: Write failing three-way ELBO and predictive-recursion tests**

```python
def test_tiny_language_oracle_closes_all_identities() -> None:
    oracle = TinyLanguageOracle.reference()
    report = oracle.evaluate_analytic_fixture()
    assert report.joint_normalization_residual <= report.rounding_budget
    assert report.predictive_row_residual <= report.rounding_budget
    assert report.chain_rule_residual <= report.rounding_budget
    assert report.monolithic_local_residual <= report.total_budget
    assert report.evidence_gap_residual <= report.total_budget
    assert report.information_moment_residual <= report.rounding_budget


def test_evaluation_and_generation_share_the_pre_assimilation_law() -> None:
    oracle = TinyLanguageOracle.reference()
    prefix = (2,)
    assert torch.equal(
        oracle.evaluation_token_law(prefix).log_probs,
        oracle.generation_token_law(prefix).log_probs,
    )
```

- [ ] **Step 2: Run the red tiny-language gate**

Run:

```powershell
python -m pytest tests\oracles\test_tiny_causal_lm.py tests\integration\test_evaluation_generation_identity.py --junitxml=artifacts\junit\phase2-tiny-red.xml
```

Expected: nonzero exit code because the tiny oracle and deterministic quadrature are absent.

- [ ] **Step 3: Implement analytic and latent-dependent emission cases**

The analytic case sets both latent readouts to zero and uses nonuniform vocabulary bias, so all continuous expectations are analytic. The latent-dependent case uses nonzero readouts and tensor-product Gauss-Hermite quadrature with order doubling. Its acceptance threshold is fixed from an independently implemented deterministic float64-or-higher reference calculation on the bounded fixture, with method, precision, reference value, and calibration residual stored in `QuadratureCalibration`. The difference between consecutive Gauss-Hermite orders is recorded only as a convergence diagnostic; it is never promoted into an error bound or acceptance budget by itself. This branch reports `DETERMINISTIC_QUADRATURE` and never calls the integral analytic.

The oracle enumerates every labeled source assignment and all (3^2) observed token sequences. It checks the product of target-blind token conditionals against joint evidence, then assimilates each token. It separately evaluates a target-aware filtering recognition law and confirms that target mutation changes recognition without changing the pre-assimilation predictive law.

- [ ] **Step 4: Run the complete Phase 2 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles tests\integration\test_causal_no_leakage.py tests\integration\test_mstep_accounting.py tests\integration\test_evaluation_generation_identity.py --junitxml=artifacts\junit\phase2-gate.xml
```

Expected: exit code 0; inspect `phase2-gate.xml` for zero failures and errors before allowing Phase 3 work.

- [ ] **Step 5: Commit the causal-LM oracle gate**

```powershell
git add src\vfe4\numerics\quadrature.py tests\oracles\tiny_model_factory.py tests\oracles\test_tiny_causal_lm.py tests\integration\test_evaluation_generation_identity.py
git commit -m "test: gate causal language modeling on the tiny oracle"
```

### Task 2.5: Run the bounded CPU B0-versus-minimal-pq falsifier

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\train.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\data.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\metrics.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\experiments\pilot.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\experiments\run_minimal_pq_pilot.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_pilot_protocol.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_minimal_pq_predictive_smoke.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\artifacts\pilots\minimal-pq-pilot-cpu.json`

**Interfaces:**

- Consumes: the semantically exact causal model, bounded dense Gaussian backend, normalized recognition disintegration, target-blind predictive API, categorical emission, and complete objective ledger.
- Produces: `DeterministicStateBaseline`, `MinimalPQCell`, `PilotProtocol`, `PilotResult`, and a bounded CPU comparison artifact before any structured production, gauge, hierarchy, sparse, mixture, or irrep work.

- [ ] **Step 1: Write failing nested-baseline and causal-metric tests**

```python
def test_b0_is_the_deterministic_state_restriction_of_the_same_model_interfaces() -> None:
    b0, pq = matched_minimal_cells()
    assert b0.transition_signature == pq.generative_transition_signature
    assert b0.emission_signature == pq.emission_signature
    assert b0.objective_terms == ("categorical_observation_nll",)
    assert b0.has_variational_posterior is False


def test_cpu_pilot_metrics_filter_the_prefix_without_using_the_scored_target() -> None:
    result = run_tiny_pilot_cpu()
    assert result.predictive_dependencies.includes("normalized_prefix_filter_state")
    assert result.predictive_dependencies.excludes("current_target")
    assert result.metrics_are_finite
```

- [ ] **Step 2: Run the red CPU pilot gate**

```powershell
python -m pytest tests\contracts\test_pilot_protocol.py tests\integration\test_minimal_pq_predictive_smoke.py --junitxml=artifacts\junit\phase2-early-pilot-red.xml
```

Expected: a nonzero exit because the nested B0 and minimal `pq` cells and pilot protocol do not exist.

- [ ] **Step 3: Implement only the bounded dense CPU vertical slice**

B0 is a deterministic-state restriction of the same causal transition and normalized emission interfaces. It has no variational posterior, latent entropy, or latent KL, and it uses typed tensor parameters rather than `nn.Linear` or another neural module. `MinimalPQCell` adds only the state latent and normalized recognition law needed for the ordinary `pq` ELBO. The default pilot uses a hash-pinned WikiText-103 shard and the tokenizer identity from the established V3 baseline protocol, recorded as external data provenance rather than imported through a V3 runtime dependency. It fixes context, seed, update count, parameter accounting, evaluation points, and the target-blind predictive metric path. It reports held-out NLL, perplexity, wall time, peak memory, ELBO sectors, posterior-prior KL, collapse diagnostics, evaluation method, and numerical error. It does not claim parameter or compute matching when those quantities differ.

- [ ] **Step 4: Run the green CPU gate and the fixed click-run pilot**

```powershell
python -m pytest tests\contracts\test_pilot_protocol.py tests\integration\test_minimal_pq_predictive_smoke.py --junitxml=artifacts\junit\phase2-early-pilot.xml
python experiments\run_minimal_pq_pilot.py
```

Expected: the XML has zero failures and errors, and `minimal-pq-pilot-cpu.json` contains complete B0 and minimal-`pq` records. A negative scientific result is preserved as a falsification result rather than converted into a software failure. Before Phase 3 begins, `docs/build-handoff.md` records an explicit go, revise-minimal-model, or stop decision. No optional infrastructure proceeds without that record.

- [ ] **Step 5: Commit the CPU falsifier and inspected evidence**

```powershell
git add src\vfe4\runtime\train.py src\vfe4\language\data.py src\vfe4\language\metrics.py src\vfe4\experiments\pilot.py experiments\run_minimal_pq_pilot.py tests\contracts\test_pilot_protocol.py tests\integration\test_minimal_pq_predictive_smoke.py docs\build-handoff.md artifacts\pilots\minimal-pq-pilot-cpu.json artifacts\junit\phase2-early-pilot.xml
git commit -m "test: run the bounded CPU minimal pq falsifier"
```

## Phase 3: Structured Recognition and Information Backends

### Task 3.1: Add structured Gaussian recognition and explicit moment requests

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\moments.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\blankets.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\structured.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\mean_field.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\backend.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\certificates.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_moment_request_compiler.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_update_certificates.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_structured_dense_q.py`

**Interfaces:**

- Consumes: graph factor inventory and `PrecisionBackend`.
- Produces: `BlockLayout`, `MomentRequest`, `MomentRequestSet`, `MarkovBlanket`, `UpdateCertificate`, `compile_moment_requests(graph, objective)`, `StructuredGaussianLaw(information_vector, precision_structure, backend)`, and explicit `MeanFieldGaussianLaw`.

- [ ] **Step 1: Write failing request-minimality and dense structured-Q tests**

```python
def test_emission_requests_only_token_marginal_blocks() -> None:
    graph = compiled_tiny_model().graph
    requests = compile_moment_requests(graph, objective="ordinary_conditional_elbo")
    assert requests.contains_token_marginals()
    assert not requests.contains_global_covariance()


def test_structured_dense_q_matches_full_gaussian() -> None:
    fixture = coupled_gaussian_fixture(dtype=torch.float64)
    structured = StructuredGaussianLaw.from_factors(fixture.factors, backend=DenseF64Oracle(32))
    assert torch.allclose(structured.mean(), fixture.posterior_mean, atol=1e-12, rtol=1e-12)


def test_exact_coordinate_certificate_covers_every_affected_ledger_row() -> None:
    graph = compiled_tiny_model().graph
    certificate = compile_update_certificate(graph, variable_id="state[1]")
    assert certificate.reads == graph.markov_blanket("state[1]").current_values
    assert certificate.ledger_rows == graph.ledger_rows_affected_by("state[1]")
```

- [ ] **Step 2: Run the red structured-recognition tests**

Run:

```powershell
python -m pytest tests\contracts\test_moment_request_compiler.py tests\contracts\test_update_certificates.py tests\oracles\test_structured_dense_q.py --junitxml=artifacts\junit\phase3-structured-red.xml
```

Expected: nonzero exit code because moment requests and structured recognition do not exist.

- [ ] **Step 3: Implement request-driven structured Gaussian evaluation**

`StructuredGaussianLaw` owns an `information_vector`, a typed precision structure, and a factorization handle. It exposes mean solves, selected moments, log normalizer, entropy, and sampling. It does not expose unrestricted covariance. `MeanFieldGaussianLaw` is a separately named restricted variational family and records `RESTRICTED_VARIATIONAL_FAMILY`; it is never an implicit default. The objective compiler derives requests and Markov blankets from actual factor incidence; diagnostics cannot silently widen the request set. An exact-coordinate certificate proves that a step reads current values throughout its blanket and optimizes every affected ledger row. Parallel schedules require conditional independence or a declared Jacobi bound and otherwise fail compilation.

- [ ] **Step 4: Run green structured-recognition tests**

Run:

```powershell
python -m pytest tests\contracts\test_moment_request_compiler.py tests\contracts\test_update_certificates.py tests\oracles\test_structured_dense_q.py --junitxml=artifacts\junit\phase3-structured.xml
```

Expected: exit code 0; dense structured and direct full-Gaussian values agree.

- [ ] **Step 5: Commit the structured-Q contract**

```powershell
git add src\vfe4\graph\moments.py src\vfe4\graph\blankets.py src\vfe4\inference\structured.py src\vfe4\inference\mean_field.py src\vfe4\inference\backend.py src\vfe4\objectives\certificates.py tests\contracts\test_moment_request_compiler.py tests\contracts\test_update_certificates.py tests\oracles\test_structured_dense_q.py
git commit -m "feat: add structured Gaussian recognition and moment requests"
```

### Task 3.2: Implement the block-banded precision backend

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\block_banded.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_block_banded_structure.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_block_banded_dense_equality.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_block_banded_no_dense_allocation.py`

**Interfaces:**

- Consumes: `BlockLayout`, square-root information factors, and `MomentRequestSet`.
- Produces: `BlockBandedStructure(block_size, half_bandwidth, length)`, `BlockBandedBackend`, and `BlockBandedFactorization` with solve, log determinant, requested-block inverse, and triangular-solve sampling.

- [ ] **Step 1: Write failing dense-equality and no-global-matrix tests**

```python
def test_block_banded_matches_dense_oracle() -> None:
    fixture = block_tridiagonal_fixture(length=5, block_size=2, dtype=torch.float64)
    banded = BlockBandedBackend().factorize(fixture.banded_system)
    dense = DenseF64Oracle(max_dimension=32).factorize(fixture.dense_precision)
    assert torch.allclose(banded.solve(fixture.rhs).value, dense.solve(fixture.rhs).value, atol=1e-11)
    assert torch.allclose(banded.logdet().value, dense.logdet().value, atol=1e-11)


def test_banded_api_has_no_global_materializer() -> None:
    assert not hasattr(BlockBandedFactorization, "to_dense")
    assert not hasattr(BlockBandedFactorization, "covariance")
```

- [ ] **Step 2: Run the red block-banded tests**

Run:

```powershell
python -m pytest tests\contracts\test_block_banded_structure.py tests\oracles\test_block_banded_dense_equality.py tests\integration\test_block_banded_no_dense_allocation.py --junitxml=artifacts\junit\phase3-banded-red.xml
```

Expected: nonzero exit code because the block-banded backend is absent.

- [ ] **Step 3: Implement block Cholesky without global covariance**

Store only diagonal and declared lower-band blocks. The numeric plan performs block Cholesky, forward/back substitution, log determinant from diagonal Cholesky blocks, selected covariance by bounded basis solves or a banded selected-inverse recursion, and sampling by triangular solve. Validate each pivot block with `cholesky_ex`; a failed pivot rejects the candidate without jitter.

- [ ] **Step 4: Run green equality and allocation tests**

Run:

```powershell
python -m pytest tests\contracts\test_block_banded_structure.py tests\oracles\test_block_banded_dense_equality.py tests\integration\test_block_banded_no_dense_allocation.py --junitxml=artifacts\junit\phase3-banded.xml
```

Expected: exit code 0; values match dense float64 and no tensor with global `(T*K, T*K)` extent is produced by the structured path.

- [ ] **Step 5: Commit the first scalable information backend**

```powershell
git add src\vfe4\numerics\block_banded.py tests\contracts\test_block_banded_structure.py tests\oracles\test_block_banded_dense_equality.py tests\integration\test_block_banded_no_dense_allocation.py
git commit -m "feat: add the block-banded precision backend"
```

### Task 3.3: Promote dense and block-banded execution to the RTX 5090

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\cuda.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\torch_dense.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\torch_banded.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\device.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\experiments\run_minimal_pq_pilot_cuda.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_early_cuda_dense_banded.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\artifacts\environment\rtx5090.json`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\artifacts\pilots\minimal-pq-pilot-rtx5090.json`

**Interfaces:**

- Consumes: the recorded CPU falsifier decision, dense and block-banded CPU references, target-blind pilot protocol, tested dependency resolution, and operation-specific tolerances.
- Produces: `CudaFloat32Policy`, `TorchDenseBackend`, `TorchBlockBandedBackend`, an RTX 5090 environment record, and the device-specific rerun of the already defined pilot.

- [ ] **Step 1: Write the failing CUDA agreement test**

```python
@pytest.mark.cuda
def test_early_cuda_dense_and_banded_match_float64_references() -> None:
    report = run_early_rtx5090_oracle_gate()
    assert report.hardware.device_name == "NVIDIA GeForce RTX 5090"
    assert report.dense.within_registered_tolerance
    assert report.block_banded.within_registered_tolerance
    assert report.cuda_tests_skipped == 0
```

- [ ] **Step 2: Run the red RTX gate**

```powershell
python -m pytest tests\cuda\test_early_cuda_dense_banded.py --junitxml=artifacts\junit\phase3-early-cuda-red.xml
```

Expected: a nonzero exit because production dense and block-banded CUDA kernels do not exist.

- [ ] **Step 3: Implement the direct CUDA lowerings**

Use float32 tensors and disable autocast and TF32 inside SPD assembly, factorization, solves, log determinants, sampling, and oracle comparisons. Both lowerings report separate linear-algebra and arithmetic records and compare against the same float64 CPU problems. Record Python, PyTorch, CUDA runtime, driver, GPU identity, library versions, tolerance-registry digest, and test artifact hashes in `artifacts/environment/rtx5090.json`. A skipped CUDA test is not a passed promotion gate.

- [ ] **Step 4: Run the green RTX gate and the fixed CUDA pilot**

```powershell
python -m pytest tests\cuda\test_early_cuda_dense_banded.py --junitxml=artifacts\junit\phase3-early-cuda.xml
python experiments\run_minimal_pq_pilot_cuda.py
```

Expected: the XML has zero failures, errors, and skips, and the CUDA pilot preserves the CPU protocol while adding device-specific NLL, compute, memory, and numerical records.

- [ ] **Step 5: Commit the direct CUDA promotion**

```powershell
git add src\vfe4\numerics\cuda.py src\vfe4\numerics\torch_dense.py src\vfe4\numerics\torch_banded.py src\vfe4\runtime\device.py experiments\run_minimal_pq_pilot_cuda.py tests\cuda\test_early_cuda_dense_banded.py artifacts\environment\rtx5090.json artifacts\pilots\minimal-pq-pilot-rtx5090.json artifacts\junit\phase3-early-cuda.xml
git commit -m "feat: promote dense and banded execution to RTX 5090"
```

### Task 3.4: Add the static sparse-direct CPU backend

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\sparse.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_sparse_structure.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_sparse_dense_equality.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_sparse_no_dense_allocation.py`

**Interfaces:**

- Consumes: immutable CSR precision pattern, square-root factor contributions, and requested blocks.
- Produces: `SparsePrecisionStructure`, `SparseDirectCPU`, `SparseSymbolicPlan`, `SparseFactorization`, fill-ratio report, solve, log determinant, selected-block solves, and perturb-and-solve sampling. Initial sparse factorization is value-only and advertises no autodiff capability.

- [ ] **Step 1: Write failing sparse equality and capability tests**

```python
def test_sparse_direct_matches_dense_and_declares_no_autodiff() -> None:
    fixture = sparse_spd_fixture(dimension=12, seed=17)
    sparse = SparseDirectCPU().factorize(fixture.csr_precision)
    dense = DenseF64Oracle(max_dimension=32).factorize(fixture.dense_precision)
    assert sparse.manifest.autodiff == "none"
    assert torch.allclose(sparse.solve(fixture.rhs).value, dense.solve(fixture.rhs).value, atol=1e-10)
    assert torch.allclose(sparse.logdet().value, dense.logdet().value, atol=1e-10)
```

- [ ] **Step 2: Run the red sparse tests**

Run:

```powershell
python -m pytest tests\contracts\test_sparse_structure.py tests\oracles\test_sparse_dense_equality.py tests\integration\test_sparse_no_dense_allocation.py --junitxml=artifacts\junit\phase3-sparse-red.xml
```

Expected: nonzero exit code because `SparseDirectCPU` does not exist.

- [ ] **Step 3: Implement immutable CSR assembly and direct CPU solves**

Use SciPy CSR/CSC plus a statically imported direct factorization. Cache only symbolic pattern metadata; invalidate every numeric factorization after values change. Reject nonfinite entries, asymmetry beyond the declared input tolerance, failed factorization, nonpositive determinant sign, or solve residual above budget. Selected inverse blocks use explicit basis right-hand sides and never request the complete identity. Record fill ratio and peak allocated bytes.

- [ ] **Step 4: Run green sparse tests and block-banded overlap tests**

Run:

```powershell
python -m pytest tests\contracts\test_sparse_structure.py tests\oracles\test_sparse_dense_equality.py tests\oracles\test_block_banded_dense_equality.py tests\integration\test_sparse_no_dense_allocation.py --junitxml=artifacts\junit\phase3-sparse.xml
```

Expected: exit code 0; sparse, block-banded, and dense values agree on their overlap fixtures.

- [ ] **Step 5: Commit the sparse backend**

```powershell
git add src\vfe4\numerics\sparse.py tests\contracts\test_sparse_structure.py tests\oracles\test_sparse_dense_equality.py tests\integration\test_sparse_no_dense_allocation.py
git commit -m "feat: add the static sparse-direct CPU backend"
```

### Task 3.5: Make every projection and backend fallback explicit

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\projections.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_projection_contract.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_backend_failure_ledger.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_dense_structured_elbo_equality.py`

**Interfaces:**

- Consumes: structured laws, explicit target family, requested sparsity pattern, complete ELBO evaluator, and approximation ledger.
- Produces: `ProjectionRequest`, `ProjectedLaw`, `ProjectionReport`, `project_gaussian_forward_kl`, and `reject_or_record_backend_failure`.

- [ ] **Step 1: Write failing explicit-projection and failure-ledger tests**

```python
def test_projection_changes_type_and_records_error() -> None:
    source = dense_gaussian_fixture()
    request = ProjectionRequest(target="block_diagonal", retained_blocks=((0, 0), (1, 1)))
    projected = project_gaussian_forward_kl(source, request)
    assert isinstance(projected, ProjectedLaw)
    assert projected.report.family_relation is FamilyRelation.PROJECTED_FAMILY
    assert projected.report.projection_error is not None
    assert projected.report.objective_evaluation is None


def test_failed_factorization_never_returns_a_repaired_handle() -> None:
    with pytest.raises(NonSPDPrecisionError):
        BlockBandedBackend().factorize(indefinite_banded_fixture())
    assert last_ledger_event().linear_algebra is LinearAlgebraStatus.FAILED
```

- [ ] **Step 2: Run the red projection and ledger tests**

Run:

```powershell
python -m pytest tests\contracts\test_projection_contract.py tests\integration\test_backend_failure_ledger.py tests\integration\test_dense_structured_elbo_equality.py --junitxml=artifacts\junit\phase3-projection-red.xml
```

Expected: nonzero exit code because typed projections and backend failure integration are absent.

- [ ] **Step 3: Implement opt-in projection and fresh-candidate semantics**

`ProjectedLaw` cannot be passed where `ExactDeclaredLaw` is required without an explicit acceptance policy. Projection changes the family-relation axis; it does not assign an objective-evaluation label until the new law is evaluated. A regularized or rescaled precision receives a new candidate ID, `REGULARIZED_CANDIDATE` linear-algebra status, a new arithmetic report, and a new complete-ELBO evaluation. Backend selection never catches a failure and silently tries a less exact implementation.

- [ ] **Step 4: Run the complete Phase 3 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles tests\integration --junitxml=artifacts\junit\phase3-gate.xml
```

Expected: exit code 0; inspect the XML for zero failures and errors. Dense, block-banded, and sparse ELBOs agree on shared fixtures, and failure/projection events are present in the ledger.

- [ ] **Step 5: Commit the explicit approximation boundary**

```powershell
git add src\vfe4\inference\projections.py tests\contracts\test_projection_contract.py tests\integration\test_backend_failure_ledger.py tests\integration\test_dense_structured_elbo_equality.py
git commit -m "feat: type projections and backend failures explicitly"
```

## Phase 4: Principal and Associated-Bundle Geometry

### Task 4.1: Define groups, representations, and typed fibers

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\groups.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\representations.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\fibers.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_group_representation_contract.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_representation_laws.py`

**Interfaces:**

- Consumes: `VectorSpace`, strict tensor validation, and frozen catalogs.
- Produces: `GroupElement[G]`, `MatrixLieGroup`, `GeneralLinearGroup`, `TrivialGroup`, `Representation[G, V]`, `TrivialRepresentation`, `DefiningRepresentation`, `DualRepresentation`, and `Fiber[Base, V]`.

- [ ] **Step 1: Write failing group-law and representation-law tests**

```python
import torch

from vfe4.geometry.groups import GeneralLinearGroup
from vfe4.geometry.representations import DefiningRepresentation, DualRepresentation


def test_gl_group_and_dual_representation_laws() -> None:
    group = GeneralLinearGroup(dimension=2, dtype=torch.float64)
    g = group.element(torch.tensor([[2.0, 0.3], [0.0, 0.8]], dtype=torch.float64))
    h = group.element(torch.tensor([[1.1, 0.0], [0.2, 1.4]], dtype=torch.float64))
    rho = DefiningRepresentation(group)
    dual = DualRepresentation(rho)
    assert torch.allclose(rho.matrix(group.compose(g, h)), rho.matrix(g) @ rho.matrix(h))
    assert torch.allclose(dual.matrix(g), torch.linalg.inv(rho.matrix(g)).transpose(-1, -2))
    reflection = group.element(torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64)))
    assert group.component_id(reflection) == "det_negative"
    assert {group.component_id(x) for x in group.component_representatives()} == {
        "det_positive",
        "det_negative",
    }
```

- [ ] **Step 2: Run the red geometry-contract tests**

Run:

```powershell
python -m pytest tests\contracts\test_group_representation_contract.py tests\oracles\test_representation_laws.py --junitxml=artifacts\junit\phase4-representations-red.xml
```

Expected: nonzero exit code because the geometry package does not exist.

- [ ] **Step 3: Implement minimal group and representation protocols**

```python
class MatrixLieGroup(Protocol):
    dimension: int

    def identity(self) -> GroupElement: ...
    def element(self, matrix: torch.Tensor) -> GroupElement: ...
    def compose(self, left: GroupElement, right: GroupElement) -> GroupElement: ...
    def inverse(self, element: GroupElement) -> GroupElement: ...
    def component_id(self, element: GroupElement) -> str: ...
    def component_representatives(self) -> tuple[GroupElement, ...]: ...


class Representation(Protocol):
    domain: MatrixLieGroup
    codomain: VectorSpace

    def matrix(self, element: GroupElement) -> torch.Tensor: ...
```

`GeneralLinearGroup.element` validates shape, finiteness, and nonsingularity against a declared condition envelope. It records the positive- and negative-determinant components and supplies at least one certified representative of each. A separately registered `GLPositive` group restricts the domain to the identity component. The first implementation includes only the trivial and defining representations plus their duals. No string name implies a representation or fiber dimension.

- [ ] **Step 4: Run the green representation tests**

Run:

```powershell
python -m pytest tests\contracts\test_group_representation_contract.py tests\oracles\test_representation_laws.py --junitxml=artifacts\junit\phase4-representations.xml
```

Expected: exit code 0; group identity, inverse, composition, homomorphism, and dual-action checks pass in float64.

- [ ] **Step 5: Commit the typed geometry foundation**

```powershell
git add src\vfe4\geometry tests\contracts\test_group_representation_contract.py tests\oracles\test_representation_laws.py
git commit -m "feat: add groups representations and typed fibers"
```

### Task 4.2: Add principal bundles, associated bundles, morphisms, and flat transport

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\bundles.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\morphisms.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\transport.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_bundle_types.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_flat_transport_cocycle.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_typed_morphisms.py`

**Interfaces:**

- Consumes: group, representation, and fiber types.
- Produces: `PrincipalBundle`, `AssociatedBundle`, `FrameSection`, `FixedIntertwiner[Source, Target]`, `CovariantHomField[Source, Target]`, `FlatVertexConnection`, and `Transport[SourceFiber, TargetFiber]`.

- [ ] **Step 1: Write failing cocycle and rectangular-morphism tests**

```python
def test_flat_vertex_transport_telescopes() -> None:
    connection = flat_connection_fixture(length=3, dimension=2)
    omega_20 = connection.transport(source=0, target=2).matrix
    omega_21 = connection.transport(source=1, target=2).matrix
    omega_10 = connection.transport(source=0, target=1).matrix
    assert torch.allclose(omega_21 @ omega_10, omega_20, atol=1e-12, rtol=1e-12)


def test_model_to_state_morphism_allows_distinct_dimensions() -> None:
    morphism = CovariantHomField(source=model_fiber(3), target=state_fiber(2), matrix=torch.ones(2, 3))
    assert morphism.apply(torch.ones(3)).shape == (2,)


def test_fixed_map_must_satisfy_the_intertwiner_identity() -> None:
    with pytest.raises(IntertwinerCompatibilityError):
        FixedIntertwiner.from_matrix(
            source=nontrivial_source_representation(),
            target=nontrivial_target_representation(),
            matrix=nonintertwining_matrix(),
        )


def test_full_gl_intertwiner_is_checked_on_negative_determinant_component() -> None:
    candidate = map_that_passes_lie_generators_but_fails_reflection()
    with pytest.raises(IntertwinerCompatibilityError, match="det_negative"):
        FixedIntertwiner.from_matrix(
            source=full_gl_source_representation(),
            target=full_gl_target_representation(),
            matrix=candidate,
        )
```

- [ ] **Step 2: Run the red bundle and transport tests**

Run:

```powershell
python -m pytest tests\contracts\test_bundle_types.py tests\oracles\test_flat_transport_cocycle.py tests\oracles\test_typed_morphisms.py --junitxml=artifacts\junit\phase4-bundles-red.xml
```

Expected: nonzero exit code because bundle, morphism, and transport types do not exist.

- [ ] **Step 3: Implement deterministic Regime-I geometry**

`FlatVertexConnection` stores one frame element per vertex and returns

```python
omega_target_source = target_frame @ group.inverse(source_frame)
```

through the associated representation. The identity, inverse, endpoint, and cocycle laws are checked on construction for oracle-sized graphs. `FixedIntertwiner` validates the fixed equivariance identity on registered Lie-algebra generators and on every registered connected-component representative. Generator checks alone certify only the identity component and cannot certify full `GL(K)`. `CovariantHomField` is not called an intertwiner; under frame changes it transforms by the target action on the left and the inverse source action on the right. Both validate source and target dimensions. No edge-local curvature, BCH truncation, or latent frame distribution is added in this phase.

- [ ] **Step 4: Run the green bundle tests**

Run:

```powershell
python -m pytest tests\contracts\test_bundle_types.py tests\oracles\test_flat_transport_cocycle.py tests\oracles\test_typed_morphisms.py --junitxml=artifacts\junit\phase4-bundles.xml
```

Expected: exit code 0; cocycle and typed-dimension identities hold.

- [ ] **Step 5: Commit deterministic flat geometry**

```powershell
git add src\vfe4\geometry\bundles.py src\vfe4\geometry\morphisms.py src\vfe4\geometry\transport.py tests\contracts\test_bundle_types.py tests\oracles\test_flat_transport_cocycle.py tests\oracles\test_typed_morphisms.py
git commit -m "feat: add associated bundles and flat transport"
```

### Task 4.3: Make the complete causal objective gauge covariant

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\gauge.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\transition.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\emission.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\recognition.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_factor_gauge_covariance.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_complete_elbo_gauge_covariance.py`

**Interfaces:**

- Consumes: compiled causal model, recognition law, deterministic flat geometry, dual readouts, and controlled frame changes.
- Produces: `GaugeTransformation`, `pushforward_model`, `pushforward_recognition`, `GaugeResidualReport`, and `check_complete_gauge_covariance`.

- [ ] **Step 1: Write failing factor and complete-objective covariance tests**

```python
def test_complete_elbo_is_invariant_under_simultaneous_pushforward() -> None:
    model, recognition, observations = gauge_causal_fixture(dtype=torch.float64)
    transform = controlled_frame_change_fixture(max_norm=2.0, max_condition=8.0)
    original = evaluate_monolithic_elbo(model, recognition, observations)
    moved_model = pushforward_model(model, transform)
    moved_q = pushforward_recognition(recognition, transform)
    moved = evaluate_monolithic_elbo(moved_model, moved_q, observations)
    assert abs(float(moved.value - original.value)) <= 5e-11


def test_transported_prior_is_the_target_of_forward_kl() -> None:
    q_receiver, p_sender, transport = unequal_transport_fixture(dtype=torch.float64)
    transported_prior = transport.pushforward(p_sender)
    term = transported_transition_complexity(q_receiver, p_sender, transport)
    assert torch.allclose(term, KLDivergence().evaluate(q_receiver, transported_prior))
    assert not torch.allclose(term, KLDivergence().evaluate(transported_prior, q_receiver))
```

The factor test separately checks covariance congruence, precision contragredience, receiver-measure Jacobians, rectangular morphisms, and invariance of centered categorical logits.

- [ ] **Step 2: Run the red gauge-covariance tests**

Run:

```powershell
python -m pytest tests\oracles\test_factor_gauge_covariance.py tests\integration\test_complete_elbo_gauge_covariance.py --junitxml=artifacts\junit\phase4-gauge-red.xml
```

Expected: nonzero exit code because simultaneous model and recognition pushforward is absent.

- [ ] **Step 3: Implement complete pushforward with Jacobian accounting**

Transform means, covariances, precisions, offsets, transports, morphisms, and decoder readouts. Transition complexity is the forward divergence from the receiver recognition law to the transported sender prior, `KL(q_receiver || transport_* p_sender)`; argument reversal is rejected by the oriented divergence type and asymmetric oracle fixture. Continuous coordinate densities acquire inverse absolute-Jacobian factors; categorical sources remain unchanged. `GaugeResidualReport` records absolute ELBO residual, relative residual, round-trip backward residual, frame norms, inverse-frame norms, operand condition estimates, and the oracle tolerance.

Holding decoder weights fixed is allowed only through a separately typed `EmissionStabilizerCheck`; it is never treated as full local gauge covariance.

- [ ] **Step 4: Run the complete Phase 4 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles tests\integration --junitxml=artifacts\junit\phase4-gate.xml
```

Expected: exit code 0; inspect the XML for zero failures and errors. The complete objective, not only transitions, passes the bounded float64 gauge test.

- [ ] **Step 5: Commit complete gauge covariance**

```powershell
git add src\vfe4\geometry\gauge.py src\vfe4\factors\transition.py src\vfe4\factors\emission.py src\vfe4\inference\recognition.py tests\oracles\test_factor_gauge_covariance.py tests\integration\test_complete_elbo_gauge_covariance.py
git commit -m "feat: enforce complete causal gauge covariance"
```

## Phase 5: Optional h-s-p-q Hierarchy Profiles

### Task 5.1: Compile hierarchy profiles without turning q into a generative variable

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\profiles.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\hierarchy.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\compose.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_hierarchy_profiles.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_profile_composition.py`

**Interfaces:**

- Consumes: graph compiler, normalized factor specifications, frozen catalogs, and recognition types.
- Produces: `HierarchyProfile`, built-ins `PQ_STATE`, `HSPQ_FIXED_MODEL_BOUNDARY`, and `HSPQ_LATENT_MODEL`, `GenerativeHierarchy`, `RecognitionProfile`, and `compose_runtime(profile, catalogs, backend)`.

- [ ] **Step 1: Write failing profile-inventory and q-role tests**

```python
def test_hspq_alias_expands_only_to_event_variables_kernels_and_recognition_factors() -> None:
    runtime = compose_test_runtime(profile=HSPQ_LATENT_MODEL)
    assert runtime.graph.semantic_node_roles() == ("state:z", "model:m", "observation:x")
    assert runtime.graph.has_normalized_kernel_roles("state:Kz", "model:Km", "model_to_state")
    assert runtime.recognition.factor_roles() == ("state:qz", "model:qm")
    assert not runtime.graph_or_recognition.has_semantic_ids("h", "s", "p", "q")
    assert runtime.graph.factor_inventory() == HSPQ_LATENT_MODEL.expected_factor_inventory


def test_disabled_profile_allocates_no_hierarchy_parameters() -> None:
    runtime = compose_test_runtime(profile=PQ_STATE)
    assert runtime.parameter_owners.with_prefix("hierarchy.") == ()
```

- [ ] **Step 2: Run the red hierarchy-profile tests**

Run:

```powershell
python -m pytest tests\contracts\test_hierarchy_profiles.py tests\integration\test_profile_composition.py --junitxml=artifacts\junit\phase5-profiles-red.xml
```

Expected: nonzero exit code because hierarchy profiles and the composition root do not exist.

- [ ] **Step 3: Implement explicit generative and recognition profile halves**

`pq` and `hspq` are composition aliases only. `PQ_STATE` emits state variables \(z\), normalized state kernels \(K^z\), and recognition conditionals \(q^{z,(r)}\). `HSPQ_LATENT_MODEL` additionally emits model variables \(m\), normalized model kernels \(K^m\), model recognition conditionals \(q^{m,(r)}\), and a normalized model-to-state kernel. The labels `h`, `s`, `p`, and `q` never become semantic node IDs, tensor classes, or factor kinds.

A fixed model boundary and a random model or hyperstate are different probability models. `HSPQ_FIXED_MODEL_BOUNDARY` treats the boundary as deterministic conditioned data and introduces no model entropy. `HSPQ_LATENT_MODEL` introduces a normalized generative law and corresponding recognition factor. They have separate factor inventories and cannot be selected by a runtime flag inside one compiled graph.

`compose_runtime` resolves every catalog entry once, freezes the catalogs, validates the graph, instantiates the selected backend, and returns typed runtime services. Call sites receive objects and contain no profile-name branch.

- [ ] **Step 4: Run green profile tests**

Run:

```powershell
python -m pytest tests\contracts\test_hierarchy_profiles.py tests\integration\test_profile_composition.py --junitxml=artifacts\junit\phase5-profiles.xml
```

Expected: exit code 0; disabled profiles create no dead parameters or inert factor records.

- [ ] **Step 5: Commit hierarchy composition**

```powershell
git add src\vfe4\graph\profiles.py src\vfe4\factors\hierarchy.py src\vfe4\runtime\compose.py tests\contracts\test_hierarchy_profiles.py tests\integration\test_profile_composition.py
git commit -m "feat: add typed optional hierarchy profiles"
```

### Task 5.2: Gate hierarchy profiles on normalization and objective-ledger identity

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_hierarchy_normalization.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_hierarchy_elbo_inventory.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_profile_dense_backend_identity.py`

**Interfaces:**

- Consumes: all hierarchy profiles, the semantically exact tiny-language reference, dense backend, and ELBO decomposition.
- Produces: no new public runtime API; this task establishes the promotion gate for optional profiles.

- [ ] **Step 1: Write failing profile-wide normalization and ledger tests**

```python
@pytest.mark.parametrize("profile", [PQ_STATE, HSPQ_FIXED_MODEL_BOUNDARY, HSPQ_LATENT_MODEL])
def test_profile_joint_normalizes_and_ledgers_match(profile: HierarchyProfile) -> None:
    runtime = compose_tiny_runtime(profile=profile, backend="dense_f64")
    assert runtime.enumerate_joint_normalization_residual() <= 1e-12
    monolithic = runtime.evaluate_monolithic_elbo()
    local = runtime.evaluate_local_elbo()
    assert monolithic.inventory.identity_hash == local.inventory.identity_hash
    assert abs(float(monolithic.value - local.value)) <= 1e-11
```

- [ ] **Step 2: Run the red hierarchy gate**

Run:

```powershell
python -m pytest tests\oracles\test_hierarchy_normalization.py tests\integration\test_hierarchy_elbo_inventory.py tests\integration\test_profile_dense_backend_identity.py --junitxml=artifacts\junit\phase5-hierarchy-gate-red.xml
```

Expected: nonzero exit code until every profile supplies a complete normalized factor inventory.

- [ ] **Step 3: Repair only missing profile declarations exposed by the red tests**

Add no free-floating hierarchy penalty. Every local term must arise from `log p - log q` for a declared normalized factor or recognition conditional. Profile-specific parameter ownership is included in the catalog manifest and objective inventory hash.

- [ ] **Step 4: Run the complete Phase 5 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles tests\integration --junitxml=artifacts\junit\phase5-gate.xml
```

Expected: exit code 0; inspect the XML for zero failures and errors before enabling mixture or representation-decomposition work.

- [ ] **Step 5: Commit the hierarchy gate**

```powershell
git add tests\oracles\test_hierarchy_normalization.py tests\integration\test_hierarchy_elbo_inventory.py tests\integration\test_profile_dense_backend_identity.py src\vfe4\graph\profiles.py src\vfe4\factors\hierarchy.py
git commit -m "test: gate hierarchy profiles on normalized ELBO identity"
```

## Phase 6: Labeled mixtures, source priors, representations, and typed divergences

### Task 6.1: Distinguish labeled mixtures from marginalized mixtures

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\mixture.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_mixture_types.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_labeled_mixture_entropy.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_marginal_mixture_entropy.py`

**Interfaces:**

- Consumes: normalized laws, finite counting measures, continuous base measures, and `ProjectedLaw`.
- Produces: `LabeledMixture`, `MarginalMixture`, `MixtureLabel`, `IntegralEvaluation[T]`, and explicit `marginalize_labels` and `moment_project` operations.

- [ ] **Step 1: Write failing type, normalization, and entropy tests**

```python
def test_labeled_and_marginal_mixtures_are_distinct_laws() -> None:
    labeled = make_two_component_labeled_gaussian()
    marginal = labeled.marginalize_labels()
    assert isinstance(labeled, LabeledMixture)
    assert isinstance(marginal, MarginalMixture)
    assert labeled.space != marginal.space
    assert abs(labeled.normalization_residual()) <= 1e-12
    assert abs(marginal.normalization_residual()) <= 1e-12


def test_overlap_breaks_the_labeled_entropy_formula_after_marginalization() -> None:
    labeled = make_overlapping_labeled_gaussian()
    marginal = labeled.marginalize_labels()
    labeled_formula = labeled.weight_entropy() + labeled.weighted_component_entropy()
    assert abs(float(labeled.entropy() - labeled_formula)) <= 1e-12
    marginal_entropy = marginal.entropy(evaluator=calibrated_mixture_quadrature())
    assert marginal_entropy.status is EvaluationStatus.DETERMINISTIC_QUADRATURE
    assert marginal_entropy.evaluator_id == "calibrated_mixture_quadrature"
    assert marginal_entropy.error_budget is not None
    assert marginal_entropy.seed is None
    assert abs(float(marginal_entropy.value - labeled_formula)) >= 1e-5
```

- [ ] **Step 2: Run the red mixture tests**

Run:

```powershell
python -m pytest tests\contracts\test_mixture_types.py tests\oracles\test_labeled_mixture_entropy.py tests\oracles\test_marginal_mixture_entropy.py --junitxml=artifacts\junit\phase6-mixtures-red.xml
```

Expected: nonzero exit code because the two mixture laws and their measure spaces are not implemented.

- [ ] **Step 3: Implement measure-aware mixture laws**

`LabeledMixture` is a normalized law on the product of a finite label space and a component event space. Its entropy is exactly the categorical entropy plus the weight-averaged component entropy. `MarginalMixture` is the label-marginalized law on the component event space. Its pointwise density contains an exact finite label sum, but its continuous entropy is not thereby a finite exact oracle. `entropy(evaluator=...)` returns an `IntegralEvaluation` containing value, evaluator identity, evaluation status, residual or error budget, and seed where applicable. A bare tensor-returning entropy API is forbidden for this nonanalytic case. `moment_project` returns `ProjectedLaw` and cannot silently replace either mixture law.

- [ ] **Step 4: Run green mixture tests**

Run:

```powershell
python -m pytest tests\contracts\test_mixture_types.py tests\oracles\test_labeled_mixture_entropy.py tests\oracles\test_marginal_mixture_entropy.py --junitxml=artifacts\junit\phase6-mixtures.xml
```

Expected: exit code 0; both laws normalize, their types remain distinct, and the entropy identity is used only for the labeled law.

- [ ] **Step 5: Commit mixture semantics**

```powershell
git add src\vfe4\families\mixture.py tests\contracts\test_mixture_types.py tests\oracles\test_labeled_mixture_entropy.py tests\oracles\test_marginal_mixture_entropy.py
git commit -m "feat: separate labeled and marginal mixture laws"
```

### Task 6.2: Add target-blind source priors and target-aware source recognition

**Files:**

- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\factors\source.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\source_priors.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\source_posteriors.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_source_prior_protocol.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_source_prior_normalization.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_source_no_leakage.py`

**Interfaces:**

- Consumes: causal masks, typed dependency declarations, position indices, frozen catalogs, and recognition contexts.
- Produces: `SourcePrior`, built-ins `UniformCausalPrior`, `LearnedPositionPrior`, `AlibiSourcePrior`, `T5BucketSourcePrior`, and `GeometryInvariantPrior`, plus the separate `SourcePosterior` protocol.

- [ ] **Step 1: Write failing normalization and dependency-audit tests**

```python
@pytest.mark.parametrize("name", ["uniform", "learned_position", "alibi", "t5_bucket", "geometry_invariant"])
def test_source_prior_is_normalized_over_allowed_sources(name: str) -> None:
    prior = source_prior_catalog.resolve(name)
    log_probs = prior.log_probs(context=target_blind_context(), allowed=causal_mask())
    assert torch.allclose(log_probs.logsumexp(dim=-1), torch.zeros_like(log_probs[..., 0]))


def test_generative_source_prior_rejects_target_tokens() -> None:
    with pytest.raises(TargetLeakageError):
        source_prior_catalog.resolve("learned_position").log_probs(
            context=context_with_current_target(), allowed=causal_mask()
        )


def test_source_posterior_support_must_be_contained_in_prior_support() -> None:
    with pytest.raises(SourceSupportMismatchError):
        compile_source_pair(
            prior=source_prior_with_support({0, 1}),
            posterior=source_posterior_with_support({0, 1, 2}),
        )
```

- [ ] **Step 2: Run the red source-prior tests**

Run:

```powershell
python -m pytest tests\contracts\test_source_prior_protocol.py tests\oracles\test_source_prior_normalization.py tests\integration\test_source_no_leakage.py --junitxml=artifacts\junit\phase6-source-red.xml
```

Expected: nonzero exit code because no typed source-prior seam exists.

- [ ] **Step 3: Implement source priors as normalized generative factors**

Apply the causal mask before normalization. Each prior declares its dependencies, parameter owner, support, log normalizer, sampling operation, and whether its probabilities are exact or estimated. Position-only priors may depend on source and destination positions but not the current or future token. `GeometryInvariantPrior` may use deterministic geometry, transports, and invariant edge data only, and it must normalize over the positive causal support. RoPE remains a representation or geometry transform and is not registered as a source prior. A content-conditioned generative source distribution is a future extension requiring a separately named, target-blind implementation and a fresh normalization audit. Observation-conditioned source recognition is registered only in `SourcePosterior` and cannot be passed where `SourcePrior` is required.

- [ ] **Step 4: Run green source-prior tests**

Run:

```powershell
python -m pytest tests\contracts\test_source_prior_protocol.py tests\oracles\test_source_prior_normalization.py tests\integration\test_source_no_leakage.py --junitxml=artifacts\junit\phase6-source.xml
```

Expected: exit code 0; every row normalizes over its causal support and dependency audits reject target leakage.

- [ ] **Step 5: Commit source priors**

```powershell
git add src\vfe4\factors\source.py src\vfe4\language\source_priors.py src\vfe4\inference\source_posteriors.py tests\contracts\test_source_prior_protocol.py tests\oracles\test_source_prior_normalization.py tests\integration\test_source_no_leakage.py
git commit -m "feat: add normalized causal source priors"
```

### Task 6.3: Compile irreducible-representation layouts and commutants

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\irreps.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\commutants.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\geometry\layouts.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\irrep_decomposed.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_irrep_layouts.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_commutant_basis.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_irrep_backend_identity.py`

**Interfaces:**

- Consumes: group representations, decomposition certificates, parameter spaces, and exact dense operators.
- Produces: `ScalarField`, `RealIrrepType`, `IrrepLabel`, `IsotypicBlock`, `IrrepLayout`, `CommutantBasis`, `ClosureCertificate`, and `IrrepDecomposedBackend`.

- [ ] **Step 1: Write failing closure and dense-identity tests**

```python
def test_equivalent_irreps_share_a_multiplicity_block() -> None:
    layout = make_two_copy_standard_representation_layout()
    assert layout.isotypic_blocks[0].multiplicity == 2
    assert layout.allows_multiplicity_coupling()


def test_irrep_backend_matches_dense_oracle() -> None:
    problem = make_commutant_closed_gaussian_problem()
    dense = DenseF64Oracle().solve(problem)
    decomposed = IrrepDecomposedBackend(problem.layout).solve(problem)
    assert decomposed.closure_certificate.is_exact
    assert_close_solution(decomposed, dense, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize(
    ("real_type", "division_algebra_dimension"),
    [(RealIrrepType.REAL, 1), (RealIrrepType.COMPLEX, 2), (RealIrrepType.QUATERNIONIC, 4)],
)
def test_real_irrep_type_selects_the_qualified_commutant(
    real_type: RealIrrepType,
    division_algebra_dimension: int,
) -> None:
    basis = commutant_basis(single_real_irrep_fixture(real_type))
    assert basis.division_algebra_dimension == division_algebra_dimension
```

- [ ] **Step 2: Run the red representation-layout tests**

Run:

```powershell
python -m pytest tests\contracts\test_irrep_layouts.py tests\oracles\test_commutant_basis.py tests\integration\test_irrep_backend_identity.py --junitxml=artifacts\junit\phase6-irreps-red.xml
```

Expected: nonzero exit code because decomposition layouts and closure certificates do not exist.

- [ ] **Step 3: Implement isotypic rather than scalar-block decomposition**

Group equivalent irreducible representations into an isotypic block and retain arbitrary operators on its multiplicity space. Record the scalar field and, for real representations, the real, complex, or quaternionic irrep type; do not assume the complex Schur form for every carrier. Build the qualified commutant basis from the declared representation, validate every precision and factor operator against that basis, and emit a `ClosureCertificate` before dispatching to the decomposed backend. A failed certificate is a hard error unless the caller explicitly requests a recorded projection.

- [ ] **Step 4: Run green representation-layout tests**

Run:

```powershell
python -m pytest tests\contracts\test_irrep_layouts.py tests\oracles\test_commutant_basis.py tests\integration\test_irrep_backend_identity.py --junitxml=artifacts\junit\phase6-irreps.xml
```

Expected: exit code 0; commutant-closed models match the dense oracle and non-closed models fail before solving.

- [ ] **Step 5: Commit representation layouts**

```powershell
git add src\vfe4\geometry\irreps.py src\vfe4\geometry\commutants.py src\vfe4\geometry\layouts.py src\vfe4\numerics\irrep_decomposed.py tests\contracts\test_irrep_layouts.py tests\oracles\test_commutant_basis.py tests\integration\test_irrep_backend_identity.py
git commit -m "feat: add exact isotypic block layouts"
```

### Task 6.4: Give each divergence family a separate type and parameter

**Files:**

- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\divergences.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\families\divergence_config.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\objectives\divergence_roles.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_divergence_types.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_divergence_limits.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_divergence_properties.py`

**Interfaces:**

- Consumes: normalized laws, support checks, exact categorical and Gaussian oracles, and family-specific configuration.
- Produces: `KLDivergence`, `RenyiDivergence(order)`, `AmariAlphaDivergence(alpha)`, `RenyiOrder`, `AmariAlpha`, and role wrappers `EvidenceRatio`, `GeneralizedBayesLoss`, `ProjectionDivergence`, `ConsensusPenalty`, and `DiagnosticDivergence`.

- [ ] **Step 1: Write failing self-divergence, non-negativity, and limit tests**

```python
@pytest.mark.parametrize(
    "divergence",
    [
        KLDivergence(),
        RenyiDivergence(order=RenyiOrder(0.7)),
        AmariAlphaDivergence(alpha=AmariAlpha(-0.3)),
    ],
)
def test_divergence_is_zero_on_identical_categorical_laws(divergence: Divergence) -> None:
    p = categorical_law([0.2, 0.3, 0.5])
    assert abs(float(divergence.evaluate(p, p))) <= 1e-12


@pytest.mark.parametrize(
    "divergence",
    [
        KLDivergence(),
        RenyiDivergence(order=RenyiOrder(0.7)),
        AmariAlphaDivergence(alpha=AmariAlpha(-0.3)),
    ],
)
def test_divergence_is_nonnegative_on_unequal_full_support_laws(divergence: Divergence) -> None:
    p, q = unequal_full_support_categorical_laws()
    assert float(divergence.evaluate(p, q)) >= -1e-13


def test_family_specific_unit_limits_recover_forward_kl() -> None:
    p, q = unequal_full_support_categorical_laws()
    kl = KLDivergence().evaluate(p, q)
    assert RenyiDivergence(order=RenyiOrder(1.0)).evaluate(p, q) == kl
    assert AmariAlphaDivergence(alpha=AmariAlpha(-1.0)).evaluate(p, q) == kl
    reverse_kl = KLDivergence().evaluate(q, p)
    assert AmariAlphaDivergence(alpha=AmariAlpha(1.0)).evaluate(p, q) == reverse_kl
```

- [ ] **Step 2: Run the red divergence-family tests**

Run:

```powershell
python -m pytest tests\contracts\test_divergence_types.py tests\oracles\test_divergence_limits.py tests\oracles\test_divergence_properties.py --junitxml=artifacts\junit\phase6-divergences-red.xml
```

Expected: nonzero exit code until separately typed divergence families replace any overloaded scalar selector.

- [ ] **Step 3: Implement family-specific formulas and exact limit branches**

Use `RenyiOrder` only for Rényi divergence and `AmariAlpha` only for Amari's alpha-divergence. State the orientation in the API: Rényi order one and Amari alpha minus one recover forward `KL(source || target)`, while Amari alpha plus one recovers the reverse orientation. Validate supports before evaluation. Implement the defining formulas from normalized densities, use exact branches at the KL limits, and test nearby parameter values against high-precision references. `EvidenceRatio` accepts only the log-density ratio induced by declared normalized model and recognition laws and integrates to ordinary KL with coefficient one. Alternative divergences enter only through `GeneralizedBayesLoss`, `ProjectionDivergence`, `ConsensusPenalty`, or `DiagnosticDivergence`; none can preserve the ordinary evidence-ELBO contract.

- [ ] **Step 4: Run green divergence-family tests**

Run:

```powershell
python -m pytest tests\contracts\test_divergence_types.py tests\oracles\test_divergence_limits.py tests\oracles\test_divergence_properties.py --junitxml=artifacts\junit\phase6-divergences.xml
```

Expected: exit code 0; every checked divergence has zero self-divergence, nonnegative values within the declared numerical tolerance, and the correct KL limit.

- [ ] **Step 5: Commit typed divergence families**

```powershell
git add src\vfe4\families\divergences.py src\vfe4\families\divergence_config.py src\vfe4\objectives\divergence_roles.py tests\contracts\test_divergence_types.py tests\oracles\test_divergence_limits.py tests\oracles\test_divergence_properties.py
git commit -m "feat: separate divergence families and parameters"
```

### Task 6.5: Gate Phase 6 on semantic and backend identities

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\inference\information_projection.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_mixture_projection_ledger.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_source_profile_inventory.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_representation_layout_profiles.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_divergence_objective_contracts.py`

**Interfaces:**

- Consumes: Phase 6 laws, projection records, source profiles, representation layouts, divergence types, objective contracts, and all exact backends.
- Produces: `InformationProjection` and the complete Phase 6 promotion gate.

- [ ] **Step 1: Write failing cross-seam tests**

Test that a labeled-to-marginal projection changes the law type and emits an approximation-ledger record, that each enabled source prior adds exactly one normalized factor to the inventory, that every admitted representation layout carries a closure certificate, and that a generalized divergence cannot be passed to the ordinary ELBO constructor.

- [ ] **Step 2: Run the red Phase 6 integration gate**

Run:

```powershell
python -m pytest tests\integration\test_mixture_projection_ledger.py tests\integration\test_source_profile_inventory.py tests\integration\test_representation_layout_profiles.py tests\integration\test_divergence_objective_contracts.py --junitxml=artifacts\junit\phase6-integration-red.xml
```

Expected: nonzero exit code until every semantic conversion and structural assumption is represented in types and ledgers.

- [ ] **Step 3: Implement only the explicit projection and validation adapters**

`InformationProjection` records the source family, destination family, projection direction, sufficient statistics preserved, solver, tolerance, and residual. It never changes the source object in place. Wire source profiles, representation layouts, and objective contracts through catalog composition without adding call-site mode branches.

- [ ] **Step 4: Run the complete Phase 6 gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles tests\integration --junitxml=artifacts\junit\phase6-gate.xml
```

Expected: exit code 0; inspect the XML for zero failures and errors before beginning CUDA and artifact work.

- [ ] **Step 5: Commit the Phase 6 gate**

```powershell
git add src\vfe4\inference\information_projection.py tests\integration\test_mixture_projection_ledger.py tests\integration\test_source_profile_inventory.py tests\integration\test_representation_layout_profiles.py tests\integration\test_divergence_objective_contracts.py
git commit -m "test: gate mixture and representation semantics"
```

## Phase 7: Production CUDA, durable artifacts, migration, and matched experiments

### Task 7.1: Resolve configuration and freeze runtime-state boundaries

**Files:**

- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\schema.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\codec.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\migrations.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\config\resolved.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\graph\state_layout.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\state.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_config_codec.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_runtime_state_boundaries.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_compiled_signature_purity.py`

**Interfaces:**

- Consumes: component descriptors, graph compilation results, exactness records, and semantic node identifiers.
- Produces: frozen `ExperimentSpec`, `ModelSpec`, `RecognitionSpec`, `ObjectiveSpec`, `InferenceSpec`, `NumericsSpec`, `TrainingSpec`, `DataSpec`, `ArtifactSpec`, typed `ComponentRef`, `CanonicalSpec`, `ResolvedSpec`, `StateLayout`, `RunState`, `BatchState`, and `Workspace`.

- [ ] **Step 1: Extend the codec tests with failing resolution and boundary tests**

```python
def test_resolved_spec_records_every_compatibility_predicate() -> None:
    program = compile_tiny_program(valid_tiny_spec())
    assert program.resolved_spec.compatibility_records
    assert all(record.predicate_id and record.inputs_digest for record in program.resolved_spec.compatibility_records)


def test_compiled_update_signature_has_no_config_or_catalog() -> None:
    program = compile_tiny_program(valid_tiny_spec())
    parameters = inspect.signature(program.update_plan.steps[0]).parameters
    assert "config" not in parameters
    assert "catalog" not in parameters
```

- [ ] **Step 2: Run the red configuration tests**

Run:

```powershell
python -m pytest tests\contracts\test_config_codec.py tests\contracts\test_runtime_state_boundaries.py tests\integration\test_compiled_signature_purity.py --junitxml=artifacts\junit\phase7-config-red.xml
```

Expected: nonzero exit code because final resolved compatibility records and runtime-state boundaries do not exist.

- [ ] **Step 3: Implement immutable configuration and separated mutable state**

The authored spec preserves exactly what the experiment declares. Pure sequential schema migrations produce a canonical spec with explicit defaults. Compilation produces a resolved spec containing descriptor versions, compatibility results, dimensions, layouts, bound kernels, and exactness. Unknown fields, ambiguous booleans, incompatible references, and newer unsupported schemas fail before tensor allocation. `RunState` owns persistent trajectory-bearing state, `BatchState` owns observation-conditioned local variables, and `Workspace` owns disposable backend scratch. `StateLayout` maps semantic identifiers bidirectionally to packed tensor paths.

- [ ] **Step 4: Run green configuration tests**

Run:

```powershell
python -m pytest tests\contracts\test_config_codec.py tests\contracts\test_runtime_state_boundaries.py tests\integration\test_compiled_signature_purity.py --junitxml=artifacts\junit\phase7-config.xml
```

Expected: exit code 0; strict decoding, pure migrations, stable semantic layouts, and already-bound runtime signatures are enforced.

- [ ] **Step 5: Commit configuration and state boundaries**

```powershell
git add src\vfe4\config src\vfe4\graph\state_layout.py src\vfe4\runtime\state.py tests\contracts\test_config_codec.py tests\contracts\test_runtime_state_boundaries.py tests\integration\test_compiled_signature_purity.py
git commit -m "feat: add strict resolved configuration contracts"
```

### Task 7.2: Extend the verified CUDA path with sparse and matrix-free operations

**Files:**

- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\cuda.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\torch_dense.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\torch_banded.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\matrix_free.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\device.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_cuda_policy.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\oracles\test_float32_tolerance_registry.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_cuda_oracle_agreement.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_matrix_free_ledger.py`

**Interfaces:**

- Consumes: compiled precision operators, state layouts, backend capability predicates, operation-specific tolerance records, and the float64 oracle.
- Produces: `CudaFloat32Policy`, `TorchDenseBackend`, `TorchBlockBandedBackend`, `CudaSparseOperator`, `MatrixFreeCudaBackend`, and `NumericalTolerance`.

- [ ] **Step 1: Write failing dtype, oracle-agreement, and approximation tests**

```python
@pytest.mark.cuda
def test_cuda_banded_backend_matches_float64_oracle() -> None:
    problem = make_tiny_banded_problem()
    reference = DenseF64Oracle().solve(problem.to(device="cpu", dtype=torch.float64))
    actual = TorchBlockBandedBackend().solve(problem.to(device="cuda", dtype=torch.float32))
    assert_operation_close("banded_solve", actual.mean.cpu(), reference.mean)
    assert actual.execution.objective_evaluation is None
    assert actual.execution.linear_algebra is LinearAlgebraStatus.STRUCTURE_EQUIVALENT_DIRECT
    assert actual.execution.arithmetic.dtype == "float32"


@pytest.mark.cuda
def test_matrix_free_result_records_residual_and_estimator_budget() -> None:
    result = MatrixFreeCudaBackend(cg_tolerance=1e-6, max_iterations=200).solve(
        make_tiny_sparse_problem(device="cuda")
    )
    assert result.execution.objective_evaluation is None
    assert result.execution.linear_algebra is LinearAlgebraStatus.ITERATIVE_CERTIFIED
    assert result.approximation.residual is not None
    assert result.approximation.iteration_limit == 200
```

- [ ] **Step 2: Run the red CUDA backend tests on the RTX 5090**

Run:

```powershell
python -m pytest tests\contracts\test_cuda_policy.py tests\oracles\test_float32_tolerance_registry.py tests\cuda\test_cuda_oracle_agreement.py tests\cuda\test_matrix_free_ledger.py --junitxml=artifacts\junit\phase7-cuda-red.xml
```

Expected: nonzero exit code because the sparse operator and matrix-free extensions do not exist; the already promoted dense and block-banded path remains green.

- [ ] **Step 3: Implement direct structured and declared matrix-free CUDA paths**

Retain the already verified CUDA float32 dense and block-banded kernels and their no-autocast, no-TF32 SPD policy. CUDA sparse lowering supplies explicit operators; conjugate-gradient solves or stochastic log determinants are matrix-free approximations with tolerance, iteration, residual, seed, probe count, and estimator uncertainty in the ledger. Objective evaluation, linear algebra, and arithmetic remain separate records. Backend choice is bound at compilation, and the new paths cannot replace the earlier direct paths under the same descriptor.

- [ ] **Step 4: Run green CUDA backend tests**

Run:

```powershell
python -m pytest tests\contracts\test_cuda_policy.py tests\oracles\test_float32_tolerance_registry.py tests\cuda\test_cuda_oracle_agreement.py tests\cuda\test_matrix_free_ledger.py --junitxml=artifacts\junit\phase7-cuda.xml
```

Expected: exit code 0 on the required GPU; each tested production kernel meets its named float32 tolerance and reports separate objective-evaluation, linear-algebra, arithmetic, family, and optimization records.

- [ ] **Step 5: Commit CUDA backends**

```powershell
git add src\vfe4\numerics\cuda.py src\vfe4\numerics\torch_dense.py src\vfe4\numerics\torch_banded.py src\vfe4\numerics\matrix_free.py src\vfe4\runtime\device.py tests\contracts\test_cuda_policy.py tests\oracles\test_float32_tolerance_registry.py tests\cuda\test_cuda_oracle_agreement.py tests\cuda\test_matrix_free_ledger.py
git commit -m "feat: add explicit float32 CUDA backends"
```

### Task 7.3: Gate the RTX 5090 structured scale path

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\numerics\allocation_trace.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\experiments\hardware.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_rtx5090_identity.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_rtx5090_structured_scale.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_cuda_no_global_dense_allocation.py`

**Interfaces:**

- Consumes: compiled block-banded and sparse layouts, CUDA device properties, allocation tracing, and the production device policy.
- Produces: `HardwareIdentity`, semantic `AllocationTrace`, `CudaAllocatorEvidence`, and the mandatory `T=128`, `K=20` RTX 5090 scale gate.

- [ ] **Step 1: Write failing device and memory-residency tests**

```python
@pytest.mark.cuda
def test_required_device_is_rtx5090() -> None:
    identity = HardwareIdentity.capture()
    assert identity.cuda_available
    assert identity.device_name == "NVIDIA GeForce RTX 5090"


@pytest.mark.cuda
def test_t128_k20_banded_operations_allocate_no_quadratic_global_buffer() -> None:
    cell = compile_scale_cell(sequence_length=128, fiber_dimension=20, backend="cuda_banded_f32")
    with AllocationTrace() as trace, CudaAllocatorEvidence() as allocator:
        factorization = cell.assemble_and_factorize(one_scale_batch())
        solve = factorization.solve(cell.rhs())
        logdet = factorization.logdet()
        marginals = factorization.requested_marginals(cell.moment_requests())
        sample = factorization.sample(generator=cell.generator())
        loss = cell.objective(solve, logdet, marginals, sample)
        loss.backward()
        diagnostics = cell.diagnostics(factorization)
        report = cell.report(loss, diagnostics)
    assert report.is_finite
    assert trace.operation_coverage == {
        "construction", "factorization", "solve", "logdet", "marginals",
        "sampling", "backward", "diagnostics", "reporting",
    }
    assert not trace.contains_global_quadratic_extent(total_dimension=128 * 20)
    assert not cell.compiled_plan.exposes_global_materializer
    assert allocator.peak_allocated_bytes <= cell.predeclared_peak_byte_ceiling
    assert trace.all_tensor_devices == {torch.device("cuda:0")}
    assert trace.all_floating_dtypes == {torch.float32}
```

- [ ] **Step 2: Run the red RTX scale tests**

Run:

```powershell
python -m pytest tests\cuda\test_rtx5090_identity.py tests\cuda\test_rtx5090_structured_scale.py tests\cuda\test_cuda_no_global_dense_allocation.py --junitxml=artifacts\junit\phase7-rtx5090-red.xml
```

Expected: nonzero exit code until device identity and allocation contracts are implemented.

- [ ] **Step 3: Implement auditable allocation and residency checks**

Record logical allocation shapes, storage bytes, dtype, device, semantic owner, and operation phase through semantic constructors and materializer guards. Pair that record with machine-readable CUDA allocator or profiler evidence and an asserted peak-byte ceiling derived before the run from the compiled layout. Exercise construction, factorization, solve, log determinant, requested marginals, sampling, backward, diagnostics, and reporting separately. Reject any global \((TK)\times(TK)\) precision, covariance, inverse, workspace, or equivalent quadratic buffer, including a hidden diagnostic or compatibility materializer. A skipped CUDA test is not a successful RTX promotion gate.

- [ ] **Step 4: Run the green RTX scale gate**

Run:

```powershell
python -m pytest tests\cuda\test_rtx5090_identity.py tests\cuda\test_rtx5090_structured_scale.py tests\cuda\test_cuda_no_global_dense_allocation.py --junitxml=artifacts\junit\phase7-rtx5090.xml
```

Expected: exit code 0 on the RTX 5090; inspect the XML for zero skipped tests, zero failures, and zero errors.

- [ ] **Step 5: Commit the hardware gate**

```powershell
git add src\vfe4\numerics\allocation_trace.py src\vfe4\experiments\hardware.py tests\cuda\test_rtx5090_identity.py tests\cuda\test_rtx5090_structured_scale.py tests\cuda\test_cuda_no_global_dense_allocation.py
git commit -m "test: gate RTX 5090 structured execution"
```

### Task 7.4: Emit canonical manifests and immutable resumable checkpoints

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\schema.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\atomic.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\fingerprints.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\manifest.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\checkpoint.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\events.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\migrations\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_manifest_schema.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_checkpoint_container.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_checkpoint_resume_modes.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_checkpoint_validation_failures.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_checkpoint_schema_migrations.py`

**Interfaces:**

- Consumes: stable artifact data-transfer objects, authored/canonical/resolved specs, graph and layout digests, catalog manifest, exactness report, ledgers, runtime state, data identity, and hardware identity.
- Produces: `BuildManifest`, `RunManifest`, `CheckpointManifest`, `CheckpointWriter`, `CheckpointLoader`, `ResumeMode`, append-only event JSON Lines, and pure metadata migrations.

- [ ] **Step 1: Write failing manifest, corruption, and continuation tests**

```python
def test_exact_continuation_restores_trajectory_state(tmp_path: Path) -> None:
    original = make_advanced_tiny_run_state()
    checkpoint = save_checkpoint(tmp_path, original)
    restored = load_checkpoint(checkpoint, mode=ResumeMode.EXACT_CONTINUATION)
    assert_state_identity(restored, original)
    assert next_training_step(restored) == next_training_step(original)


def test_hash_corruption_fails_before_live_state_mutation(tmp_path: Path) -> None:
    checkpoint = save_checkpoint(tmp_path, make_tiny_run_state())
    corrupt_one_tensor_byte(checkpoint / "tensors.safetensors")
    target = make_sentinel_run_state()
    with pytest.raises(ArtifactHashError):
        load_into(checkpoint, target, mode=ResumeMode.EXACT_CONTINUATION)
    assert target == make_sentinel_run_state()
```

- [ ] **Step 2: Run the red artifact tests**

Run:

```powershell
python -m pytest tests\contracts\test_manifest_schema.py tests\contracts\test_checkpoint_container.py tests\integration\test_checkpoint_resume_modes.py tests\integration\test_checkpoint_validation_failures.py tests\integration\test_checkpoint_schema_migrations.py --junitxml=artifacts\junit\phase7-artifacts-red.xml
```

Expected: nonzero exit code because versioned manifests and immutable checkpoints do not exist.

- [ ] **Step 3: Implement canonical manifests and atomic checkpoint publication**

The build manifest contains authored, canonical, and resolved specs; catalog snapshot and digest; graph variables, factors, measures, causal order, recognition information sets, source supports, geometry, ledger, parameter dependencies, compatibility results, selected kernels, exactness, and graph and layout digests. The run manifest adds data and tokenizer hashes, source-control identity and dirty-state fingerprints at start and finish, software and CUDA versions, device, dtype, determinism, seeds, parameter counts, checkpoint hashes, and lifecycle state. Fingerprints for model, recognition, objective, inference, numerics, data, training policy, and whole run remain separate.

Publish each checkpoint from a same-volume temporary sibling into an immutable directory containing `manifest.json`, `state.json`, `tensors.safetensors`, and `hashes.json`. Durable artifacts contain no custom Python-object pickle. Validate hashes, versions, graph and catalog digests, components, parameter names, shapes, dtypes, optimizer groups, data, and cursor before mutating live state. `exact_continuation`, `weights_only`, `fork_training`, and `initialization_import` have separate allowlists and lifecycle labels. Metadata migrations are pure sequential transformations and never rewrite the source artifact.

- [ ] **Step 4: Run green artifact tests**

Run:

```powershell
python -m pytest tests\contracts\test_manifest_schema.py tests\contracts\test_checkpoint_container.py tests\integration\test_checkpoint_resume_modes.py tests\integration\test_checkpoint_validation_failures.py tests\integration\test_checkpoint_schema_migrations.py --junitxml=artifacts\junit\phase7-artifacts.xml
```

Expected: exit code 0; exact continuation is trajectory identical, allowed forks are labeled, incompatible resumes fail before mutation, and all durable tensors use safetensors.

- [ ] **Step 5: Commit artifacts and checkpointing**

```powershell
git add src\vfe4\artifacts tests\contracts\test_manifest_schema.py tests\contracts\test_checkpoint_container.py tests\integration\test_checkpoint_resume_modes.py tests\integration\test_checkpoint_validation_failures.py tests\integration\test_checkpoint_schema_migrations.py pyproject.toml
git commit -m "feat: add immutable manifest-backed checkpoints"
```

### Task 7.5: Build the standalone one-way V3 initialization importer

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\initialization.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tools\v3_import\__init__.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tools\v3_import\schema.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tools\v3_import\converter.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\migration\fixtures\v3_minimal\manifest.json`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\migration\fixtures\v3_minimal\tensors.safetensors`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\migration\test_v3_import_report.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\migration\test_v3_import_rejections.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\migration\test_no_v3_runtime_dependency.py`

**Interfaces:**

- Consumes: frozen V3 JSON and tensor-state fixtures without importing or executing V3 code.
- Produces: V4 `InitializationBundle`, `ConversionReport`, and per-field statuses `IMPORTED`, `TRANSFORMED`, `SKIPPED`, or `REINITIALIZED`.

- [ ] **Step 1: Write failing conversion and dependency-isolation tests**

```python
def test_v3_converter_reports_every_source_and_destination_field() -> None:
    bundle, report = convert_v3_fixture(V3_MINIMAL_FIXTURE, target_spec=v4_target_spec())
    assert bundle.resume_mode is ResumeMode.INITIALIZATION_IMPORT
    assert report.unreported_source_fields == ()
    assert report.unreported_destination_fields == ()
    assert all(record.source_hash or record.status is ImportStatus.REINITIALIZED for record in report.records)


def test_v4_runtime_has_no_v3_import_edge() -> None:
    assert forbidden_imports_under("src/vfe4", names={"vfe3", "tools.v3_import"}) == ()
```

- [ ] **Step 2: Run the red migration tests**

Run:

```powershell
python -m pytest tests\migration\test_v3_import_report.py tests\migration\test_v3_import_rejections.py tests\migration\test_no_v3_runtime_dependency.py --junitxml=artifacts\junit\phase7-v3-import-red.xml
```

Expected: nonzero exit code because the standalone converter and audited initialization bundle do not exist.

- [ ] **Step 3: Implement only evidence-supported field mappings**

Map compatible embedding tables, prior tables, frame parameters, decoder parameters, dimensions, and selected metadata when source semantics, shapes, dtypes, and hashes are verified. Do not infer joint precision cross-blocks, normalized transitions, probabilistic source variables, structured recognition factors, hierarchy latents, or update certificates. Each absent V4 requirement is explicitly reinitialized or rejected. The output can initialize a new V4 run but is never accepted for exact continuation. No module under `src/vfe4` imports the standalone tool.

- [ ] **Step 4: Run green migration tests**

Run:

```powershell
python -m pytest tests\migration --junitxml=artifacts\junit\phase7-v3-import.xml
```

Expected: exit code 0; all fields receive auditable statuses and static import checks find no V3 runtime dependency.

- [ ] **Step 5: Commit the one-way importer and frozen fixtures**

```powershell
git add src\vfe4\artifacts\initialization.py tools\v3_import tests\migration
git commit -m "feat: add audited one-way V3 initialization import"
```

### Task 7.6: Define and smoke-test matched baseline and ablation cells

**Files:**

- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\runtime\train.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\experiments\components.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\experiments\protocol.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\experiments\baselines.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\experiments\runner.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\data.py`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\language\metrics.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\experiments\run_baseline_suite.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_experiment_protocol.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_matched_baseline_cells.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_predictive_metric_information_sets.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_baseline_artifact_contract.py`

**Interfaces:**

- Consumes: frozen experiment specs, separate training/evaluation/generation programs, data and tokenizer identities, seed schedules, manifests, metrics, and artifact writers.
- Produces: click-run baseline cells `B0`, `H1`, `H2`, `H3a`, `H3b`, and `H4` through `H7`, `MatchedProtocol`, `ExperimentRequest`, `ExperimentRunner`, and append-only multi-seed artifacts.

- [ ] **Step 1: Write failing matching and predictive-information tests**

```python
def test_all_cells_share_declared_matching_axes() -> None:
    suite = build_baseline_suite()
    for cell in suite.cells:
        assert cell.data_digest == suite.protocol.data_digest
        assert cell.tokenizer_digest == suite.protocol.tokenizer_digest
        assert cell.context_length == suite.protocol.context_length
        assert cell.seed_schedule == suite.protocol.seed_schedule
        assert cell.training_budget == suite.protocol.training_budget


def test_perplexity_uses_only_target_blind_prior_predictive_probabilities() -> None:
    evaluator = compile_baseline_suite_for_tiny_fixture().predictive_evaluator
    assert evaluator.information_set.excludes_current_target
    assert evaluator.information_set.includes_normalized_prefix_filter_state
    assert evaluator.information_set.excludes_target_aware_training_recognition
    assert evaluator.metrics == ("token_nll", "perplexity")
```

- [ ] **Step 2: Run the red experiment-protocol tests**

Run:

```powershell
python -m pytest tests\contracts\test_experiment_protocol.py tests\integration\test_matched_baseline_cells.py tests\integration\test_predictive_metric_information_sets.py tests\integration\test_baseline_artifact_contract.py --junitxml=artifacts\junit\phase7-experiments-red.xml
```

Expected: nonzero exit code because matched cells and their artifact contract do not exist.

- [ ] **Step 3: Implement the predeclared B0 and H1 through H7 suite**

`B0` reuses the early deterministic-state CE control and the same causal transition and normalized emission interfaces. `H1` compares the minimal `pq` ELBO with `B0`. `H2` compares structured precision with an explicitly selected mean-field restriction; mean-field is a control, not the default. `H3a` compares information-form and moment-form solver parameterizations of the same Gaussian law. `H3b` separately compares registered Fisher-natural and Euclidean optimization of the same parameter block. `H4` compares explicit source recognition with fixed or uniform routing. `H5` compares trivial geometry, fixed transport, and learned covariant geometry. `H6` compares dense direct, structure-preserving direct, and approximate matrix-free backends with separate evaluation, linear-algebra, and arithmetic records. `H7` compares `pq` with normalized `hspq` and a matched larger state-only control.

Match data, tokenizer, context, seed schedule, training budget, parameter accounting, and prior-predictive evaluation. Record parameter count, wall time, peak memory, and approximation status rather than calling a comparison parameter- or compute-matched when only the shared protocol is matched. Report uncapped token-weighted held-out NLL and perplexity from target-blind prediction, ELBO ledger sectors, posterior-prior KL, source entropy and utilization, effective rank or precision diagnostics, calibration, gradient and update acceptance diagnostics, and variation across seeds. Posterior-assisted scores remain diagnostics and cannot populate predictive metric fields.

- [ ] **Step 4: Run green protocol tests before reserving run directories**

Run:

```powershell
python -m pytest tests\contracts\test_experiment_protocol.py tests\integration\test_matched_baseline_cells.py tests\integration\test_predictive_metric_information_sets.py tests\integration\test_baseline_artifact_contract.py --junitxml=artifacts\junit\phase7-experiments.xml
```

Expected: exit code 0; the complete multi-seed request validates before any artifact directory is created.

- [ ] **Step 5: Commit the frozen experiment suite**

```powershell
git add src\vfe4\runtime\train.py src\vfe4\experiments src\vfe4\language\data.py src\vfe4\language\metrics.py experiments\run_baseline_suite.py tests\contracts\test_experiment_protocol.py tests\integration\test_matched_baseline_cells.py tests\integration\test_predictive_metric_information_sets.py tests\integration\test_baseline_artifact_contract.py
git commit -m "feat: add matched VFE baseline experiment suite"
```

- [ ] **Step 6: Execute the click-run multi-seed architecture-smoke suite on the RTX 5090**

Run:

```powershell
python experiments\run_baseline_suite.py
```

Expected: exit code 0; using the checked-in finite fixture and smoke seed schedule, the runner writes manifests, append-only metrics and events, immutable checkpoints, lifecycle status, and materialized summaries for every cell and seed. These smoke artifacts validate comparability and lifecycle mechanics and do not support a scientific performance claim. A full-data multi-seed protocol requires its own frozen `DataSpec`, tokenizer and split hashes, budget, hypothesis record, and execution approval.

- [ ] **Step 7: Validate the produced artifacts without asserting a favored hypothesis**

Run:

```powershell
python -m pytest tests\integration\test_baseline_artifact_contract.py --junitxml=artifacts\junit\phase7-baseline-artifacts.xml
```

Expected: exit code 0; every scheduled cell has a terminal lifecycle record and complete metric, exactness, compute, uncertainty, and provenance fields. A scientific hypothesis may fail without failing the artifact contract.

### Task 7.7: Run the complete architecture gate and write the execution handoff

**Files:**

- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\src\vfe4\artifacts\junit.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tools\verify_junit_reports.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\contracts\test_junit_summary.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_extension_without_callsite_edits.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\integration\test_architectural_completion.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\tests\cuda\test_architectural_completion_cuda.py`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\architecture-contract.md`
- Create: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\verification-matrix.md`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\docs\build-handoff.md`
- Modify: `C:\Users\chris and christine\Desktop\VFE_4.0\README.md`

**Interfaces:**

- Consumes: all phase gates, JUnit XML, catalog and runtime signatures, exact and production backends, checkpoints, importer, and baseline artifacts.
- Produces: `JUnitSummary`, an architecture-completion test, a verification matrix linked to machine-readable evidence, and an execution handoff that states remaining work without converting it into a completion claim.

- [ ] **Step 1: Write failing extension and completion tests**

The extension test snapshots shared training, evaluation, generation, graph, objective, and backend call sites. Fresh builders then add experiment-local descriptors for a normalized factor, statistical family, mixture composer, gauge group, representation, frame-derived connection, independent-edge connection, cross-fiber morphism, typed divergence, hierarchy profile, inference plan, source prior, and numerical backend. Each extension receives its own contract fixture. The builders freeze and compile valid combinations, reject deliberately incompatible ones through capability diagnostics, and prove that none of the shared call-site snapshots changed. The CPU completion test constructs a click-run cell, emits its complete compatibility and exactness manifest, executes the dense float64 reference, reproduces the ordinary ELBO from the ledger, saves and exactly resumes a checkpoint, and rejects every unlabeled approximation. The separately marked CUDA completion test runs one float32 training and target-blind predictive-evaluation step through the same compiled graph.

- [ ] **Step 2: Run the red final architecture tests**

Run:

```powershell
python -m pytest tests\contracts\test_junit_summary.py tests\integration\test_extension_without_callsite_edits.py tests\integration\test_architectural_completion.py tests\cuda\test_architectural_completion_cuda.py --junitxml=artifacts\junit\phase7-architecture-red.xml
```

Expected: nonzero exit code until JUnit validation, extension proof, and completion composition are wired.

- [ ] **Step 3: Implement the evidence reader and write contract documentation**

`JUnitSummary` reads XML and returns test, failure, error, and skipped counts without scraping console text. The fixed, click-run `tools/verify_junit_reports.py` reads the required report set, rejects missing or dirty reports, and writes one JSON summary; it takes no command-line arguments. `docs/verification-matrix.md` maps every gate to its XML and durable artifact. `docs/architecture-contract.md` records the stable public boundaries and pure-path obligations. `docs/build-handoff.md` records the branch, last completed checkbox, exact failing command if any, owned artifacts, and the next executable step. Documentation does not claim a gate passed until the corresponding machine-readable artifact has been inspected.

- [ ] **Step 4: Run the complete CPU gate**

Run:

```powershell
python -m pytest tests\contracts tests\oracles tests\integration tests\migration -m "not cuda" --junitxml=artifacts\junit\phase7-full-cpu.xml
```

Expected: exit code 0; inspect the XML and require zero failures and zero errors. CUDA tests deselected by this CPU command do not satisfy the CUDA gate.

- [ ] **Step 5: Run the complete CUDA gate on the RTX 5090**

Run:

```powershell
python -m pytest tests\cuda --junitxml=artifacts\junit\phase7-full-cuda.xml
```

Expected: exit code 0; inspect the XML and require zero failures, zero errors, and zero skipped CUDA tests.

- [ ] **Step 6: Run static quality and architecture checks**

Run:

```powershell
python -m ruff check src tests tools experiments
python -m mypy src tests tools experiments
python -m pytest tests\migration\test_no_v3_runtime_dependency.py tests\integration\test_compiled_signature_purity.py tests\integration\test_extension_without_callsite_edits.py --junitxml=artifacts\junit\phase7-static-architecture.xml
python tools\verify_junit_reports.py
```

Expected: all four commands exit 0; inspect `artifacts\junit\summary.json` and require every mandatory report to have zero failures and errors and the CUDA reports to have zero skips.

- [ ] **Step 7: Update the handoff from inspected evidence and commit final verification surfaces**

Record exact XML counts and artifact digests only after reading them. If any command fails, leave the checkbox open and write the failure and next command in `docs/build-handoff.md`.

```powershell
git add README.md docs src\vfe4\artifacts\junit.py tools\verify_junit_reports.py tests\contracts\test_junit_summary.py tests\integration\test_extension_without_callsite_edits.py tests\integration\test_architectural_completion.py tests\cuda\test_architectural_completion_cuda.py artifacts
git diff --cached --check
git status --short
git commit -m "docs: record VFE4 architecture verification"
```

- [ ] **Step 8: Complete the mandatory remote and worktree lifecycle**

Fetch and inspect `origin/main`, push the task branch, merge it from a clean integration worktree only after all required gates pass, push `main`, and fetch again to verify the resulting `origin/main` commit. Fast-forward the user's local `main` only if its tracked and untracked state proves that the operation cannot alter user work. Remove the task worktree and local task branch only after the remote merge is verified. The final report names the task branch, commit, pushed remote branch, resulting `origin/main`, CPU and CUDA XML evidence, worktree removal, and the owner of every remaining dirty path.

## Definition of Done

- [ ] The dense CPU float64 path still executes independently and passes all exact finite and Gaussian oracle gates.
- [ ] Every registered probability law and factor normalizes on its declared support and reference measure.
- [ ] Every recognition graph is either one normalized joint law or an acyclic ordered disintegration whose nonoverlapping targets cover the complete latent event exactly once.
- [ ] Monolithic and local objective ledgers have identical inventories and values for every enabled hierarchy profile.
- [ ] Target-aware training recognition is absent from predictive evaluation and generation signatures and manifests, while the normalized filter over the observed prefix remains present.
- [ ] Natural, expectation, and moment charts represent the same full-Gaussian law within the oracle tolerance.
- [ ] Dense, block-banded, sparse-direct, irrep-decomposed, and bounded CUDA results match the appropriate oracle within named tolerances.
- [ ] Every matrix-free, stochastic, projected, or regularized route is opt-in and carries separate objective-evaluation, linear-algebra, arithmetic, family-relation, and optimization records.
- [ ] Complete model and recognition paths satisfy their declared gauge-covariance tests.
- [ ] The RTX 5090 `T=128`, `K=20` gate runs with float32 CUDA and no global dense precision or covariance allocation.
- [ ] Checkpoints are immutable, hash-validated, pickle-free, and exactly resumable under the exact-continuation contract.
- [ ] The one-way V3 converter has no V3 runtime import and cannot claim exact continuation.
- [ ] The bounded B0-versus-minimal-`pq` falsifier is recorded before sparse, gauge, hierarchy, mixture, or irrep expansion.
- [ ] B0, H1, H2, H3a, H3b, and H4 through H7 use the declared matching protocol, and predictive metrics use only target-blind prior-predictive probabilities.
- [ ] Factor, family, mixture, group, representation, connection, morphism, divergence, hierarchy, inference-plan, source-prior, and numerical-backend extensions compile through local descriptors without shared runtime call-site edits.
- [ ] Machine-readable JUnit XML, manifests, hashes, metrics, and lifecycle records support every reported count and status.
- [ ] The task branch is committed, pushed, merged to `main`, verified against `origin/main`, and cleaned up without altering user work.

## Execution Handoff

The implementing coordinator updates this plan's checkboxes only from current tool evidence and keeps `docs/build-handoff.md` current at every phase boundary. A worker stops at the first failing promotion gate, records the exact command, XML path, failure category, owned files, and next test, and does not begin the next phase. No future experimental outcome, verification count, repository state, or remote status is claimed by this plan document.
