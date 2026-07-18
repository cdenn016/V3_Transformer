# VFE 4.0 Modular Architecture Design

## Status, authority, and scope

This document defines the greenfield architecture for VFE 4.0. It is a design specification, not an implementation record. Implementation has not begun. No VFE 3.0 module is designated for in-place conversion, and no VFE 4.0 package, checkpoint, or experimental result exists at the time of writing.

The mathematical authority for the design is the local [VFE 4.0 Gauge-Causal ELBO White Paper](<C:/Users/chris and christine/Desktop/Research/manuscripts/VFE4_gauge_causal_elbo_whitepaper.tex>) and its included chapters under `Research/manuscripts/vfe4_whitepaper/`. The architectural motivation and the boundary between the current V3 objective and a genuine global evidence bound are documented in the [Global ELBO and Language-Model Observations investigation](../../2026-07-17-global-elbo-language-model-observations-investigation.md). Where this design is more restrictive than the set of mathematically imaginable models, the restriction serves one of three purposes: preserving a normalized joint and a truthful evidence interpretation, making component compatibility decidable before execution, or retaining small finite-discrete and conjugate-Gaussian reference programs against which production approximations can be tested.

The selected architecture is Option C: an immutable typed generative and recognition `ModelGraph`, instance-scoped capability catalogs frozen before compilation, and compile-time lowering to resolved inference and numerical backends. Extensibility ends at the compiler boundary. A running model does not perform string-based family, group, divergence, hierarchy, or backend dispatch.

V1 retains the project's no-neural-network constraint. It introduces no `nn.Linear`, multilayer perceptron, or activation-module capacity. Model capacity resides in probability-law parameters, normalized factors, gauge and representation data, and iterative inference. In this document, amortized initialization means a registered analytic or table-based map unless that project constraint is separately revised.

## Architectural decision

Three broad architectures were considered. Option A extends the V3 monolith by adding more fields to one configuration object and more registry-dependent branches to shared update routines. Option B treats every semantic and numerical object as a dynamically discoverable plugin. Option C fixes a small semantic intermediate representation while allowing registered implementations to populate it and lower it.

| Option | Construction | Short-term advantage | Long-term failure mode | Decision |
|---|---|---|---|---|
| A. Monolithic extension | One model class, one large configuration schema, shared update functions with mode branches | The first additional family or group can be added quickly | The cross-product of family, hierarchy, geometry, objective, and backend choices becomes a runtime validation and branching problem; objective semantics become difficult to audit | Rejected |
| B. Fully dynamic plugins | Runtime discovery of arbitrary graph, law, objective, and backend plugins | Maximum external extensibility | Hidden replacement, unstable capability contracts, weak checkpoint reproducibility, and no bounded semantic surface on which to prove normalization or objective identity | Rejected for V1 |
| C. Typed graph plus frozen catalogs | Closed semantic graph types, open build-time catalogs, compile-time capability validation, resolved callable plans | Each extension has an explicit contract and leaves the hot path unchanged | Requires a real compiler and versioned capability model before broad experimentation | Selected |

Option C is not a compromise in which part of the runtime remains monolithic. The closed portion is the language used to state probability models: variables, measures, fibers, normalized factors, recognition factors, parameter blocks, and objective class. The open portion supplies lawful implementations of those types. After compilation, the result is a fixed tensor layout and a tuple of bound operations.

## The normalized probability boundary

The exact core follows the white paper's conditional generative model. With deterministic geometric data \(\Gamma\), continuous state variables \(\boldsymbol z\), continuous model variables \(\boldsymbol m\), categorical source variables \(a,b\), and observed tokens \(x\), the authoritative form is

\[
\begin{aligned}
p_\theta(x,\boldsymbol z,\boldsymbol m,a,b\mid\Gamma)
={}&p_{\theta,0}(z_0,m_0\mid\Gamma) \\
&\prod_{t=1}^{T}\pi_t^m(b_t)
K^m_{\theta,tb_t}(m_t\mid m_{b_t},x_{<t},\Gamma) \\
&\prod_{t=1}^{T}\pi_t^z(a_t)
K^z_{\theta,ta_t}(z_t\mid z_{a_t},m_t,x_{<t},\Gamma) \\
&\prod_{t=1}^{T}L_{\theta,t}(x_t\mid z_t,m_t,\Gamma).
\end{aligned}
\]

Every displayed factor is normalized over its declared child and reference measure for every fixed parent configuration. A generative factor may depend on model parameters and earlier generated or observed history permitted by the causal graph. It may not depend on a variationally moving posterior object. This rule excludes the direct reuse of a live posterior-to-posterior divergence as a transition law.

The recognition graph declares a normalized law \(Q_\psi^{(r)}(Y\mid x,\Gamma)\), where \(Y=(\boldsymbol z,\boldsymbol m,a,b)\) and \(r\) identifies filtering or smoothing information. The ordinary evidence lower bound is defined only as

\[
\mathcal L^{(r)}(\theta,\psi;x,\Gamma)
=\mathbb E_{Q_\psi^{(r)}}
\left[
\log p_\theta(x,Y\mid\Gamma)
-\log Q_\psi^{(r)}(Y\mid x,\Gamma)
\right].
\]

The compiler may decompose this scalar into local terms, but it cannot create, delete, or reweight those terms while retaining the evidence-ELBO label. The identity

\[
\log p_\theta(x\mid\Gamma)
=\mathcal L^{(r)}
+D_{\mathrm{KL}}
\left(
Q_\psi^{(r)}(Y\mid x,\Gamma)
\mathrel{\|}
p_\theta(Y\mid x,\Gamma)
\right)
\]

is the semantic acceptance test for the ordinary objective. Generalized Bayes, consensus regularization, projection penalties, and diagnostic divergences remain available, but they occupy different objective types and produce different artifact labels.

## Cross-entropy as the categorical observation sector

Language tokens are observations. For a categorical emission with vocabulary \(\mathcal V\),

\[
L_{\theta,t}(x_t=w\mid z_t,m_t,\Gamma)
=\operatorname{softmax}(\ell_{\theta,t}(z_t,m_t,\Gamma))_w.
\]

Its negative log likelihood is categorical cross-entropy. VFE 4.0 therefore does not replace cross-entropy in an autoregressive language model. It places the expected categorical cross-entropy inside the global latent-variable free energy. If the posterior over \((z_t,m_t)\) is nondegenerate, the exact accuracy term is

\[
-\mathbb E_{Q_\psi^{(r)}}
\left[
\log L_{\theta,t}(x_t\mid z_t,m_t,\Gamma)
\right],
\]

not the decoder evaluated at a posterior mean unless a registered analytic identity proves equality. A deterministic state collapses the expectation and recovers ordinary pointwise cross-entropy. A Gaussian-template decoder, energy-based vocabulary decoder, or other categorical parameterization still requires a normalized softmax or equivalent vocabulary partition function if the system is to define token probabilities and perplexity.

This boundary also separates training recognition from evaluation. A filtering posterior may observe \(x_t\) while inferring the latent state associated with \(x_t\). The predictive prior used for next-token evaluation and generation may depend only on \(x_{<t}\), earlier latent variables, and deterministic geometry. Perplexity is computed from the target-blind predictive distribution after integrating latent uncertainty. It is never computed from an observation-conditioned posterior that has already seen the scored token.

## The statistical-law type system

The core abstraction is a probability law on a declared measurable event space with a declared reference measure. The architecture must not assume that every law is Gaussian, belongs to a regular exponential family, possesses a finite covariance, or admits reparameterized sampling. These distinctions are represented in types and capability records rather than inferred from names.

`Law` is the most general runtime contract. It declares its event domain, support, reference measure, batch and event shapes, normalized log density or mass, and sampling capabilities. A law may additionally expose entropy, moments, a pushforward, or a conditional normalizer, but absence is represented explicitly. An `ExponentialFamily` refines `Law` by providing sufficient statistics \(T(x)\), natural parameters \(\eta\), and log partition \(A(\eta)\). Regularity alone does not imply a globally invertible natural--expectation chart. The separate `LegendreDualCapable` contract requires a minimal regular representation, or an explicit quotient or gauge fixing, together with a proved bijection between its declared natural and expectation chart domains. Generic reverse chart conversion and dual-coordinate conjugacy are available only through that capability.

Fisher geometry is a separate capability. A regular differentiable law may implement `FisherMetricCapable` without possessing a global natural--expectation Legendre chart. A labeled or marginalized mixture may therefore expose an exact or estimated Fisher operator while remaining outside `LegendreDualCapable`. A natural-gradient label requires both this metric capability and an update implementation that applies the inverse Fisher operator; neither an information-form storage layout nor a natural-coordinate field is sufficient.

`LabeledMixture` is a joint law over a categorical label and a component event. It retains the label in the event space, so its entropy and source posterior can be decomposed without pretending that the continuous marginal belongs to the component family. `MarginalMixture` hides the label and evaluates the resulting log-sum-exp density. It is a separate type because its entropy, Fisher geometry, moments, and closure properties differ from those of both a labeled joint and a single component. A moment-matched family is a declared projection result, not an alias for the mixture.

The initial contracts are compositional rather than one nominal inheritance tree. `DiscreteLaw`, `ContinuousLaw`, and `MixedLaw` classify the reference-measure domain. `ExponentialFamily` refines any compatible law and may therefore describe both a categorical discrete law and a Gaussian continuous law; `LegendreDualCapable` is attached only after minimality or quotient handling is certified. Full categorical logits, for example, have a common-shift redundancy and do not receive a bijective dual-chart certificate until a reference logit or zero-sum gauge is fixed. `LabeledMixture` and `MarginalMixture` are composite-law contracts that retain their component capabilities without inheriting unsupported closed forms. A future family enters through the smallest truthful combination. For example, a varying-location Laplace law can implement `ContinuousLaw` without claiming chart capabilities that do not exist for that parameterization.

## Natural, expectation, moment, and optimization charts

VFE 4.0 uses chart-aware parameter layouts. A law implementation declares which charts it supports and which conversion directions are certified on a stated domain. Four chart roles are distinguished. The natural chart supports exponential-family algebra and information-form conditioning. A `LegendreDualCapable` expectation chart supports dual geometry and invertible moment constraints after redundancy has been removed. The moment chart supports interpretation, diagnostics, and interfaces such as decoders. The unconstrained optimization chart supplies numerically valid trainable coordinates, such as a Cholesky factor or matrix-log precision.

For a multivariate Gaussian, the canonical inference storage is the information pair \((h,J)\), where

\[
J=\Sigma^{-1},
\qquad
h=J\mu.
\]

The Gaussian natural chart is \((h,-J/2)\), not the storage pair \((h,J)\). In code, the fields are named `information_vector` and `precision` so the mathematical \(h\) is not confused with the hierarchy-profile label. The moment pair \((\mu,\Sigma)\) is derived when required. The expectation chart contains \(\mathbb E[z]\) and \(\mathbb E[zz^\top]\). Trainable precision may be represented through a registered positive-definite parameterization, but the compiler records that optimization chart separately from both information storage and the law's semantic natural chart.

No universal `mu`, `sigma`, or `covariance` fields appear in `Law`, graph nodes, checkpoints, or backend protocols. Components request capabilities such as `natural_chart`, `second_moment`, or `precision_operator`. This prevents a new family from being forced into Gaussian-shaped state and prevents a diagonal scale from being silently interpreted as a variance.

## The p, q, h, and s hierarchy

The names `pq` and `hspq` are user-facing graph-profile aliases, not four semantic tensor or random-variable classes. The semantic IR names realized latent variables by their event spaces. The state variable is \(z_t\in E^z_t\), the optional model variable is \(m_t\in E^m_t\), the generative graph supplies normalized kernels \(K^z\) and \(K^m\), and the recognition graph supplies separately named conditionals \(q^{z,(r)}\) and \(q^{m,(r)}\).

Under the profile crosswalk, `p` refers to the relevant state predictive kernel or conditional law and `q` to the recognition conditional over the same state event. The `h` label refers to the model-channel generative prior or transition, while `s` refers to the model-channel recognition conditional. These labels do not appear as node kinds in the compiled graph. The ordinary ELBO compares recognition and generative conditionals only after their complete conditioning sets have been aligned, verifies absolute continuity on the same event fiber, and averages the conditional KL over the recognition law of the parents and history. A naked marginal \(D_{\mathrm{KL}}(q_t\mathrel{\|}p_t)\) is not substituted for that expected conditional term.

The model fiber \(E^m_t\) need not have the same dimension as the state fiber \(E^z_t\). The channels communicate through explicit normalized kernels. A typical bridge is

\[
K^z_{\theta,tj}(dz_t\mid z_j,m_t,x_{<t},\Gamma),
\]

whose location may contain a typed morphism \(B_t:E^m_t\to E^z_t\). If two adjacent hierarchy levels use different event dimensions, they likewise require a normalized conditional or two explicit pushforwards to a common measure space. A direct KL across different event fibers is undefined, and a non-square deterministic map may produce a singular pushforward. Equating a model distribution with a state prior, copying a tuple between fibers, or relying on equal dimensions is not a valid bridge.

Hierarchy profiles are build-time graph constructors. A `pq` profile emits \(z\), its generative kernels, and \(q^{z,(r)}\). An `hspq` profile additionally emits \(m\), \(K^m\), \(q^{m,(r)}\), and normalized model-to-state coupling. A deeper profile may add layers only by emitting explicit variables and normalized kernels. Once expanded, all profiles compile to the same graph IR; runtime code has no hierarchy-mode branch.

## Generative and recognition graphs

The immutable `ModelGraph` contains variable nodes, parameter nodes, normalized factors, reference measures, fibers, causal order, and observation status. A companion `RecognitionGraph` contains normalized recognition factors, their conditioning sets, source supports, and information regime. The recognition graph may be mean-field, chain structured, tree structured, block Gaussian, source-labeled mixture, or another registered factorization. Mean-field is one explicit profile rather than an architectural assumption.

Each generative factor declares exactly one normalized child, its parents, the reference measure over the child, its parameter dependencies, and a registered kernel. Each recognition factor declares its target block, available observations, latent conditioning set, support contract, and law family. A `RecognitionGraph` is legal only when it is either one normalized joint law over all latent variables \(Y\), or an ordered disintegration whose target blocks partition \(Y\) exactly once. In the disintegrated form, every factor is normalized over its target block conditional on earlier latent targets and its permitted observations. The compiler proves target coverage, nonoverlap, acyclicity, and a topological integration order; an uncovered latent, duplicate target, backward dependency, or recognition-only cycle is a build error. This global invariant, rather than factor-level normalization alone, establishes one normalized \(Q\).

The compiler also rejects missing generative parents, target leakage, duplicate child generation, inconsistent support, unnormalized factor types, cycles that violate the declared causal order, and recognition factors that cannot be integrated or summed over their target.

Source variables are ordinary categorical nodes. Marginalizing a source generally produces a mixture. The graph must retain the source label when an exact labeled-mixture backend is selected; replacing the mixture by one Gaussian is an opt-in projection with an approximation record.

The graph compiler derives a parameter-dependency map and the Markov blanket of every latent or parameter block. It also derives an ELBO ledger with one stable identifier for every generative log factor and every recognition log factor. Specialized analytic objectives may replace groups of ledger rows only when they declare the exact rows they implement and pass equality tests against the generic evaluator.

## Gauge groups, representations, fibers, morphisms, and connections

The geometry system separates objects that V3 sometimes packages together. A `GaugeGroup` defines group multiplication, inverse, identity, Lie algebra operations where applicable, connected-component data, determinant behavior, and numerical parameterizations. It does not by itself define how a state transforms. A `Representation` binds a group to a carrier fiber through \(\rho:G\to\mathrm{GL}(E)\). A `FiberSpec` identifies the channel, base position, carrier dimension, scalar field, representation, reference measure, and optional decomposition. A `FiberMorphism` has explicit source and target fibers. A `Connection` assigns typed transports to graph edges and declares whether they are derived from frames, independent edge variables, fixed data, or learned parameters.

State and model channels may use different dimensions and different representations of the same group. They may instead use a product gauge group \(G_z\times G_m\), with separate local actions on \(E^z\) and \(E^m\). Under a product action, a cross-channel field \(B_t\in\operatorname{Hom}(E^m_t,E^z_t)\) transforms as

\[
B_t\mapsto \rho_z(g^z_t)B_t\rho_m(g^m_t)^{-1}.
\]

This covariant Hom field is distinct from a fixed intertwiner. A fixed map \(B\) between two representations of one group is an intertwiner only when

\[
\rho_z(g)B=B\rho_m(g)
\]

for every \(g\). The compiler verifies this identity analytically from a registered decomposition when possible and numerically on generators as a secondary check. A learned unrestricted matrix is neither an invariant intertwiner nor a covariant field unless its transformation law is declared and enforced.

An irrep decomposition is typed representation metadata. For a semisimple complex representation written schematically as

\[
E\cong\bigoplus_\lambda V_\lambda\otimes M_\lambda,
\]

equivariant endomorphisms act through the commutant on multiplicity spaces. The catalog records the scalar-field and real-type qualification needed for real, complex, and quaternionic commutants rather than assuming the complex Schur form universally. Exact block layouts are derived from this decomposition. A block-diagonal approximation that discards allowed coupling is a separate numerical projection and receives an approximation label.

Frame-derived transport and independent connection transport remain separate connection kinds. Frame-derived transport may be pure gauge and have identity loop holonomy. Nonzero curvature, Wilson-loop, or Yang-Mills sectors require independent connection degrees of freedom or another graph construction that can support them. The deterministic-geometry V1 core conditions on \(\Gamma\); a latent-geometry ELBO is outside V1 because it additionally requires a proper geometry prior, recognition law, reference measure, and treatment of noncompact gauge volume.

The categorical emission participates in the same gauge contract. State and model readouts transform by the dual or contragredient representations so their vocabulary logits are unchanged under simultaneous frame changes. If decoder weights are held fixed, the compiler restricts the claimed symmetry to the emission-kernel stabilizer; for a softmax this may include transformations that change every vocabulary logit by the same scalar. Complete-objective covariance tests transform the generative law, recognition law, transition covariances and precisions, morphisms, decoder readouts, and reference-measure Jacobians together. Transition-only covariance is not accepted as a full-ELBO certificate.

## Source-prior taxonomy

Source priors are normalized categorical generative factors, not arbitrary attention-score providers. Every source-prior descriptor declares its support, conditioning set, target-blindness, log normalizer, sampling operation, and whether its probabilities are exact or estimated.

| Source-prior class | Permitted conditioning | Exact-core status | Interpretation |
|---|---|---|---|
| Uniform structural | Causal parent set only | Included | Exchangeable ancestry over permitted sources |
| Positional causal | Distances, masks, and fixed position data | Included | Locality or recency prior |
| Geometry conditioned | Deterministic \(\Gamma\), transports, and invariant edge data | Included when normalized | Structural geometric ancestry |
| History conditioned | Earlier observed tokens and earlier generated latent variables | Future extension | A content-dependent generative prior that changes the joint and requires a new normalization audit |
| Recognition source posterior | Current or future observations permitted by filtering or smoothing regime | Included in recognition only | Attention-like posterior routing |
| Heuristic score row | Query-key or consensus score without a declared generative normalizer | Diagnostic or generalized objective only | Routing computation, not an evidence-model prior |

The exact V1 joint begins with fixed structural, positional, or deterministic-geometry priors. Posterior source rows may be content dependent because they belong to recognition. A future history-conditioned generative source prior must remain target blind and normalized over its positive support. The compiler records the distinction so a recognition score cannot be silently reclassified as a model prior.

Uniform causal, learned position-only, ALiBi, and T5-relative-bucket variants are source-prior implementations only after their masked logits have been normalized on the declared causal parent set. RoPE is not itself a source prior; it is deterministic positional geometry acting on features or transports. A RoPE-dependent content score becomes a source prior only when it is declared as a new target-blind normalized generative conditional. Source-prior and positional-geometry ablations therefore remain separate axes.

## Divergence semantics

The divergence catalog is role typed. `EvidenceRatio` is reserved for the log-density ratio induced by a declared normalized model and recognition law; its integrated complexity is an ordinary KL divergence with coefficient one. `GeneralizedBayesLoss` defines a loss-based posterior with its own temperature and normalizer. `ProjectionDivergence` selects an approximation within a family. `ConsensusPenalty` coordinates replicas or local solvers. `DiagnosticDivergence` is reported but not optimized as evidence.

An f-divergence, Rényi divergence, Hellinger distance, Bhattacharyya contrast, or weighted peer KL can be registered for the appropriate role. No configuration field can substitute one of these for the evidence KL while preserving the `evidence_elbo` objective type. If a divergence changes the model's normalized density, the corresponding factor and normalizer must be declared in the generative graph. If it changes only the loss, the run is labeled generalized Bayes or regularized training and cannot report its objective as log evidence.

## Exactness as a vector of claims

A single `pure` Boolean is too coarse. The compiler produces an `ExactnessReport` whose axes are independently `exact`, `approximate`, `unverified`, or `not_applicable`, with a reason and responsible component for every nonexact value.

| Axis | Question answered |
|---|---|
| Model normalization | Is every generative factor normalized over its declared child and is the full joint normalized by construction? |
| Recognition normalization | Is every recognition factor normalized on the declared support? |
| Evidence identity | Is the optimized scalar the ordinary ELBO of the declared joint? |
| Variational-family status | Is the selected representation proved equivalent to the declared family, a proper restricted family, or a projection with a recorded error? |
| Objective evaluation | Are all ledger terms evaluated exactly, analytically or by exact finite summation/integration available to the selected backend? |
| E update | Is each recognition update an exact coordinate optimum, a valid bound step, an accepted generalized-EM step, or an ordinary approximate optimizer step? |
| M update | Is each model-parameter update an exact maximizer, an accepted generalized-EM step, or a gradient proposal? |
| Source treatment | Are categorical sources retained, exactly marginalized, sampled, or projected? |
| Linear algebra | Are solves, log determinants, and marginal blocks exact for the declared finite representation, or estimated iteratively? |
| Geometry | Is geometry deterministic, exactly marginalized, variationally approximated, or omitted? |
| Gauge covariance | Do every factor, morphism, and numerical kernel satisfy the declared transformation law? |
| Causal evaluation | Does scoring use only target-blind predictive information? |

The semantically exact path requires normalized model and recognition graphs, the ordinary evidence identity, a declared recognition family, and update certificates truthful about their guarantees. Exact finite source enumeration, analytic or structure-preserving Gaussian algebra, continuous-expectation evaluation, and floating-point error remain separate execution claims. Production float32 arithmetic is not called mathematically exact merely because it implements an exact formula.

## Catalogs, capabilities, and compilation

Catalogs are instance scoped. A mutable `CatalogBuilder` exists only while built-ins and experiment-local extensions are registered. Duplicate keys fail closed. Replacement requires an explicit operation naming the expected previous descriptor digest. Calling `freeze()` produces immutable mappings, a canonical catalog manifest, and a digest. Tests construct fresh builders and never save, mutate, or restore process-global dictionaries.

A component descriptor contains a stable key, catalog API version, implementation version, local immutable configuration schema, category-specific capability record, factory, compatibility predicates, and source provenance. Capability records are typed dataclasses rather than open dictionaries or string lists. Unary capability flags are insufficient for relations such as family closure under a representation or backend support for a particular factor-family pair, so descriptors may supply pure versioned compatibility predicates. The build manifest records every predicate identifier, its inputs, result, and diagnostic.

Compilation proceeds in a fixed order. The compiler strictly decodes and migrates the authored configuration, freezes or verifies the supplied catalog snapshot, expands hierarchy profiles, constructs the generative and recognition graphs, validates probability and geometry contracts, derives the ELBO ledger and Markov blankets, selects an inference backend, lowers semantic nodes to tensor layouts, binds numerical kernels, and emits a frozen `CompiledProgram`. Validation is aggregate: one failed build reports every discovered incompatibility rather than stopping at the first field.

The compiled hot path contains no unresolved component references. Its essential structure is

```python
@dataclass(frozen=True, slots=True)
class CompiledProgram:
    state_layout: StateLayout
    initialize: InitializeKernel
    e_sweep: tuple[UpdateKernel, ...]
    m_sweep: tuple[UpdateKernel, ...]
    objective: ObjectiveKernel
    predictive: PredictiveKernel
    generate: GenerationKernel
    manifest: BuildManifest
```

Training iterates over bound kernels. The authored configuration, catalog, string keys, and compatibility logic are absent from update signatures. An optimized implementation may replace a reference kernel only when both share a semantic operation identifier and the optimized descriptor states its exactness and tested equivalence domain.

## Core graph and catalog contracts

The semantic IR should remain small enough to inspect as data. The following sketch fixes ownership and dependency direction without prescribing implementation details beyond the public contracts:

```python
@dataclass(frozen=True, slots=True)
class VariableNodeSpec:
    node_id: NodeId
    role: VariableRole
    domain: EventDomainRef
    reference_measure: MeasureRef
    fiber: FiberRef | None
    observed: bool
    plate: PlateSpec

@dataclass(frozen=True, slots=True)
class ConditionalFactorSpec:
    factor_id: FactorId
    child: NodeId
    parents: tuple[NodeId, ...]
    parameters: tuple[ParameterId, ...]
    kernel: ComponentRef[ConditionalKernelSpec]
    normalized_over: NodeId

@dataclass(frozen=True, slots=True)
class RecognitionFactorSpec:
    factor_id: FactorId
    targets: tuple[NodeId, ...]
    latent_conditioning: tuple[NodeId, ...]
    observation_conditioning: tuple[NodeId, ...]
    law: ComponentRef[RecognitionLawSpec]
    support: SupportRef

@dataclass(frozen=True, slots=True)
class RecognitionDisintegrationSpec:
    latent_event: tuple[NodeId, ...]
    ordered_factors: tuple[RecognitionFactorSpec, ...]

@dataclass(frozen=True, slots=True)
class ParameterBlockSpec:
    parameter_id: ParameterId
    semantic_owner: FactorId | NodeId
    event_shape: tuple[int, ...]
    constraint: ConstraintRef
    optimization_chart: ChartRef
    update_block: UpdateBlockId
```

`normalized_over` is not documentation. It determines the integration or summation contract tested by the factor's common contract suite. A multi-output normalized conditional is represented by one joint child block rather than several factors that accidentally multiply overlapping conditionals. The recognition disintegration supplies the corresponding global coverage certificate: the ordered target blocks are pairwise disjoint, their union equals the declared latent event, and every latent conditioner occurs earlier in the order. Deterministic maps are represented as typed deterministic factors or as internal parameterizations of a normalized kernel; they are not assigned fictitious differential entropy.

Node and factor identifiers are semantic and stable across tensor layouts. A backend may pack several nodes into one tensor or split one structured node into several buffers, but it preserves an explicit bidirectional `StateLayout` map. Ledger entries and checkpoint parameter manifests refer to semantic identifiers first and packed tensor paths second. This makes a layout migration auditable and lets dense and sparse backends consume the same graph.

Catalog descriptors are similarly explicit:

```python
@dataclass(frozen=True, slots=True)
class ComponentDescriptor(Generic[LocalSpecT, ProductT, CapabilityT]):
    key: ComponentKey
    api_version: int
    implementation_version: str
    spec_type: type[LocalSpecT]
    capabilities: CapabilityT
    factory: ComponentFactory[LocalSpecT, ProductT]
    validators: tuple[CompatibilityPredicate, ...]
    provenance: ComponentProvenance

class CatalogBuilder:
    def register_family(self, descriptor: FamilyDescriptor) -> None: ...
    def register_factor(self, descriptor: FactorDescriptor) -> None: ...
    def register_group(self, descriptor: GroupDescriptor) -> None: ...
    def register_representation(self, descriptor: RepresentationDescriptor) -> None: ...
    def register_backend(self, descriptor: BackendDescriptor) -> None: ...
    def freeze(self) -> CatalogSnapshot: ...
```

Factories receive only their validated local spec and a read-only `BuildContext` containing resolved spaces, device, dtype, and dependent component products. They do not receive the root experiment configuration. This prevents a family or kernel from reading unrelated toggles and developing hidden cross-seam behavior. A descriptor's provenance contains its source module, implementation version, catalog API version, and distribution or source-tree identity. The catalog digest is computed from canonical descriptor metadata and capability records; source-control identity separately binds the implementation bytes.

The compiler maintains a typed obligation set. A factor contributes normalization, domain, support, chart, parameter, and representation requirements. A family contributes law and chart capabilities. A representation contributes carrier and invariant-structure capabilities. A backend contributes factor, storage, dtype, and estimator capabilities. Compatibility is the successful discharge of every obligation. It is not a best-effort search followed by runtime fallback.

Representative obligations include absolute continuity of recognition with respect to the relevant generative support, identical event fibers for any evidence KL, a declared normalized bridge between distinct model and state fibers, closure of a covariance or precision structure under the chosen representation, availability of each entropy and expected-log-factor operation used by the ledger, source support inclusion, availability of a predictive marginalization path, and backend support for every selected chart and parameter constraint. The compiler also checks parameter ownership. A tensor may be owned by the model, a recognition map or table, deterministic geometry, or local variational state, but never by two update blocks with incompatible semantics.

## Runtime state and inference scheduling

Immutable model meaning and mutable execution state are kept separate. `CompiledProgram` contains graph semantics, tensor layouts, bound kernels, update plans, and manifests. `RunState` contains model parameters \(\theta\), persistent recognition parameters \(\psi\), optimizer and scaler state, random streams, data cursor, and optional persistent variational caches. `BatchState` contains observation-conditioned local variational coordinates and temporary source beliefs for one batch. `Workspace` contains backend-owned scratch buffers whose contents have no checkpoint meaning.

This separation resolves an ambiguity common in iterative VFE implementations. A local posterior coordinate produced by an E sweep is not automatically a learned model parameter. If it is recomputed from each observation batch, it belongs to `BatchState`. If a registered analytic amortizer or recognition table predicts its initialization, its parameters belong to \(\psi\), while the realized posterior remains local. If an experiment deliberately persists local variational states across visits to examples, that cache receives a data-identity contract and a separate artifact schema.

The inference compiler creates a schedule from update blocks and Markov blankets. A schedule may perform exact block-coordinate ascent, conjugate message passing, generalized EM, amortized initialization followed by local refinement, or stochastic variational optimization. Each operation is bound to its ledger slice and update certificate before execution. Repeated refinement depth is a schedule parameter, not a stack of independently parameterized transformer layers unless the model explicitly declares time- or sweep-indexed parameters.

An exact coordinate schedule must establish that each update reads the current values of every variable in its Markov blanket and optimizes every affected ledger row. Parallel updates are allowed only when conditional independence or a valid Jacobi-style bound justifies them. Otherwise the compiler emits a sequential schedule or rejects the exact-coordinate request. Generalized-EM proposals may use any registered optimizer, but acceptance is tested against the complete compiled scalar and rejected proposals do not advance optimizer, scheduler, exponential-moving-average, checkpoint, or iteration state.

Decoding is not a separate loss seam. The categorical emission is a generative factor owned by \(\theta\), and its parameters appear in the same ledger as transition and complexity terms. An exact M coordinate, generalized-EM proposal, or stochastic gradient step updates those parameters according to the declared global objective. A decoder-only CE step followed by unrelated belief refinement is available only as the deterministic CE baseline or a labeled composite regularized objective, not as coordinate EM.

## Reference V1 semantically exact profile

The architecture needs one concrete profile that exercises its semantics without requiring every planned extension. The semantically exact reference profile conditions on deterministic geometry \(\Gamma\), uses a finite causal parent set at each position, retains categorical state and model source variables, represents continuous state with a joint full Gaussian information form, and uses a normalized categorical token emission. The smallest `pq` instance omits the model channel. The `hspq` instance adds a full-Gaussian model state on its own fiber and a normalized Gaussian state kernel whose location contains a typed model-to-state morphism.

In the dense reference programs, the sequence length, vocabulary subset, source supports, and fiber dimensions are deliberately tiny. Categorical sources are enumerated, not relaxed. Conditional Gaussian components use full precision blocks. When source labels are retained, the recognition law is a labeled mixture of structured Gaussian components and its mixed joint entropy is exact. Marginalizing the finite labels yields an exact pointwise log-sum-exp density, but it does not make the entropy or an arbitrary expectation of the resulting continuous mixture analytic. Those quantities use a registered identity where available, deterministic quadrature with a convergence report, or a stochastic estimator with a bias and error record. Moment matching remains a separate projection. The observation likelihood is evaluated at the latent state and integrated under the same recognition law used by the complexity terms.

The first geometry can be the trivial group or a compact matrix group with a verified finite-dimensional representation. The exact graph remains the same when a nontrivial deterministic connection is enabled; only registered transports and typed morphisms change. General \(\mathrm{GL}(K)\) representations may be added after the compiler can reject incompatible diagonal or block structures and after numerical tests cover nonorthogonal congruence. The exact reference does not require latent frames or a gauge-volume integral.

The reference inference family supports a dense structured recognition law and an explicitly separate mean-field restriction. In a conjugate Gaussian-observation test model, the dense posterior, objective evaluation, and coordinate updates are analytic up to floating-point error. In the categorical language model, the emission is nonconjugate, so the normalized model and ordinary-ELBO identity remain exact while continuous expectation evaluation and optimization carry their own quadrature, Monte Carlo, generalized-EM, or stochastic certificates. This distinction tests that a semantically exact objective does not imply analytic evaluation or a closed-form optimizer.

The production float32 profile lowers the same reference graph rather than constructing a second model. It may select banded or sparse structure when graph and representation contracts justify that layout. A proved structure-preserving lowering leaves the declared recognition family unchanged. Covariance truncation, low-rank replacement, or sparse information projection changes the variational-family status; source sampling changes source evaluation; stochastic log determinants and iterative solves change numerical evaluation. None is described as only a backend switch. The shared semantic graph and separately recorded family relation make reference comparison meaningful.

## Failure and fallback policy

Configuration and semantic failures occur before model tensor allocation or run-directory reservation. Unknown component keys, duplicate registrations, unmet capabilities, unsupported chart conversions, graph normalization failures, source-support mismatches, target leakage, and unavailable predictive evaluation are build errors. The diagnostic names the semantic nodes, factors, descriptors, and compatibility predicates involved.

Runtime numerical failure does not silently select a different family, precision structure, retraction, source treatment, or backend. A failed Cholesky factorization, nonfinite ledger term, iterative-solver nonconvergence, or violated positive-definite constraint produces a structured failure event and follows the experiment's explicit numerical policy. An opt-in recovery policy may reject an update, construct a separately identified regularized candidate, or terminate the cell. Adding jitter, rescaling, or flooring changes the Gaussian law even when the graph topology is unchanged; the altered candidate receives a conditioning report and a fresh complete-objective evaluation before it can be accepted. It is never an invisible factorization fallback for the original candidate.

Approximate fallbacks are therefore compile-time alternatives, not exception handlers. If an experiment wants dense-then-matrix-free fallback, it declares two separately compiled cells or an explicit backend policy whose possible outcomes and exactness labels are present in the manifest before training.

## ELBO ledger, Markov blankets, and update certificates

The ledger is the executable bridge between the probability graph and training. Each row records a stable term identifier, originating factor, sign, coefficient, expectation law, estimator, parameter dependencies, latent dependencies, normalization status, and exactness. The evidence compiler obtains coefficients from the log joint and log recognition law; users do not tune them independently. A generalized objective creates a different ledger type.

Markov blankets are derived from graph incidence and used to scope coordinate updates. A local update may read only the ledger rows whose expectations depend on its target block. This prevents an attention or decoder term from being dropped merely because an older closed-form update did not include it.

Every update kernel declares an `UpdateCertificate`. An exact coordinate certificate identifies the optimized block and proves or tests that the returned state maximizes the complete local objective. A minorize-maximize certificate identifies the bound and its touching ledger rows. A generalized-EM certificate requires complete-ELBO re-evaluation and a deterministic error bound or a declared confidence procedure whose lower improvement bound is nonnegative. If evaluation uncertainty cannot certify the sign of the improvement, the proposal is recorded as stochastic or heuristic rather than generalized EM. A stochastic-gradient certificate records estimator bias, variance assumptions, sample count, and optimization step. A heuristic certificate cannot claim an exact coordinate, minorize-maximize, or generalized-EM update. It places the optimization axis in an experimental regime without changing an otherwise normalized model or turning its ordinary ELBO into a generalized objective.

The compiler does not require every useful model to have closed-form E and M coordinates. A nonconjugate categorical emission can use generalized EM or stochastic variational inference while preserving one global ELBO. What it cannot do is label an update as a coordinate optimum when the update ignores a changing ledger term.

## Numerical backend hierarchy

The semantic graph is independent of numerical storage. Backends lower the same graph into dense, block, banded, sparse, or matrix-free operators when their capability predicates are satisfied.

Gaussian precision is assembled by construction from square-root information factors,

\[
J=J_0+\sum_f A_f^\top P_fA_f,
\qquad
P_f=R_fR_f^\top,
\]

with a strictly positive-definite anchor. A backend receives the factor structure and an objective-derived set of moment requests. Its factorization handle exposes `solve`, `quadratic`, `logdet`, bounded `selected_inverse`, `sample`, and `condition_report`. It does not expose a public full inverse or unrestricted covariance materializer. Mean vectors are obtained by solves, and sampling uses triangular solves, perturb-and-solve from the declared square-root factors, or a separately certified square-root action.

The `DenseF64Oracle` backend uses CPU float64, full dense precision matrices, finite enumeration of categorical labels, direct factorizations, and tiny problem sizes. It supplies trusted Gaussian algebra, exact labeled-source sums, and reference values for every analytic ledger term. It does not relabel a nonanalytic expected softmax likelihood or marginalized-mixture entropy as exact. The `DenseTorch` backend applies the same operations to small production systems in CUDA float32.

A block-diagonal backend is equivalent only when graph separation or representation theory proves the omitted blocks are zero; otherwise it is a restricted variational family. A block-banded backend is structure preserving for the declared finite-order Markov graph when the full required band is retained. A sparse-direct CPU backend records its symbolic ordering, factor fill pattern, fill ratio, numeric factor identity, and selected-inverse method; Cholesky fill is not mistaken for a change to the input precision law. An irrep-decomposed backend is a wrapper over one of these layouts and requires a representation certificate for every preserved block and multiplicity-space coupling.

A matrix-free CUDA backend reports each operation separately. Conjugate-gradient solves and basis-right-hand-side marginal blocks report tolerances, iterations, and residuals. Stochastic Lanczos log determinants report probes, truncation, seeds, and uncertainty. Sampling uses perturb-and-solve or a declared approximation to \(J^{-1/2}\); solving \(Jx=\epsilon\) for white \(\epsilon\) is rejected because it produces covariance \(J^{-2}\). A matrix-free path that needs a stochastic log determinant cannot claim exact objective evaluation.

Backend selection occurs during compilation. The runtime never checks whether covariance is diagonal, whether an irrep layout exists, or whether sparse algebra is enabled. The selected backend supplies already-bound operations and an operation-specific capability manifest.

Every production kernel that has a float64 reference analogue is tested against it on the largest tiny instance covered by the analytic or deterministic reference. Float32 tolerances are operation specific. SPD assembly, factorization, triangular solves, log determinants, and matrix exponentials run with autocast and TF32 disabled. Each accepted factorization reports symmetry defect, factorization status, pivot margins, a condition estimate, finite log determinant, and solve backward error. A failed partial factor is unusable. Higher-precision diagnostic reductions are declared, and silent dtype promotion is prohibited.

## Causal runtime separation

Training recognition, prior-predictive evaluation, and generation are separate compiled entry points. The training program may instantiate a filtering or smoothing recognition graph that observes the token being inferred. Evaluation still performs inference over the observed prefix; it excludes only the token currently being scored.

The predictive contract maintains a typed `PrefixFilterState` representing a normalized exact or declared approximate law over the complete latent history \(U_{t-1}\) after assimilating \(x_{<t}\). `predict` propagates that state through source and transition kernels to a `PredictiveState` over \(U_t\) without access to \(x_t\). `token_law` integrates the normalized categorical emission against this predictive state and returns \(p_\theta(x_t\mid x_{<t},\Gamma)\). The token is scored or sampled before `assimilate` updates the prefix filter with the newly observed occurrence. Every integration variable and reference measure is inherited from the compiled graph.

Generation uses the same `predict` and `token_law` operations, samples a token, and then assimilates that sampled occurrence. Teacher-forced evaluation assimilates the true token only after scoring it. A bootstrap particle filter, quadrature filter, or other scalable recursion may implement the prefix filter, but its estimator and approximation status are recorded separately from the normalized generative model. Removing the training recognition graph does not mean discarding latent inference from the prefix.

The artifact manifest records the information set of every runtime entry point. Mutating \(x_t\) or a suffix while holding \(x_{<t}\) fixed must leave the pre-assimilation predictive law unchanged. A model that can train but has no normalized predictive path cannot report perplexity. A recognition-only attention row may improve training inference, but it is not automatically available at generation time.

## Configuration architecture

Configuration uses nested frozen dataclasses with slots and a strict explicit codec. The root `ExperimentSpec` composes `ModelSpec`, `RecognitionSpec`, `ObjectiveSpec`, `InferenceSpec`, `NumericsSpec`, `TrainingSpec`, `DataSpec`, and `ArtifactSpec`. Components are selected through typed `ComponentRef` values containing a stable key and a local settings object validated by the selected descriptor. Unknown fields, ambiguous booleans, wrong scalar types, unsupported newer schema versions, and incompatible component references fail closed.

Three representations are retained. The authored spec records exactly what the experiment declared. The canonical spec applies explicit schema migrations and materializes defaults without changing semantics. The resolved spec records descriptor versions, derived dimensions, tensor layouts, selected kernels, compatibility results, and exactness. Compilation never mutates the authored object.

Fingerprints are separated by meaning. Model semantics, recognition semantics, objective, inference plan, numerical backend, data identity, training policy, and complete run each receive a named canonical digest. Operational fields are excluded only through a documented field policy, not name-prefix heuristics. A resumed run states which identities must match under its selected resume policy.

Click-to-run experiments remain ordinary small Python files with no command-line parser. Reusable constructors live under `experiments/components.py`, and experiment files compose frozen specs with `dataclasses.replace`. A multi-seed request is fully compiled and validated before any run directory is reserved. Each cell receives a fresh compiled program, seed, data seed, and run directory.

## Repository and dependency layout

The proposed repository is

```text
VFE_4.0/
├── pyproject.toml
├── src/vfe4/
│   ├── core/
│   │   ├── ids.py
│   │   ├── domains.py
│   │   ├── measures.py
│   │   ├── spaces.py
│   │   ├── exactness.py
│   │   └── errors.py
│   ├── catalog/
│   │   ├── descriptor.py
│   │   ├── builder.py
│   │   ├── snapshot.py
│   │   └── builtins.py
│   ├── families/
│   │   ├── protocols.py
│   │   ├── charts.py
│   │   ├── exponential.py
│   │   ├── mixture.py
│   │   ├── gaussian.py
│   │   └── categorical.py
│   ├── geometry/
│   │   ├── groups.py
│   │   ├── representations.py
│   │   ├── irreps.py
│   │   ├── fibers.py
│   │   └── connections.py
│   ├── factors/
│   │   ├── protocols.py
│   │   ├── initial.py
│   │   ├── source.py
│   │   ├── transition.py
│   │   └── emission.py
│   ├── graph/
│   │   ├── model.py
│   │   ├── recognition.py
│   │   ├── hierarchy.py
│   │   ├── validate.py
│   │   ├── compile.py
│   │   └── resolved.py
│   ├── objectives/
│   │   ├── evidence_elbo.py
│   │   ├── generalized_bayes.py
│   │   ├── divergences.py
│   │   └── ledger.py
│   ├── inference/
│   │   ├── protocols.py
│   │   ├── plan.py
│   │   ├── dense_discrete.py
│   │   ├── dense_gaussian.py
│   │   ├── sparse_gaussian.py
│   │   ├── coordinate.py
│   │   ├── generalized_em.py
│   │   └── stochastic_vi.py
│   ├── numerics/
│   │   ├── linear_algebra.py
│   │   ├── sparse.py
│   │   ├── estimators.py
│   │   └── diagnostics.py
│   ├── config/
│   │   ├── schema.py
│   │   ├── codec.py
│   │   ├── migrations.py
│   │   └── resolved.py
│   ├── runtime/
│   │   ├── build.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── generate.py
│   │   └── state.py
│   ├── artifacts/
│   │   ├── schema.py
│   │   ├── atomic.py
│   │   ├── fingerprints.py
│   │   ├── manifest.py
│   │   ├── checkpoint.py
│   │   └── migrations/
│   └── experiments/
│       ├── compose.py
│       ├── request.py
│       └── runner.py
├── tools/v3_import/
├── experiments/
└── tests/
```

The geometry package also owns a focused `morphisms.py` surface for fixed intertwiners, bilinear intertwiners, and covariant Hom fields. Domain-specific token observations, predictive-state types, source-prior implementations, and causal metrics live in a `language/` package between the compiled graph and runtime entry points rather than entering the general probability core.

The allowed construction stack runs from `core`, through families and geometry, factors, graph, objectives and compilation, inference, runtime, and experiments. Higher layers may depend inward on lower layers; lower layers never import outward. Artifact code depends only on stable serialized data-transfer objects and core identifiers, not live runtime classes. Families do not import inference. Backends do not construct semantic graphs. The V3 importer depends on V4 initialization schemas, but no V4 runtime package imports the importer.

## Manifests and checkpoints

Compilation emits a canonical build manifest containing the authored, canonical, and resolved specs; catalog snapshot and digest; graph variables and factors; reference measures; causal order; recognition information sets; source supports; representations, irreps, morphisms, and connections; ELBO ledger; parameter dependencies; Markov blankets; compatibility predicate results; selected kernels; exactness report; and graph and layout digests.

A run manifest adds data and tokenizer hashes, source-control identity and dirty-state fingerprint at start and finish, Python and PyTorch versions, CUDA and device information, dtype, determinism settings, seeds, parameter counts, compiled-manifest digest, checkpoint hashes, and lifecycle state. Metrics and events use append-only JSON Lines with periodic materialized summaries. Machine-readable records are authoritative for counts and final status.

Each checkpoint is an immutable directory published from a same-volume temporary sibling:

```text
checkpoints/step_000010000/
├── manifest.json
├── state.json
├── tensors.safetensors
└── hashes.json
```

The JSON state contains canonical scalar and structural data. The tensor archive contains model, recognition, optimizer, scaler, random-number-generator, and data-cursor tensors under a versioned naming layout. Durable artifacts contain no custom Python object pickle. The checkpoint container, configuration, graph, parameter layout, catalog API, and component implementations have independent versions.

Loading validates all file hashes, schema versions, graph and catalog digests, component availability, parameter names, shapes, dtypes, optimizer groups, data identity, and cursor identity before mutating any live state. `exact_continuation` requires identity of all semantic and trajectory-bearing fields and restores the full training state. `weights_only` imports compatible parameter blocks and resets trajectory state. `fork_training` permits only a declared allowlist of training-policy changes and records them. `initialization_import` consumes a converter report and never claims exact continuation.

Schema migrations are pure sequential transformations over immutable metadata fixtures. A semantic conversion creates a new artifact and a conversion report; it does not rewrite or silently reinterpret the original checkpoint.

## One-way V3 import

VFE 4.0 has no runtime dependency on V3. The standalone `tools/v3_import` package reads frozen V3 JSON and tensor-state layouts without importing `vfe3`. It maps compatible embedding tables, prior tables, frame parameters, decoder parameters, dimensions, and selected metadata into a V4 `InitializationBundle`.

The converter cannot infer missing V4 probability structure. It cannot manufacture joint precision cross-blocks, normalized transition kernels, probabilistic source variables, structured recognition factors, hierarchy latents, or update certificates from V3 arrays. Every field is reported as imported, transformed, skipped, or reinitialized, with source and destination hashes. The result initializes a new V4 run and is never accepted as a V4 exact-resume checkpoint. Frozen V3 fixtures define the supported one-way schema.

## Extension recipes

Each extension consists of a local frozen spec, a protocol implementation, a typed capability record, compatibility predicates, contract tests, and one descriptor registration during catalog construction. No training, evaluation, generation, or shared model call site changes.

| Extension | New implementation surface | Compiler obligations |
|---|---|---|
| Statistical family | `Law`, optional `ExponentialFamily`, and optional `LegendreDualCapable`, with support, density, entropy and sampling where available | Reference measure, minimality or quotient status, chart requirements, factor compatibility, transport closure, estimator availability |
| Mixture | Labeled or marginalized composer over component references | Label semantics, support, entropy method, exact marginalization or projection status |
| Generative or recognition factor | Normalized conditional kernel or recognition disintegration with local settings and parameter dependencies | Child normalization, causal and information scope, support, reference measure, ledger coverage, Markov-blanket closure |
| Gauge group | Group operations, Lie algebra and parameterizations | Representation and connection compatibility, determinant and component constraints |
| Representation or irrep tower | Carrier action, dual action, decomposition and commutant metadata | Fiber dimensions, homomorphism, invariant layouts, morphism compatibility |
| Cross-fiber morphism | Fixed intertwiner or covariant Hom-field implementation | Source and target fibers, transformation law, parameter ownership |
| Connection | Frame-derived or independent-edge connection implementation | Group and representation compatibility, orientation reversal, loop semantics, curvature status |
| Divergence | Diagnostic, projection, consensus, or generalized-Bayes implementation | Role compatibility; ordinary evidence objective rejects unauthorized substitution |
| Source prior | Normalized categorical kernel and positive-support contract | Causal conditioning, target blindness, exact normalizer, recognition support inclusion |
| Hierarchy profile | Pure graph builder that emits variables and normalized factors | Complete normalization, acyclicity, bridge kernels, no implicit tuple handoff |
| Inference backend | Graph-to-update-plan compiler and update certificates | Domain, factor, family, source, layout, dtype, and exactness compatibility |
| Numerical backend | Solve, log-determinant, marginal, sample, and estimator kernels | Layout preservation, tolerance policy, oracle comparison, approximation declaration |

## Verification architecture

Unit tests verify each law, chart conversion, factor normalizer, group operation, representation, connection, morphism, and numerical kernel. Catalog contract tests verify duplicate rejection, explicit replacement, freeze immutability, deterministic digesting, local builder isolation, capability serialization, and compatibility-predicate reporting.

Probability-oracle tests use finite discrete enumeration and tiny dense Gaussian systems to verify joint normalization, recognition normalization, evidence decomposition, ledger equality, source posteriors, expected categorical likelihood, and exact coordinate updates. Structured-posterior tests compare dense joint information form with chain, tree, banded, and sparse lowerings. Mixture tests distinguish labeled entropy, marginalized density, and moment projection. Gauge tests verify group composition, representation homomorphism, covariant factor values, intertwiners, Hom-field transformation, irrep layouts, and loop behavior for each connection kind.

Negative compatibility tests are first-class. They reject a diagonal Gaussian under a representation that does not preserve diagonal structure, a marginalized mixture without an entropy estimator, an exact-coordinate backend applied to a nonconjugate blanket, source posterior support outside the generative prior, a fixed cross-channel map that is not an intertwiner, a geometry-dependent factor without a geometry reference measure, and a generalized objective presented as evidence.

Checkpoint tests cover exact continuation, corrupted hashes, missing components, graph drift, parameter-layout drift, data drift, weights-only import, forked training, and every schema migration fixture. V3 interoperation tests use checked-in frozen artifacts. CUDA tests compare selected float32 kernels with float64 CPU oracles and exercise memory-residency and sparse-layout contracts on the RTX 5090. Runtime architecture tests verify that compiled update signatures contain no configuration or catalog object and that tests never mutate global registries.

The sparse-production gate includes the white paper's explicit allocation oracle at \(T=128\) and total latent block width \(K=20\). It traces construction, solve, log determinant, requested marginal extraction, sampling, backward, diagnostics, and reporting. Any global \((TK)\times(TK)\) covariance, inverse, or equivalent quadratic buffer is forbidden on this route, including compatibility and diagnostic materializers. Backend-specific peak-memory bounds, symbolic fill, and allocation shapes are asserted from machine-readable traces.

## Phased acceptance gates

The phases below are architectural gates. The separate implementation plan may subdivide them, but no later gate may weaken an earlier exact contract.

| Gate | Deliverable | Exit condition |
|---|---|---|
| 0. Semantic kernel | Measures, law protocols, graph IR, strict config, frozen catalogs, exactness records | Invalid normalized models and incompatible components fail before tensor allocation |
| 1. Dense reference oracles | CPU float64 categorical and Gaussian reference evaluators | Finite discrete enumeration and analytic Gaussian tests verify normalization and the evidence identity; nonanalytic continuous expectations carry a separate evaluation record |
| 2. Minimal causal language model | `pq` hierarchy, categorical emission, filtering recognition, predictive evaluation | Training posterior may observe the target; evaluation cannot; CE appears exactly as the observation sector |
| 3. Structured inference | Joint Gaussian information form, source variables, ledger-derived blankets and update certificates | Dense structured results match the applicable reference oracles and no mean-field assumption is hidden |
| 4. Early predictive smoke falsifier | Matched deterministic-state CE baseline and minimal dense `pq` language model on a bounded corpus | The target-blind prior-predictive path is runnable and its NLL, compute, collapse, and estimator error are measured before production or geometry infrastructure is added |
| 5. CUDA dense and block/banded production | CUDA float32 dense plus structure-certified block and banded lowerings, followed by the same matched predictive pilot | Each kernel matches its float64 reference within an operation-specific tolerance; the pilot records device-specific NLL, compute, memory, collapse, and evaluation error |
| 6. Model hierarchy and geometry | Optional `hspq`, distinct fibers, normalized bridge, group, representations, morphisms, deterministic connection | Cross-channel dimensions are typed and all selected factors pass covariance tests |
| 7. Sparse and matrix-free production | Sparse-direct CPU and opt-in matrix-free CUDA lowerings | Sparse fill and selected marginals are audited; every iterative or stochastic operation reports its own guarantee |
| 8. Extension proof | One factor, family, mixture, group, representation, connection, morphism, divergence, hierarchy profile, inference plan, source prior, and numerical backend extension | Each extension is added through descriptors and contract tests without modifying shared runtime call sites |
| 9. Full baseline experiments | Matched latent-ELBO, structure, hierarchy, geometry, source-prior, and backend comparisons | Multi-seed artifacts report held-out predictive metrics, ELBO sectors, exactness, compute, and uncertainty |

The semantically exact model and evidence-identity path remains executable at every gate after Gate 1. Analytic evaluation is claimed only for the discrete and conjugate Gaussian reference cases that possess it. Approximate routes are opt-in and cannot reuse an analytic or structure-equivalent configuration key or status label.

## Baselines and falsifiable hypotheses

The empirical program must distinguish a valid probabilistic construction from a useful language model. All comparisons use matched data, tokenizer, context, seed schedule, training budget, parameter accounting, and predictive evaluation. The CE-only deterministic baseline remains necessary because the latent model contains CE rather than replacing it. B0 is the deterministic-state restriction of the same causal transition and normalized emission interfaces: it has no variational posterior, latent entropy, or latent KL, and it uses raw typed parameter tensors rather than neural modules. This makes it a nested observation-only control instead of an unrelated transformer baseline. Any parameter-count or compute mismatch is reported rather than hidden behind the word matched.

| Identifier | Comparison | Hypothesis and falsifier |
|---|---|---|
| B0 | Deterministic CE-only language model | Establishes the attainable NLL, perplexity, wall time, and memory floor; a VFE model that cannot approach it has not justified its latent overhead |
| H1 | Minimal `pq` ELBO versus B0 | Latent uncertainty improves calibration, representation, or data efficiency without unacceptable NLL degradation; posterior collapse or uniformly worse predictive metrics falsify the useful-latent claim |
| H2 | Structured precision versus matched mean-field recognition | Cross-token or cross-channel dependence tightens the ELBO and improves prediction where correlations matter; no repeatable ELBO-gap or predictive benefit falsifies the added structure at the tested scale |
| H3a | Information-form versus moment-form solver parameterization of the same Gaussian law | Information-form assembly improves conditioning or structured-solve efficiency without changing represented probabilities; matched stability and cost falsify the storage/solver advantage |
| H3b | Fisher-natural versus Euclidean optimization of the same parameter block | A registered Fisher-preconditioned step improves convergence or robustness at matched objective evaluations and compute; no repeatable difference falsifies the optimization advantage |
| H4 | Explicit source posterior versus fixed or uniform source routing | Probabilistic routing improves context use or sample efficiency; identical source utilization and predictive behavior falsify the routing claim |
| H5 | Trivial geometry, fixed transport, and learned covariant geometry | Gauge-compatible structure improves parameter efficiency or robustness while preserving transformation tests; gains that disappear under matched capacity or violations of equivariance falsify the claim |
| H6 | Dense float64 Gaussian reference, structure-preserving direct, and approximate matrix-free backends | Structured backends preserve the applicable objective terms at lower cost, while matrix-free error tracks declared residuals; unexplained objective drift falsifies backend fidelity |
| H7 | `pq` versus normalized `hspq` hierarchy | The model channel supplies predictive information beyond a larger state-only control; no matched benefit or an ignored model posterior falsifies the hierarchy claim |

Reports include predictive NLL and perplexity, ELBO and its ledger sectors, posterior-prior KL, source entropy and utilization, effective rank or precision diagnostics, calibration, gradient and update acceptance diagnostics, wall time, memory, and variation across seeds. The design makes no claim that a tighter ELBO necessarily yields lower perplexity or that gauge structure necessarily improves language modeling. Those are empirical hypotheses.

## V1 non-goals

V1 does not attempt a universal probabilistic-programming language, arbitrary undirected factor graphs, external package entry-point discovery, runtime plugin loading, hot replacement, a remote component registry, automatic symbolic conjugacy proofs, automatic graph rewriting, arbitrary recursively nested mixtures, every possible exponential or mixture family, or distributed training.

V1 uses explicit non-neural generative kernels and per-example iterative variational inference. It does not add an MLP, transformer block, or learned neural recognition network. A future amortized initializer would be a separate recognition component whose output is refined and scored by the same compiled ELBO; it is not required to establish the architecture.

V1 does not integrate over latent gauge geometry, claim a proper invariant probability measure on noncompact \(\mathrm{GL}(K)\), infer quotient spaces automatically, or interpret a frame-derived pure-gauge connection as nonzero curvature. Those directions require separate normalized models and measure audits.

V1 does not replace categorical cross-entropy, treat alternative divergences as interchangeable ELBO terms, identify source softmax with scaled dot-product attention, prove reduction to a conventional transformer, or claim that every refinement sweep is a coordinate update. It also does not provide exact V3 checkpoint continuation. V3 artifacts may initialize compatible V4 parameter blocks only through the audited one-way converter.

V1 does not promise that every combination present in the catalogs is compatible. Modularity means that new components can be added without changing shared call sites and that valid combinations compile. It does not mean that mathematics permits every family, representation, factor, hierarchy, and backend cross-product.

## Definition of architectural completion

The architecture is implemented only when a click-run experiment can construct an instance-scoped catalog, compile immutable generative and recognition graphs, emit a complete compatibility and exactness manifest, execute the finite-discrete and conjugate-Gaussian float64 reference oracles, train and evaluate a float32 CUDA model without runtime configuration dispatch, save and exactly resume a versioned checkpoint, and reproduce the ordinary ELBO from its ledger within the declared evaluation contract. The extension proof covers the categories named in Gate 8 without changing shared runtime call sites, and every approximation remains opt-in and labeled.

Until those conditions are met, VFE 4.0 remains a design and research program. This document does not claim that implementation has started or that any proposed hypothesis has experimental support.
