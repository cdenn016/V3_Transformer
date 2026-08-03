"""Structural and semantic contract checks for rigorous-theory-search."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SKILL_ROOT = ROOT / "agent_tooling" / "rigorous_theory_search" / "skill"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
OPENAI_YAML_PATH = SKILL_ROOT / "agents" / "openai.yaml"
REFERENCE_NAMES = (
    "problem-contract.md",
    "portfolio-search.md",
    "proof-obligations.md",
    "physics-information-audit.md",
    "renormalization-effective-theory.md",
    "adversarial-verification.md",
    "runtime-adapters.md",
    "output-contract.md",
)
TEMPLATE_NAMES = (
    "problem-contract.json",
    "approach-registry.json",
    "claim-ledger.json",
    "dependency-dag.json",
    "counterexample-register.md",
    "construction-or-strongest-theorem.md",
    "adversarial-report.json",
    "release.json",
    "final-report.md",
)
SCRIPT_NAMES = ("scaffold_run.py", "validate_run.py")
ROUTINE_NEGATIVE_TRIGGER_CASES = (
    ("routine ELBO derivation", "routine ELBO derivations"),
    ("standard block-spin RG summary", "routine RG transformations"),
)
FROZEN_CORPUS_HASHES = {
    "pressure-evals-v1.json": (
        "2085b8940264ecb93bf456e47d3ef67744ed9b52dd6934e02db4148f8f624019"
    ),
    "trigger-corpus-v1.json": (
        "57693b7e1b834216ef7205180dfeb74b71f693db895183480737ad0f96c0f392"
    ),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _skill() -> str:
    return _read(SKILL_PATH)


def _reference(name: str) -> str:
    return _read(SKILL_ROOT / "references" / name)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_frontmatter_has_only_portable_identity_and_trigger_description() -> None:
    text = _skill()
    assert text.startswith("---\n")
    frontmatter, _ = text[len("---\n"):].split("\n---\n", 1)
    parsed = yaml.safe_load(frontmatter)
    assert set(parsed) == {"name", "description"}
    assert parsed["name"] == "rigorous-theory-search"
    description = parsed["description"]
    assert description.startswith("Use when")
    for trigger in (
        "research-level", "conjectures", "unsolved problems", "full proofs", "derivations",
        "complete", "construction",
        "effective theories", "gauge", "information geometry", "ELBO", "VFE",
        "renormalization",
    ):
        assert trigger.lower() in description.lower()
    for workflow_word in ("workflow", "orchestrate", "scaffold", "validate"):
        assert workflow_word not in description.lower()


def test_openai_interface_metadata_matches_the_skill() -> None:
    assert OPENAI_YAML_PATH.is_file()
    payload = yaml.safe_load(OPENAI_YAML_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {"interface"}
    interface = payload["interface"]
    assert set(interface) == {
        "display_name",
        "short_description",
        "default_prompt",
    }
    assert interface["display_name"] == "Rigorous Theory Search"
    assert 25 <= len(interface["short_description"]) <= 64
    assert "$rigorous-theory-search" in interface["default_prompt"]


def test_frontmatter_and_scope_exclude_routine_elbo_and_rg_prompts() -> None:
    text = _skill()
    _, frontmatter, body = text.split("---", 2)
    description = yaml.safe_load(frontmatter)["description"]
    scope = f"{description}\n{body}".lower()
    assert "do not use" in scope
    for _prompt_fixture, required_exclusion in ROUTINE_NEGATIVE_TRIGGER_CASES:
        assert required_exclusion.lower() in scope


def test_frozen_evaluation_corpora_are_unchanged() -> None:
    eval_root = SKILL_ROOT / "evals"
    for name, expected in FROZEN_CORPUS_HASHES.items():
        assert hashlib.sha256((eval_root / name).read_bytes()).hexdigest() == expected


def test_standard_evals_adapter_preserves_the_frozen_pressure_cases() -> None:
    eval_root = SKILL_ROOT / "evals"
    frozen = json.loads(_read(eval_root / "pressure-evals-v1.json"))
    standard = json.loads(_read(eval_root / "evals.json"))

    assert set(standard) == {"skill_name", "evals"}
    assert standard["skill_name"] == "rigorous-theory-search"
    assert len(standard["evals"]) == len(frozen["cases"])
    for index, (adapted, source) in enumerate(
        zip(standard["evals"], frozen["cases"], strict=True),
        start=1,
    ):
        assert adapted == {
            "id": index,
            "prompt": source["prompt"],
            "expected_output": " ".join(source["expected_behaviors"]),
            "files": [],
            "expectations": source["expected_behaviors"] + [
                f"Output avoids forbidden behavior: {behavior}"
                for behavior in source["forbidden_behaviors"]
            ],
        }


def test_standard_trigger_adapter_preserves_the_frozen_trigger_cases() -> None:
    eval_root = SKILL_ROOT / "evals"
    frozen = json.loads(_read(eval_root / "trigger-corpus-v1.json"))
    standard = json.loads(_read(eval_root / "trigger-evals.json"))

    assert isinstance(standard, list)
    assert standard == [
        {
            "query": source["prompt"],
            "should_trigger": source["should_trigger"],
        }
        for source in frozen["cases"]
    ]


def test_frozen_evaluation_json_is_checked_out_with_lf_git_attribute() -> None:
    attributes_path = ROOT / ".gitattributes"
    assert attributes_path.is_file()
    rules = {
        line.strip()
        for line in attributes_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "agent_tooling/rigorous_theory_search/skill/evals/*.json text eol=lf" in rules


def test_skill_is_compact_progressive_orchestration() -> None:
    text = _skill()
    for heading in (
        "## Load the protocol progressively",
        "## Discovery and certification",
        "## Required lifecycle",
        "## Search-pressure safeguards",
        "## Release",
    ):
        assert heading in text
    for phrase in (
        "discovery proposes; certification proves",
        "problem-contract.target.search_priors",
        "SEARCH_PRIOR_AFFIRMATIVE",
        "materially new mechanism",
        "dynamic mechanism-diverse portfolio",
        "acyclic dependency DAG",
        "mature routes",
        "interface claim",
        "exact versus truncated",
        "independent reconstruction",
        "oracle erasure",
        "sequential fallback",
    ):
        assert phrase.lower() in text.lower()


def test_release_states_and_evidence_rules_are_exact() -> None:
    text = _skill()
    for status in ("COMPLETE_AFFIRMATIVE", "COMPLETE_NEGATIVE", "INCONCLUSIVE"):
        assert status in text
    for evidence in ("DERIVATION", "FORMAL_PROOF", "APPLICABLE_THEOREM"):
        assert evidence in text
    for phrase in (
        "one valid counterexample",
        "nonexistence proof",
        "every terminal status",
        "cannot establish mathematical truth",
    ):
        assert phrase.lower() in text.lower()


def test_baseline_safeguards_report_observed_control_without_invented_failures() -> None:
    text = _skill().lower()
    for phrase in (
        "no false-closure rationale",
        "forty final answers",
        "all 240 graded checks passed",
        "five constructive-theorem runs",
        "affirmative pressure",
        "finite-to-universal",
        "local or gauge-fixed to global",
        "representation, pullback, or curve parameter",
        "projected rg",
        "defensive nonproof boilerplate",
    ):
        assert phrase in text


def test_all_references_templates_and_scripts_are_real_direct_links() -> None:
    text = _skill()
    for name in REFERENCE_NAMES:
        relative = f"references/{name}"
        assert relative in text
        assert (SKILL_ROOT / relative).is_file()
    for name in TEMPLATE_NAMES:
        relative = f"assets/templates/{name}"
        assert relative in text
        assert (SKILL_ROOT / relative).is_file()
    for name in SCRIPT_NAMES:
        relative = f"scripts/{name}"
        assert relative in text
        assert (SKILL_ROOT / relative).is_file()


def test_problem_contract_freezes_typed_scope_and_contains_search_prior() -> None:
    text = _reference("problem-contract.md").lower()
    for phrase in (
        "exact quantifiers", "domains and codomains", "regularity", "measures",
        "boundary conditions", "symmetry actions", "equivalence relation",
        "modeling postulates", "search priors", "falsification criterion",
        "sha-256", "nonempty", "absolute continuity", "literature policy",
        "priority claim",
    ):
        assert phrase in text
    assert "problem-contract.target.search_priors" in text


def test_portfolio_uses_mechanism_novelty_maturity_and_retirement() -> None:
    text = _reference("portfolio-search.md")
    for phrase in (
        "mechanism, never wording", "Do not broadcast a favorite route",
        "LEMMA | CONSTRUCTION | COUNTEREXAMPLE | OBSTRUCTION", "family ID",
        "exact statement or object", "target obligations", "derivation path",
        "failure test", "open gap", "novelty fingerprint",
        "representation + mechanism + invariant/obstruction + bridge + assumption set + target obligations",
        "retired", "refuted", "mature", "new family ID", "interface claim",
        "inherit no verification state",
    ):
        assert phrase.lower() in text.lower()


def test_proof_obligations_separate_proof_computation_and_quantifier_logic() -> None:
    text = _reference("proof-obligations.md")
    for phrase in (
        "One exact claim per ledger entry", "from -> to", "from depends on to",
        "normalization", "domination", "integrability", "analytic interchanges",
        "APPLICABLE_THEOREM", "primary source", "hypothesis-by-hypothesis",
        "Numerical", "finite", "symbolic", "agent agreement",
        "One valid counterexample", "failure to find a witness",
        "scope-matched nonexistence proof", "acyclic", "formalization",
        "cocycles", "uniqueness modulo", "bridge premises",
    ):
        assert phrase.lower() in text.lower()


def test_proof_obligations_cover_every_conjunct_after_negative_closure() -> None:
    text = _reference("proof-obligations.md").lower()
    for phrase in (
        "compound target",
        "atomize",
        "each conjunct",
        "conditional theorem",
        "missing assumptions",
    ):
        assert phrase in text


def test_physics_information_audit_carries_refined_geometry_and_elbo_obligations() -> None:
    text = _reference("physics-information-audit.md")
    compact = _squash(text)
    for phrase in (
        r"D^\omega s=\operatorname{ver}^\omega\circ Ts",
        r"\operatorname{rad}h^\omega=\ker D^\omega s",
        r"\operatorname{rank}h^\omega=\operatorname{rank}D^\omega s",
        r"c_{\alpha\beta}+c_{\beta\gamma}+c_{\gamma\alpha}=0",
        r"c_{\alpha\beta}=f_\beta-f_\alpha",
        r"\alpha=d\tau",
    ):
        assert _squash(phrase) in compact
    for phrase in (
        "principal bundle", "associated statistical bundles", "positive definite",
        "square-integrable", "global-function freedom",
        "involutive radical", "regular leaf space", "basic tensor",
        "vertical curve", "horizontal", "orientation-preserving",
        "closed clock one-form", "zero periods", "operational bridge",
        "KL disintegration", "total correlation", "normalization and uniqueness",
        "representational equivalence", "ontological identity",
    ):
        assert phrase.lower() in text.lower()
    for term in ("KL(Q\\|P)", "\\sum_i KL(Q_i\\|P_i)", "TC(Q)", "\\prod_i p_i/p"):
        assert _squash(term) in compact
    for typed_curve in (
        r"\pi_E:E\to B",
        r"E_x=\pi_E^{-1}(x)",
        r"\gamma:J\to E_x",
    ):
        assert _squash(typed_curve) in compact


def test_physics_audit_has_explicit_curve_duration_and_channel_equivalence_gates() -> None:
    text = _reference("physics-information-audit.md")
    compact = _squash(text)
    for equation in (
        r"\eta=s\circ c:J\to E",
        r"\xi^\omega=\operatorname{ver}^\omega(\dot\eta)=D^\omega s(\dot c)",
        r"v_F^{\mathrm{vert}}(\lambda)=\sqrt{I_{\gamma(\lambda)}(\dot\gamma(\lambda),\dot\gamma(\lambda))}",
        r"v_F^\omega(\lambda)=\sqrt{I_{\eta(\lambda)}(\xi^\omega(\lambda),\xi^\omega(\lambda))}",
        r"\tau_F^\star(\lambda)=\tau_0+\int_{\lambda_0}^{\lambda}v_F^\star(u)\,du",
        r"p_y(e\mid r):=\frac{p(r,e,y)}{p_{R,Y}(r,y)}",
    ):
        assert _squash(equation) in compact
    for phrase in (
        "smooth statistical stratum",
        "absolutely continuous",
        "positive accumulated length on every nontrivial subinterval",
        "separate pathwise proof",
        "connection-relative",
        "fisher-null",
        "full joint law",
        "conditional independences",
        "all likelihood factors",
        "absolute continuity",
        "standard borel",
        "pointwise density ratio",
        "almost every observation",
        "conditional-posterior equality",
        "agent-only ontology",
        "modeling or operational bridge",
        "p-completed sigma-algebras",
        "interventional completeness",
    ):
        assert phrase in text.lower()
    assert "singular-stratum crossing prevents" not in text.lower()


def test_fixed_observation_equivalence_requires_slice_wise_representatives() -> None:
    def density(y: float) -> float:
        return 1.0

    def null_slice_modified_density(y: float) -> float:
        return 2.0 if y == 0.0 else 1.0

    # The representatives differ only on a Lebesgue-null singleton, yet their
    # fixed-observation log likelihoods differ there.
    assert all(
        density(y) == null_slice_modified_density(y)
        for y in (0.25, 0.5, 0.75, 1.0)
    )
    assert (
        math.log(null_slice_modified_density(0.0))
        - math.log(density(0.0))
        == math.log(2.0)
    )

    text = _reference("physics-information-audit.md").lower()
    for phrase in (
        "chosen pointwise density representatives",
        "slice-wise equality",
        "every specified observation",
        "slice zero sets",
        "version-relative",
    ):
        assert phrase in text


def test_renormalization_audit_types_exact_maps_flows_and_fixed_objects() -> None:
    text = _reference("renormalization-effective-theory.md")
    for phrase in (
        "exact coarse law", "coarse-graining Markov kernel", "contraction identity",
        "generated operator", "higher-body", "nonlocal", "truncation residual",
        "lumpability", "cross-scale coherence", "scheme change",
        "finite-difference beta", "cocycle", "nonautonomous", "semigroup",
        "bundle morphism", "covering a base map", "compatible sections",
        "compatible connections", "invariant sections", "invariant measures",
        "attractors", "ordinary zeros", "fixed points modulo", "injective",
        "common linear or affine structure",
    ):
        assert phrase.lower() in text.lower()
    assert _squash("i_\\ell:\\mathcal A_\\ell\\to\\mathcal A_*") in _squash(text)
    assert text.index("i_\\ell") < text.lower().index("finite-difference beta")
    for phrase in (
        "define the contracted functional",
        "q-independent constant",
        "regular disintegration",
        "without such maps, the levels are unrelated",
        "identity and ordered composition laws",
        "thin category",
        "cocycle additionally requires",
        "scale-translation invariance",
        "one-parameter semigroup",
    ):
        assert phrase in text.lower()
    for equation in (
        r"\mathcal F'_C[Q]:=\inf_{q:C_\#q=Q}\mathcal F[q]",
        r"P'\ll\mu'",
        r"\mathcal F[q]=\operatorname{KL}(q\|P)+c",
        r"J(dz,dZ):=P(dz)C(dZ\mid z)",
        r"(Ra)(Z)=\mathbb E_J[a(z)\mid Z]",
        r"\langle a,Uf\rangle_{L^2(P)}=\langle Ra,f\rangle_{L^2(C_\#P)}",
        r"\mathcal R:\mathcal A\to\mathcal A",
        r"\iota:\mathcal T\hookrightarrow\mathcal A",
        r"\Pi:\mathcal A\to\mathcal T",
        r"\Pi\iota=\operatorname{Id}_{\mathcal T}",
        r"\epsilon_S:=\mathcal R(\iota S)-\iota\Pi\mathcal R(\iota S)\in\mathcal A",
        r"C_{n\leftarrow n}=\operatorname{Id}_{\mathcal X_n}",
        r"\Phi(t+s,\omega)=\Phi(t,\theta_s\omega)\circ\Phi(s,\omega)",
        r"T_{a+b}=T_a\circ T_b",
        r"T_0=\operatorname{Id}",
    ):
        assert _squash(equation) in _squash(text)


def test_adversarial_protocol_reconstructs_and_adjudicates_without_voting() -> None:
    text = _reference("adversarial-verification.md").lower()
    for phrase in (
        "isolated first-pass falsifier", "defender", "falsification conditions",
        "independent reconstruction", "evidence-weighted adjudication",
        "not voting", "oracle erasure", "rechecks every ancestor",
        "does not prove the theorem",
    ):
        assert phrase in text


def test_runtime_names_are_confined_to_runtime_adapter() -> None:
    runtime = _reference("runtime-adapters.md")
    for phrase in (
        "Claude Code", "Codex", "spawn_agent", "send_message", "followup_task",
        "wait_agent", "shared-agent", "sequential fallback",
    ):
        assert phrase.lower() in runtime.lower()
    prohibited = re.compile(
        r"\b(?:Claude(?: Code)?|Codex|spawn_agent|send_message|followup_task|wait_agent)\b",
        re.IGNORECASE,
    )
    for path in (SKILL_PATH, *(SKILL_ROOT / "references" / name for name in REFERENCE_NAMES if name != "runtime-adapters.md")):
        assert prohibited.search(_read(path)) is None, path


def test_output_contract_names_durable_artifacts_and_validator_limit() -> None:
    text = _reference("output-contract.md")
    for phrase in (
        "docs/derivations/YYYY-MM-DD-<slug>/",
        "rigorous-theory-search/YYYY-MM-DD-<slug>/",
        "one contract ID", "rigorous-theory-search/v1", "safe relative",
        "computed SHA-256", "terminal_status: null", "prior leakage",
        "hash syntax and path safety", "does not verify file bytes",
        "## Frozen contract", "## Terminal status", "## Certificate",
        "## Strongest verified result", "## Dependency closure",
        "## Independent reconstruction", "## Oracle erasure",
        "## Unresolved obligations", "## Scope and limitations",
    ):
        assert phrase.lower() in text.lower()
    for name in TEMPLATE_NAMES:
        artifact = name.replace("problem-contract", "problem-contract")
        assert artifact in text


def test_touched_skill_prose_uses_american_english_without_banned_filler() -> None:
    paths = [SKILL_PATH, *(SKILL_ROOT / "references" / name for name in REFERENCE_NAMES)]
    corpus = "\n".join(_read(path) for path in paths).lower()
    british = (
        "behaviour", "colour", "centre", "modelling", "normalise", "optimise",
        "factorise", "fibre", "labelled", "licence", "defence", "analyse",
        "recognise", "summarise", "programme",
    )
    filler = (
        "key insight", "crucially", "critically", "notably", "importantly",
        "it's worth noting", "fundamentally", "leverages", "underscores",
    )
    for term in (*british, *filler):
        assert re.search(rf"\b{re.escape(term)}\b", corpus) is None, term
