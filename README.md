# V3_Transformer (VFE_3.0)

A clean-room rebuild of the gauge-theoretic variational-free-energy transformer. The
defining constraint is that there are no neural networks in the pure path: no
`nn.Linear`, no MLP, no activations. All representational capacity comes from
iteratively minimizing a variational free energy `F` over a per-token Gaussian belief
tuple `(mu, Sigma, phi)`, and the only learnable objects are the prior tables in the
`PriorBank`. Backpropagation is still used, but only to learn those priors: the loss
flows backward through the unrolled belief inference. The theory is developed in
`Manuscripts-Theory/` (`GL(K)_attention.tex`, `GL(K)_supplementary.tex`,
`Participatory_it_from_bit.tex`); design and audit notes live under `docs/`.

The codebase is built bottom-up with every kernel numerically pinned to the earlier
VFE_2.0 reference by golden tests, and every modeling choice sits behind a
config-selected registry so that variants slot in by registration rather than by editing
call sites.

## The core idea

A transformer layer is replaced by an inference step. Each token `i` carries a Gaussian
belief `q_i = N(mu_i, Sigma_i)` together with a gauge frame `phi_i` (an element of a Lie
algebra). A forward pass encodes tokens into their prior beliefs, runs an iterative
descent on a free energy that couples every belief to a learned prior and to every other
belief through gauge-covariant transport, and then decodes the converged belief by
measuring its KL divergence to each vocabulary prior. Attention is not a learned bilinear
map; it is the softmax that arises as the stationary point of the belief-coupling block of
`F`. "Heads" are the irreducible blocks of the gauge group acting on the belief.

The mechanism the code actually computes at decode time is a KL-nearest-prior readout:
the next-token logits are `-KL(q_i || pi_v)/tau_eff` over the vocabulary `v`. The richer
free-energy-principle reading — beliefs as a community of agents, the next token as the
agent that most lowers free energy — is an interpretation of this mechanism, not a
separate computation in the code.

## Architecture

The model is `encode -> E-step stack -> (optional head mix) -> (optional norm) -> decode
-> cross-entropy`. The diagram shows the three nested loops that give the model its depth:
the inner E-step iteration (run `T = n_e_steps` times), the per-block loop (`L = n_layers`
blocks, with a belief-to-prior handoff between them), and the outer M-step that
backpropagates the cross-entropy through the entire unrolled inference into the PriorBank
tables.

```
 token_ids (B, N)
      |
      v
 +-----------------------------------------------------------------------------+
 | PriorBank.encode : table lookup  token -> (mu, Sigma, phi),  q := p          |
 +-----------------------------------------------------------------------------+
      |
      |  + positional phi  (optional BCH / RoPE gauge composition)
      v
 ====== E-STEP STACK : L blocks ==============================================
 |                                                                            |
 |   per block: iterate the E-step  T = n_e_steps  times                      |
 |                                                                            |
 |   .------------------- inner E-step iteration (x T) -------------------.    |
 |   |                                                                    |    |
 |   |  transport     Omega_ij = exp(phi_i . G) exp(-phi_j . G)           |    |
 |   |  energy        E_ij = D( q_i || Omega_ij q_j )      (per irrep blk)|    |
 |   |  attention     beta_ij = softmax_j( log_prior_ij - E_ij / tau )    |    |
 |   |  belief grad   dF/d(mu, Sigma)   (envelope kernel / autograd)      |    |
 |   |  Fisher        natural-gradient precondition                       |    |
 |   |  retract       mu  <- mu - lr * nat_mu        (Euclidean)          |    |
 |   |                Sigma <- SPD affine-invariant exp-map step          |    |
 |   |                phi <- Lie retraction of preconditioned dF/dphi     |    |
 |   '--------------------------------------------------------------------'    |
 |                                                                            |
 |   handoff to next block:  mu_p <- (1-rho) mu_p + rho mu_q                   |
 |                           Sigma_p <- (1-rho_s) Sigma_p + rho_s Sigma_q      |
 ============================================================================
      |
      v   converged belief (mu*, Sigma*, phi*)
 +-----------------------------------------------------------------------------+
 | PriorBank.decode :  logits_{i,v} = -KL( q_i* || pi_v ) / tau_eff   (over V)  |
 +-----------------------------------------------------------------------------+
      |
      v
 cross-entropy  vs  next-token targets
      |
      '----------------------------------------------.
                                                     |  M-step: loss.backward()
                                                     |  through the UNROLLED E-step
                                                     v
 +-----------------------------------------------------------------------------+
 | AdamW (per-group LRs) updates ONLY the PriorBank tables:                     |
 |   mu_embed (V,K), sigma_log_embed (V,K), phi_embed (V,n_gen), decode_log_scale|
 +-----------------------------------------------------------------------------+
```

The only parameters in the pure path are the prior tables: a mean table `mu_embed`
`(V, K)`, a log-variance table `sigma_log_embed` `(V, K)`, a gauge-frame table
`phi_embed` `(V, n_gen)`, and a scalar decode temperature `decode_log_scale`. There is no
attention projection, no feedforward block, and no output embedding matrix on this path.
Everything else in the model is a fixed, parameter-free numerical operator.

### Encode

`PriorBank.encode` looks up the per-token prior `pi = N(mu_v, exp(sigma_log_v))` with
gauge frame `phi_v` and uses it as the initial belief, so inference begins at `q = p`.
The diagonal family stores `Sigma` as a variance vector; the full-covariance family
embeds the same per-token variances as a diagonal SPD matrix that the full E-step then
evolves off-diagonal mass into.

### The E-step

Each block runs `n_e_steps` iterations of `e_step_iteration`, a natural-gradient descent
on `F` over the belief. One iteration, with all positions updated in parallel
(mean-field), proceeds as follows. The gauge frames define a pairwise transport operator
`Omega_ij = exp(phi_i . G) exp(-phi_j . G)`, where `G` are the Lie-algebra generators of
the gauge group; this is the flat Regime-I cocycle, the pure path. The transport acts on a
belief by the GL(K) congruence (sandwich) action `mu -> Omega mu`, `Sigma -> Omega Sigma
Omega^T`. The per-pair belief-coupling energy is the divergence of the query belief from
each transported key belief, `E_ij = D(q_i || Omega_ij q_j)`, computed independently per
gauge-irrep block (this is what gives the per-head energy). Attention weights are the
softmax `beta_ij = softmax_j(log_prior_ij - E_ij / tau)` with temperature `tau = kappa *
sqrt(d_block)`, where `d_block` is the irrep-block size; `kappa = 1` recovers the Vaswani
`sqrt(d_k)` temperature.

The mean and covariance are then updated by a Fisher natural gradient of `F`, retracted so
that they stay on their manifolds: the mean moves by an ordinary Euclidean step, while the
covariance moves along the affine-invariant SPD geodesic `Sigma_new = Sigma^{1/2}
exp(Sigma^{-1/2} (lr * dSigma) Sigma^{-1/2}) Sigma^{1/2}`, which reduces on the diagonal
cone to `sigma_new = sigma * exp(lr * dsigma / sigma)` and is positive by construction.
The gauge frame `phi` is updated last: its gradient genuinely requires autograd (it is
taken through the belief-coupling block as a function of `phi`), after which it is
preconditioned and retracted back onto the group by a Lie-algebra step. Parallel
mean-field updates are not guaranteed monotone per iteration; free-energy descent holds as
a direction property.

### The block stack and handoff

`vfe_stack` runs `L = n_layers` blocks. After each block the converged belief is blended
into the next block's prior, `mu_p <- (1 - rho) mu_p + rho mu_q` and `Sigma_p <- (1 -
rho_s) Sigma_p + rho_s Sigma_q`, so that depth is realized by repeatedly re-priming the
inference rather than by stacking distinct weight matrices. With `rho = 0` the priors are
frozen across blocks; with `rho = 1` the belief flows fully into the next prior.

### Decode

`PriorBank.decode` scores the converged belief against every vocabulary prior and returns
`logits_{i,v} = -KL(q_i || pi_v) / tau_eff`. For the diagonal family this is computed in
closed form by a single fused matmul, with a global mean-centering shift applied before
the matmul to defeat the catastrophic cancellation that the Mahalanobis reconstruction
would otherwise suffer in float32. A chunked variant computes the cross-entropy without
ever materializing the `(B, N, V)` logit tensor, and an exact Cholesky-based kernel serves
the full-covariance family. The decode is invoked with `kl_max = inf` so that the full KL
ranking over the vocabulary is preserved (the training-time saturation that flattens
distant priors would destroy the argmax).

### The M-step

Training (`vfe3/train.py`) builds an AdamW optimizer with per-group learning rates over
the prior tables and a linear-warmup-then-cosine schedule with an optional learning-rate
floor. Because the E-step is unrolled into the autograd graph, the cross-entropy loss
backpropagates through the entire inference trajectory and into the prior tables; the
mean, log-variance, and gauge-frame tables each get their own learning rate
(`m_p_mu_lr`, `m_p_sigma_lr`, `m_phi_lr`).

## The free energy

The single authoritative scalar (assembled in `vfe3/free_energy.py`) is, on the default
path,

```
F = sum_i [ alpha_i * D(q_i || p_i)
          + lambda_beta * ( sum_j beta_ij E_ij  +  tau * sum_j beta_ij log(beta_ij / pi_ij) ) ]
```

with `E_ij = D(q_i || Omega_ij q_j)`, `beta_ij = softmax_j(log_prior - E/tau)`, and `pi =
softmax_j(log_prior)`. The first term is the self-coupling of each belief to its prior; the
bracket is the belief-coupling block, whose two pieces are the energy `beta E` and the
attention-distribution entropy `tau beta log(beta/pi)`. The entropy term is not optional
decoration: it is exactly what makes the softmax `beta` a stationary point of `F`. Without
it the row-Lagrangian yields a delta rather than a softmax, and the canonical free energy
and the entropy-suppressed surrogate (toggled by `include_attention_entropy`) differ in
their gradients. At `beta*` the whole block collapses to the reduced envelope form `-tau
log Z_i`, which the code uses on the analytic path.

`F` is divergence-agnostic. The per-pair energies and self-divergences come from a
divergence registry, so `KL` is just the Renyi divergence at order one, and a new
f-divergence or a new covariance kernel slots in by registration. The self-coupling weight
`alpha_i` is itself a registry choice: a constant, a closed-form state-dependent function
of the self-divergence, or a per-coordinate variant. For the diagonal Gaussian the KL
kernel is the familiar `0.5 [ sum_k (sigma_q/sigma_p + (mu_q - mu_p)^2/sigma_p) - K + sum_k
log(sigma_p/sigma_q) ]`.

## Gauge theory

The geometric content lives in the gauge group acting on beliefs. A `GaugeGroup` bundles
the Lie-algebra generators with the metadata transport needs — the irrep-block sizes, a
skew-symmetry fast-path flag, and the families whose divergence is invariant under the
group's representation. The admissibility condition is that the divergence be invariant
under common pushforward by the representation, `D(rho(g) q || rho(g) p) = D(q || p)`. For
the Gaussian family under the GL(K) congruence action this holds for every `g` in any
subgroup of GL(K), so every registered group is admissible for the Gaussian belief.

The default group is `block_glk`, the block-diagonal `GL(d_head)^{n_heads}`: each head is
an independent `GL(d_head)` factor, and the irrep blocks are exactly the heads. The
registry also provides full `glk` (a single `GL(K)` block), `tied_block_glk` (one shared
`GL(d_head)` frame across all heads, under which the optional head mixer is exactly
equivariant), `so_k` (the orthogonal group, with the skew-symmetric fast path), and `sp`
(the real symplectic group `Sp(2m, R)` in even dimension). Transport between two tokens is
the flat phi-cocycle, and its triangle holonomy `Omega_ij Omega_jk Omega_ki` is the
identity for the flat connection, which the diagnostics use as a flatness certificate.

## Modularity and the pure-path discipline

Every modeling seam is a named entry in a registry validated at config time: the
divergence functional, the belief family, the gauge group, the connection regime, the
SPD retraction, the phi preconditioner and retraction, the positional encoding, the
attention prior, the normalization, the encode and decode kernels, and the E-step backward
estimator. The single `VFE3Config` dataclass selects them all by name; adding a variant
means writing and registering it, never editing a call site.

A standing project rule is that a theoretically pure path always exists under the
appropriate toggles, and any computationally or theoretically aggressive feature is opt-in
and default-off. The pure path is the one described above: a single belief channel `q` over
its prior `p`, the flat Regime-I cocycle, the KL-to-prior decode, constant self-coupling,
and a fully unrolled E-step gradient. The following are all default-off extension points,
several of which are deliberately predictively inert or still partially wired, and should
not be read as part of the live default model.

The sanctioned neural-network exceptions, each a single learnable parameter rather than a
network and each byte-identical to the pure path at initialization, are: a linear output
projection that replaces the KL decode (`use_prior_bank=False`, the VFE_2.0-parity
ablation); a learned Schur-commutant head mixer (`use_head_mixer=True`); a learned bilinear
edge connection for non-flat Regime-II transport (`transport_mode='regime_ii'`); a learned
scalar self-coupling (`lambda_alpha_mode='learnable'`); and a learned belief-coupling weight
(`learnable_lambda_beta=True`). The Regime-II connection and the head mixer break strict
gauge equivariance away from their zero/identity initialization, which is documented and
user-accepted.

The hierarchical channel of the manuscript — a hyper-prior `h -> s -> p -> q` with a
model-channel belief `s` and a global centroid `r` — is only partially realized. The
hyper-prior term `lambda_h * KL(s || r)` and the model-coupling term `lambda_gamma *
F_red^s` exist as default-off training-loss regularizers on a second set of `s` tables, but
the `s` channel does not feed the belief `q` (its transport is tied and detached, so it is
predictively inert), and the full `s -> q` coupling and the `s`-channel E-step update are
deferred.

## Conventions

The code follows a strict function-signature convention (tensors first, then typed
scalars, then plain scalars, then optionals and `**kwargs`, with vertically aligned names,
types, and `=` signs and tensor-shape comments at the boundaries) and uses paper notation
for variable names (`mu_q`, `sigma_q`, `alpha`, `kappa`, `tau`). It is float32 throughout
with CUDA where applicable, and there is no CLI argument parsing: entry points are
click-to-run, so you edit a config dict and run. Tests are device-agnostic (CPU by
default; set `VFE3_TEST_DEVICE=cuda` for the GPU) and include golden-equivalence checks
against the pinned VFE_2.0 reference, finite-difference gradient checks against the
autograd-of-`F` oracle, and property tests for non-negativity, self-divergence-zero, and
gauge equivariance.
