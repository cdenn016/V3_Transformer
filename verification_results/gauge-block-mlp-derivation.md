# Gauge BlockMLP derivation

This note binds the mathematical contracts exercised by the gauge BlockMLP tests to the current worktree.

## Invariant scalar gate

Let the declared representation decompose into blocks, with a gauge action
`G = diag(G_1, ..., G_H)`, mean blocks `mu_h`, and full covariance blocks
`Sigma_hk`. Under a frame change,

`mu'_h = G_h mu_h` and `Sigma'_hh = G_h Sigma_hh G_h^T`.

For nonsingular `G_h` and positive-definite `Sigma_hh`,

`(Sigma'_hh)^-1 = G_h^-T Sigma_hh^-1 G_h^-1`,

so the Mahalanobis scalar

`s_h = mu_h^T Sigma_hh^-1 mu_h`

is invariant. Therefore any scalar network `gamma(s)` has the same value in
every gauge frame. The residual gate

`f_h(mu, Sigma) = (1 + gamma_h(s)) mu_h`
For dropout, this equality is conditioned on the same scalar mask for each
declared block. Thus the deterministic/evaluation path and each fixed mask are
strictly equivariant. Independent dropout samples are equivariant in
distribution rather than identical sample by sample.


then obeys `f_h(mu', Sigma') = G_h f_h(mu, Sigma)`. This establishes strict
equivariance of the mean map on the stated domain. Differentiating the
equivariance identity with respect to `mu' = G mu` while holding the transformed
covariance fixed gives `J' = G J G^-1`. Consequently the delta-method contract
also transforms covariantly:

The strictly positive floor is covariant for the same reason:
`J' Sigma' J'^T + epsilon Sigma' = G (J Sigma J^T + epsilon Sigma) G^T`.
For `epsilon > 0` and incoming `Sigma` positive definite, the result remains
positive definite even when `J` is rank-deficient.
`J' Sigma' J'^T = G (J Sigma J^T) G^T`.

The implementation's explicit derivative uses
`d s_h / d mu_h = 2 Sigma_hh^-1 mu_h`, which is valid because the covariance
block is symmetric. The autograd-reference tests independently check the
assembled dense Jacobian.

## Canonical-frame coordinate MLP

Let `V` be a realized vertex frame and `a = V^-1 mu`. For an ordinary residual
MLP `F` in canonical coordinates, define `f_V(mu) = V F(V^-1 mu)`. Under a left
frame change `mu' = G mu`, `V' = G V`, the canonical coordinate is unchanged:

`V'^-1 mu' = V^-1 mu`.

Thus `f_V'(mu') = G f_V(mu)`, and the realized Jacobian is
`J = V J_F V^-1`. This is left-equivariant only after fixing the realized
right-frame representative. A right change `V -> V R` generally changes the
input of an ordinary coordinate MLP, so this mode is intentionally not labeled
strictly gauge-pure.

## Covariance scope

`passthrough` leaves covariance unchanged and is not a nonlinear pushforward.
`delta_full` applies the regularized first-order closure
`Sigma_out = J Sigma J^T + epsilon Sigma`; it is
not claimed to be the exact distributional pushforward of a Gaussian through a
nonlinear map.
