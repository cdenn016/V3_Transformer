# V3_Transformer (VFE_3.0)

Clean-room rebuild of the gauge-theoretic variational free energy transformer.
No neural networks: all capacity comes from iterative VFE minimization over
Gaussian belief tuples `(mu, Sigma, phi)`. Built bottom-up with every layer
numerically pinned to VFE_2.0 by golden tests. See
`docs/superpowers/specs/2026-05-29-vfe3-clean-room-design.md`.
