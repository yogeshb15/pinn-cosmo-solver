# PINN-CosmoSolver

**A physics-informed neural network — built from scratch with hand-derived gradients (no PyTorch/TensorFlow) — that solves a nonlinear, modified Friedmann equation for a bulk-viscous dark-fluid cosmology, cross-validated against RK45.**

## The equation

For a dark fluid with equation-of-state parameter *w* and a Hubble-dependent bulk viscosity ζ(H) = ζ₀ + ζ₁H (a standard ansatz in viscous / matter-creation cosmology), the Friedmann + continuity equations reduce to:

```
dH/dt = -1.5(1+w)H² + 1.5·ζ₀·H² + 1.5·ζ₁·H³
```

This is nonlinear in H and — once ζ₁ ≠ 0 — has no simple closed form, so it's normally integrated numerically.

## Why a PINN

A trained PINN gives a smooth, differentiable, closed-form surrogate for H(t) that can be queried anywhere without re-running an integrator — useful when this kind of equation sits inside a larger pipeline (e.g. an MCMC over ζ₀, ζ₁, w) where H(t) needs to be evaluated repeatedly.

## Method

- H(t) is represented by a small MLP with the initial condition **hard-baked in**: `H_nn(t) = H_i + t · NN(t)`, so H(0) = H_i is satisfied exactly rather than penalized.
- The network's time-derivative is obtained with `autograd` (elementwise automatic differentiation) — no deep-learning framework, just NumPy + hand-built forward/backward passes.
- Trained by minimizing the mean-squared **physics residual** (dH_nn/dt − RHS(H_nn)) at 120 collocation points via L-BFGS-B.
- Validated against a high-accuracy `scipy.integrate.solve_ivp` (RK45, rtol=1e-10) reference solution.

## Results

| Metric | Value |
|---|---|
| RMS absolute error vs. RK45 | **2.6 × 10⁻⁵** |
| Max relative error | **0.004%** |
| Training | 3,000 L-BFGS-B iterations, final residual loss 2.8 × 10⁻⁹ |

![PINN vs RK45](results/pinn_vs_rk45.png)

The right-most panel shows the derived deceleration parameter q(t) = −1 − (dH/dt)/H² from both solutions — the PINN reproduces the expansion-history dynamics (not just H itself) to the same precision.

## Run it

```bash
pip install -r requirements.txt
python pinn_cosmo_solver.py
```

## Stack

Python · NumPy · `autograd` (hand-rolled PINN, no PyTorch/TensorFlow) · SciPy (RK45 reference + L-BFGS-B optimizer) · Matplotlib

## License

All rights reserved — see [LICENSE](LICENSE). This repository is shared publicly to demonstrate the work; it is not open source, and no use (including research or academic use) is permitted without written permission.

---
*by Yogesh Bhardwaj — PhD (Applied Mathematics), Delhi Technological University.*
