"""
PINN-CosmoSolver
=================
A Physics-Informed Neural Network (PINN), built from scratch with hand-derived
gradients (via `autograd`, no PyTorch/TensorFlow), that solves the modified
Friedmann equation for a bulk-viscous dark-fluid cosmology:

    dH/dt = -(3/2)(1+w) H^2 + (3/2) zeta0 H^2 + (3/2) zeta1 H^3

where w is the fluid's equation-of-state parameter and zeta(H) = zeta0 + zeta1*H
is a Hubble-dependent bulk viscosity (a standard ansatz in viscous dark-energy /
matter-creation cosmology). This ODE is nonlinear in H and has no simple closed
form once zeta1 != 0, so it is normally solved with a numerical integrator
(Runge-Kutta). Here a neural network is trained to satisfy the ODE directly
-- the network *is* the solution H(t) -- and is cross-validated against a
high-accuracy RK45 integration.

Why a PINN here
----------------
Traditional integrators (RK45) solve the ODE at a fixed initial condition.
A trained PINN gives a smooth, differentiable, closed-form surrogate for H(t)
that can be queried at any t (or even re-used for closely related initial
conditions via fine-tuning) without re-running an integrator -- useful when
this equation sits inside a larger inference pipeline (e.g. MCMC over
zeta0, zeta1, w) where H(t) must be evaluated many times.

Output
------
- results/pinn_vs_rk45.png     : H(t) from PINN vs RK45, absolute error, q(t)
- results/training_loss.png    : PINN physics-residual loss curve
- Printed RMS/max error of the PINN against the RK45 reference solution
"""

import numpy as np
import autograd.numpy as anp
from autograd import elementwise_grad as egrad, grad
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# 1. Physics: the modified (bulk-viscous) Friedmann equation
# ----------------------------------------------------------------------------
w = -0.90        # dark-fluid equation of state (near-Lambda but not exactly -1)
zeta0 = 0.05      # constant bulk-viscosity term
zeta1 = 0.10      # Hubble-dependent bulk-viscosity term
H_i = 1.0         # initial condition H(t=0) = H_i  (units of H0)
T_MAX = 3.0       # integrate over ~3 Hubble times

def rhs(t, H):
    return -1.5 * (1 + w) * H**2 + 1.5 * zeta0 * H**2 + 1.5 * zeta1 * H**3

# ----------------------------------------------------------------------------
# 2. Reference solution: high-accuracy RK45 numerical integration
# ----------------------------------------------------------------------------
t_dense = np.linspace(0, T_MAX, 400)
sol = solve_ivp(rhs, [0, T_MAX], [H_i], t_eval=t_dense, method="RK45",
                 rtol=1e-10, atol=1e-12)
H_rk45 = sol.y[0]
print(f"RK45 reference solution computed over t in [0, {T_MAX}] ({len(t_dense)} points)")

# ----------------------------------------------------------------------------
# 3. Physics-informed neural network, built by hand with autograd
#    H_nn(t) = H_i + t * NN(t)   <-- hard-constrains the initial condition
# ----------------------------------------------------------------------------
N_HIDDEN = 24

def init_params(n_hidden, seed=0):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, 1.0, n_hidden) * 0.5,
        "b1": r.normal(0, 1.0, n_hidden) * 0.5,
        "W2": r.normal(0, 1.0, n_hidden) * 0.5,
        "b2": np.array(0.0),
    }


def H_nn(t, p):
    # t: array of shape (N,)
    hidden = anp.tanh(anp.outer(t, p["W1"]) + p["b1"])   # (N, n_hidden)
    correction = anp.dot(hidden, p["W2"]) + p["b2"]        # (N,)
    return H_i + t * correction


dHdt_nn = egrad(H_nn, 0)  # elementwise d(H_nn)/dt, vectorised over t


def physics_loss(p, t_colloc):
    H = H_nn(t_colloc, p)
    dH = dHdt_nn(t_colloc, p)
    residual = dH - (-1.5 * (1 + w) * H**2 + 1.5 * zeta0 * H**2 + 1.5 * zeta1 * H**3)
    return anp.mean(residual**2)


# collocation points where we *enforce* the ODE (denser near t=0 where the
# dynamics change fastest)
t_colloc = np.concatenate([
    np.linspace(0, 0.5, 60),
    np.linspace(0.5, T_MAX, 60),
])

params = init_params(N_HIDDEN, seed=1)
flat_shapes = {k: v.shape for k, v in params.items()}


def pack(p):
    return np.concatenate([np.ravel(p[k]) for k in ["W1", "b1", "W2", "b2"]])


def unpack(x):
    out, i = {}, 0
    for k in ["W1", "b1", "W2", "b2"]:
        shp = flat_shapes[k]
        n = int(np.prod(shp)) if shp != () else 1
        out[k] = x[i:i + n].reshape(shp) if shp != () else x[i]
        i += n
    return out


def loss_flat(x):
    return physics_loss(unpack(x), t_colloc)


grad_loss_flat = grad(loss_flat)

x0 = pack(params)
loss_history = []


def callback(xk):
    loss_history.append(float(loss_flat(xk)))


print(f"\nTraining PINN ({N_HIDDEN} hidden units, {len(t_colloc)} collocation points) "
      f"via L-BFGS-B on the physics residual...")
result = minimize(loss_flat, x0, jac=grad_loss_flat, method="L-BFGS-B",
                   callback=callback, options={"maxiter": 3000, "ftol": 1e-16, "gtol": 1e-12})
x_final = result.x
final_params = unpack(x_final)
print(f"  optimizer status: {result.message}")
print(f"  final physics-residual loss: {result.fun:.3e}  ({len(loss_history)} iterations)")

# ----------------------------------------------------------------------------
# 4. Evaluate PINN on the dense grid and compare to RK45
# ----------------------------------------------------------------------------
H_pinn = np.array(H_nn(t_dense, final_params))
abs_err = np.abs(H_pinn - H_rk45)
rel_err = abs_err / np.abs(H_rk45)
print(f"\nPINN vs RK45 on {len(t_dense)} evaluation points:")
print(f"  RMS absolute error : {np.sqrt(np.mean(abs_err**2)):.3e}")
print(f"  max absolute error : {np.max(abs_err):.3e}")
print(f"  max relative error : {np.max(rel_err)*100:.3f}%")

# deceleration parameter q(t) = -1 - (dH/dt)/H^2  (q<0 => accelerating expansion)
dH_rk45 = np.array([rhs(0, h) for h in H_rk45])
q_rk45 = -1 - dH_rk45 / H_rk45**2
dH_pinn = np.array(dHdt_nn(t_dense, final_params))
q_pinn = -1 - dH_pinn / H_pinn**2

# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.plot(t_dense, H_rk45, "k-", lw=2.5, alpha=0.6, label="RK45 (reference)")
ax.plot(t_dense, H_pinn, "r--", lw=1.8, label="PINN")
ax.set_xlabel("t (units of $1/H_0$)")
ax.set_ylabel("H(t)")
ax.set_title("Bulk-viscous dark fluid: H(t)")
ax.legend()

ax = axes[1]
ax.semilogy(t_dense, abs_err, color="#C44E52")
ax.set_xlabel("t")
ax.set_ylabel("|H_PINN - H_RK45|")
ax.set_title("PINN absolute error vs. RK45")

ax = axes[2]
ax.plot(t_dense, q_rk45, "k-", lw=2.5, alpha=0.6, label="RK45")
ax.plot(t_dense, q_pinn, "r--", lw=1.8, label="PINN")
ax.axhline(0, color="gray", lw=1, ls=":")
ax.set_xlabel("t")
ax.set_ylabel("deceleration parameter q(t)")
ax.set_title("q<0: accelerated expansion")
ax.legend()

plt.tight_layout()
plt.savefig("results/pinn_vs_rk45.png", dpi=150)
print("\nSaved results/pinn_vs_rk45.png")

fig2, ax = plt.subplots(figsize=(6, 4.5))
ax.semilogy(loss_history)
ax.set_xlabel("L-BFGS-B iteration")
ax.set_ylabel("physics-residual loss (log scale)")
ax.set_title("PINN training curve")
plt.tight_layout()
plt.savefig("results/training_loss.png", dpi=150)
print("Saved results/training_loss.png")

with open("results/summary.txt", "w") as f:
    f.write("PINN-CosmoSolver results\n")
    f.write("=========================\n")
    f.write(f"Model: dH/dt = -1.5(1+w)H^2 + 1.5*zeta0*H^2 + 1.5*zeta1*H^3, "
            f"w={w}, zeta0={zeta0}, zeta1={zeta1}, H(0)={H_i}\n")
    f.write(f"RMS absolute error vs RK45: {np.sqrt(np.mean(abs_err**2)):.3e}\n")
    f.write(f"Max relative error        : {np.max(rel_err)*100:.3f}%\n")
    f.write(f"Training: {len(loss_history)} L-BFGS-B iterations, "
            f"final loss {result.fun:.3e}\n")

print("\nDone.")
