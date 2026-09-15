# PyTorch is used to define and train the neural network.
import torch

# torch.nn provides the neural-network layers and activation functions.
import torch.nn as nn

# NumPy is used for post-processing and analytic-solution evaluation.
import numpy as np

import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================

# Fix the random seed for reproducible neural-network initialization.
torch.manual_seed(42)
# Use double precision
dtype = torch.float64
# Run the calculation on the CPU.
device = torch.device("cpu")

# Length of the one-dimensional computational domain, z in [0, L].
L = 1.0
# Small perturbation parameter used to represent solution.
G = 0.1 
# Number of collocation points used to enforce the governing equation.
N_collocation = 100

# Number of Adam optimization steps before L-BFGS refinement.
adam_steps = 5000
# Learning rate for the Adam optimizer.
learning_rate = 1.0e-3
# Relative weights of the governing-equation, boundary-condition,
# and integral-constraint contributions to the total loss.
lambda_pde = 1.0
lambda_bc = 10.0
lambda_integral = 10.0


# ============================================================
# Neural network
#
# Input:
#     z
#
# Output:
#     f(z)
# ============================================================

# Define a fully connected neural network that approximates f(z).
class PINN(nn.Module):
    def __init__(self):
        # Initialize the base PyTorch neural-network class.
        super().__init__()
        # Construct a neural network with three hidden layers.
        self.net = nn.Sequential(
            # Map the scalar coordinate z to 32 hidden features.
            nn.Linear(1, 32),
            # Use tanh activation, which is smooth and therefore suitable for
            # evaluating derivatives through automatic differentiation.
            nn.Tanh(),
            # Additional hidden layers increase the expressive capacity of the network.
            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),
            # Map the final hidden representation to the scalar solution f(z).
            nn.Linear(32, 1)
        )

    def forward(self, z):
        # Evaluate the neural-network approximation f(z).
        return self.net(z)

# Instantiate the PINN and move its parameters to the selected device
# using double precision.
model = PINN().to(device=device, dtype=dtype)


# ============================================================
# Unknown constant K
#
# K is treated as an additional trainable parameter.
# ============================================================

K = nn.Parameter(
    torch.tensor(0.0, dtype=dtype, device=device)
)


# ============================================================
# Collocation points
# ============================================================
# Generate uniformly spaced collocation points over the domain [0, L].
# The points are stored as an N x 1 tensor for input to the network.
z = torch.linspace(
    0.0,
    L,
    N_collocation,
    dtype=dtype,
    device=device
# Reshape the coordinate array into the input format expected by the network.
).reshape(-1, 1)
# Enable differentiation with respect to z so that f_z and f_zz
# can be computed using automatic differentiation.
z.requires_grad_(True)


# ============================================================
# Governing equation residual
#
# First-order perturbation problem:
#
#     f_zz + f = -K + z
#
# Therefore:
#
#     R = f_zz + f + K - z
# ============================================================

# Evaluate the governing-equation residual at the collocation points.
def compute_pde_residual(model, z, K):
    # Evaluate the neural-network approximation f(z).
    f = model(z)

    # First derivative: df/dz
    f_z = torch.autograd.grad(
        f,
        z,
        # Differentiate all network outputs with respect to their corresponding inputs.
        grad_outputs=torch.ones_like(f),
        # Retain the derivative graph because a second derivative is required.
        create_graph=True
    )[0]

    # Second derivative: d^2f/dz^2
    f_zz = torch.autograd.grad(
        f_z,
        z,
        grad_outputs=torch.ones_like(f_z),
        create_graph=True
    )[0]

    residual = f_zz + f + K - z
    # Return the pointwise governing-equation residual.
    return residual


# ============================================================
# Optimizer
#
# Optimize both:
#
#     neural-network parameters
#     K
# ============================================================

optimizer = torch.optim.Adam(
    list(model.parameters()) + [K],
    lr=learning_rate
)


# ============================================================
# Adam training
# ============================================================
# Store the total Adam loss for later visualization.
loss_history = []
# Perform the first stage of optimization using Adam.
for step in range(adam_steps):
    # Clear gradients accumulated during the previous optimization step.
    optimizer.zero_grad()
    z.grad = None 

    # --------------------------------------------------------
    # PDE loss
    # --------------------------------------------------------

    residual = compute_pde_residual(model, z, K)

    # Penalize the mean-squared governing-equation residual.
    loss_pde = torch.mean(residual**2)


    # --------------------------------------------------------
    # Boundary-condition loss
    #
    # f(0) = 0
    # f(L) = 0
    # --------------------------------------------------------

    # Define the left boundary point z = 0.
    z_left = torch.tensor(
        [[0.0]],
        dtype=dtype,
        device=device
    )
    
    # Define the left boundary point z = 0.
    z_right = torch.tensor(
        [[L]],
        dtype=dtype,
        device=device
    )

    # Evaluate the PINN solution at both boundaries.
    f_left = model(z_left)
    f_right = model(z_right)

    loss_bc = f_left.pow(2).mean() + f_right.pow(2).mean()


    # --------------------------------------------------------
    # Integral constraint
    #
    #     integral_0^L f(z) dz = 0
    #
    # Numerical quadrature using trapezoidal rule.
    # --------------------------------------------------------

    f_values = model(z)

    # Approximate the integral of f over [0, L] using the trapezoidal rule.
    integral_f = torch.trapz(
        f_values.squeeze(),
        z.squeeze()
    )

    loss_integral = integral_f**2


    # --------------------------------------------------------
    # Total loss
    # --------------------------------------------------------
    # Combine the governing equation, boundary conditions, and global
    # integral constraint into a single weighted loss function.
    loss = (
        lambda_pde * loss_pde
        + lambda_bc * loss_bc
        + lambda_integral * loss_integral
    )

    # Backpropagate the total loss to compute gradients with respect to
    # both the network parameters and K.
    loss.backward()
    # Update the trainable parameters using one Adam optimization step.
    optimizer.step()
    # Record the scalar loss value for post-processing.
    loss_history.append(loss.item())


    # Report the training progress every 500 Adam iterations.
    if step % 500 == 0:

        print(
            f"step = {step:5d} | "
            f"loss = {loss.item():.3e} | "
            f"PDE = {loss_pde.item():.3e} | "
            f"BC = {loss_bc.item():.3e} | "
            f"Integral = {loss_integral.item():.3e} | "
            f"K = {K.item():.8f}"
        )


# ============================================================
# L-BFGS refinement
#
# Adam is first used for robust initial training, after which
# L-BFGS is applied to further reduce the PINN residual.
# ============================================================
# Create an L-BFGS optimizer for second-stage refinement.
optimizer_lbfgs = torch.optim.LBFGS(
    # Continue optimizing both the network parameters and K.
    list(model.parameters()) + [K],
    # Initial L-BFGS step size
    lr=1.0,
    max_iter=1000,
    max_eval=1200,
    tolerance_grad=1.0e-12,
    tolerance_change=1.0e-14,
    history_size=100,
    # Use a strong-Wolfe line search for high-accuracy L-BFGS refinement.
    line_search_fn="strong_wolfe"
)

# Define the closure required by PyTorch's L-BFGS optimizer.
# L-BFGS may evaluate the loss and its gradients multiple times per step.
lbfgs_calls = 0

def closure():
    global lbfgs_calls

    # Clear gradients before each L-BFGS loss evaluation.
    optimizer_lbfgs.zero_grad()
    z.grad = None 

    residual = compute_pde_residual(model, z, K)
    loss_pde = torch.mean(residual**2)

    z_left = torch.tensor(
        [[0.0]],
        dtype=dtype,
        device=device
    )

    z_right = torch.tensor(
        [[L]],
        dtype=dtype,
        device=device
    )

    f_left = model(z_left)
    f_right = model(z_right)

    loss_bc = f_left.pow(2).mean() + f_right.pow(2).mean()

    f_values = model(z)

    integral_f = torch.trapz(
        f_values.squeeze(),
        z.squeeze()
    )

    loss_integral = integral_f**2

    loss = (
        lambda_pde * loss_pde
        + lambda_bc * loss_bc
        + lambda_integral * loss_integral
    )

    loss.backward()

    lbfgs_calls += 1
    if lbfgs_calls == 1 or lbfgs_calls % 50 == 0:
        print(
            f"L-BFGS eval = {lbfgs_calls:5d} | "
            f"loss = {loss.item():.3e} | "
            f"PDE = {loss_pde.item():.3e} | "
            f"BC = {loss_bc.item():.3e} | "
            f"Integral = {loss_integral.item():.3e} | "
            f"K^(1) = {K.item():.8f}",
            flush=True,
        )
    return loss

# Run the L-BFGS refinement stage.
print("\nStarting L-BFGS", flush=True)
optimizer_lbfgs.step(closure)
print(f"L-BFGS complete: {lbfgs_calls} closure evaluations", flush=True)

# ============================================================
# Results
# ============================================================

print("\nTraining complete")
print(f"Learned K = {K.item():.12f}")

# Generate a finer grid for evaluating and plotting the trained solution.
z_plot = torch.linspace(
    0.0,
    L,
    500,
    dtype=dtype,
    device=device
).reshape(-1, 1)

# Disable gradient tracking because derivatives are no longer needed.
with torch.no_grad():
    # Evaluate the trained PINN approximation on the plotting grid.
    f_pred = model(z_plot)

# Convert the PyTorch tensors to one-dimensional NumPy arrays
# for analytic comparison, file output, and plotting.
z_np = z_plot.cpu().numpy().flatten()
f_np = f_pred.cpu().numpy().flatten()


# ============================================================
# Total solutions: 1 + G * first-order correction
# ============================================================

from pathlib import Path

# Analytic first-order correction: Eq. (23)
f1_exact = (
    (L / 2.0) * np.cos(z_np)
    - (L / 2.0) * (1.0 + np.cos(L)) / np.sin(L) * np.sin(z_np)
    - L / 2.0
    + z_np
)

# Reconstruct total solutions
f_total_pinn = 1.0 + G * f_np
f_total_exact = 1.0 + G * f1_exact

K_total_pinn = 1.0 + G * K.item()
K_total_exact = 1.0 + G * (L / 2.0)

print(f"Total K (PINN)     = {K_total_pinn:.12f}")
print(f"Total K (analytic) = {K_total_exact:.12f}")

# ============================================================
# Tecplot output
# ============================================================

output_path = Path(__file__).resolve().parent / "total_solution.dat"

np.savetxt(
    output_path,
    np.column_stack((
        z_np,
        f_total_pinn,
        f_total_exact,
        np.full_like(z_np, K_total_pinn),
        np.full_like(z_np, K_total_exact),
    )),
    fmt="%.12e",
    header=(
        f'TITLE = "Total solution, G={G:g}"\n'
        'VARIABLES = "z", "f_PINN", "f_analytic", '
        '"K_PINN", "K_analytic"\n'
        f'ZONE T="Solution", I={len(z_np)}, DATAPACKING=POINT'
    ),
    comments="",
)

print(f"Saved: {output_path}")

# ============================================================
# Plot total f and K
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(z_np, f_total_pinn, label="PINN")
axes[0].plot(z_np, f_total_exact, "r--", label="Analytic")
axes[0].set_xlabel(r"$z$")
axes[0].set_ylabel(r"$1 + G f^{(1)}(z)$")

axes[1].plot(
    z_np, np.full_like(z_np, K_total_pinn), label="PINN"
)
axes[1].plot(
    z_np, np.full_like(z_np, K_total_exact), "r--", label="Analytic"
)
axes[1].set_xlabel(r"$z$")
axes[1].set_ylabel(r"$1 + G K^{(1)}$")

for ax in axes:
    ax.legend()
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

fig.suptitle(f"G = {G:g}")
fig.tight_layout()
figure_dir = Path(__file__).resolve().parent / "figures"
figure_dir.mkdir(exist_ok=True)

fig.savefig(
    figure_dir / "solution_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Plot Adam loss history
# ============================================================

plt.figure()

plt.semilogy(loss_history)

plt.xlabel("Adam iteration")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()