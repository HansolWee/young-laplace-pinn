# Young–Laplace PINN

A physics-informed neural network (PINN) for the first-order
small-Bond-number perturbation problem of the Young–Laplace equation.

## Overview

The solution is reconstructed as:

- f(z) = 1 + G f¹(z)
- K = 1 + G K¹
  
where G is the bond number.

The network learns the first-order correction f¹(z) and the reference
pressure K¹ (constant) using the governing equation, boundary conditions,
and an integral constraint.

The results are compared with an analytical solution.

## Method

- PyTorch with double precision
- Three hidden layers with 32 neurons each
- Tanh activation
- Adam optimization followed by L-BFGS refinement
- Automatic differentiation for spatial derivatives

## Example result

For L = 1 and G = 0.1:

| Quantity |      PINN      | Analytical |
|----------|----------------|------------|
| K¹       | 0.500000459271 | 0.5        |
| 1 + G K¹ | 1.050000045927 | 1.05       |

## Run

Install dependencies:

    pip install torch numpy matplotlib

Run the solver:

    python YL_small_bond_number.py

## Output

- PINN and analytical solution plots
- Training loss history
- Tecplot data: total_solution.dat

## Scope

This code solves the first-order perturbation problem.
The reconstructed solution includes terms through first order in G.
