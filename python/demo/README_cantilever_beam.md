# Cantilever Beam (Linear Elasticity) — Mathematical Formulation

This document describes the mathematical model and linear algebra formulation used by `python/demo/demo_cantilever_beam.py`. It does not cover the code; instead it states the PDEs, boundary conditions, Galerkin weak form, discrete scheme, and matrix assembly operations that produce the linear system solved in the demo.

## 1. Physical Problem

- Domain: A 3D rectangular beam Ω ⊂ ℝ³ with length L in the x-direction, width W (y-direction), and height H (z-direction).
- Material: Linear, homogeneous, isotropic elasticity defined by Young’s modulus E and Poisson ratio ν. The Lamé parameters are
  - μ = E / (2(1+ν))
  - λ = E ν / ((1+ν)(1−2ν))
- Loading:
  - Clamped support (Dirichlet) on the left face Γ_D = {x = 0}: u = 0
  - Traction (Neumann) on the right face Γ_N = {x = L}: σ(u) n = t, with given traction vector t

## 2. Governing Equations

Let u: Ω → ℝ³ be the displacement field, ε(u) the small-strain tensor, and σ(u) the Cauchy stress tensor.

- Small strain:
  ε(u) = sym(∇u) = (∇u + (∇u)ᵀ)/2

- Linear elastic stress (Hooke’s law for isotropic materials):
  σ(u) = 2 μ ε(u) + λ tr(ε(u)) I

- Static equilibrium (no body force):
  −∇ · σ(u) = 0 in Ω

- Boundary conditions:
  - Dirichlet: u = 0 on Γ_D
  - Neumann: σ(u) n = t on Γ_N

## 3. Weak (Variational) Form

Let V = { v ∈ [H¹(Ω)]³ : v = 0 on Γ_D } and find u ∈ V such that
∫_Ω σ(u) : ε(v) dx = ∫_{Γ_N} t · v ds,  for all v ∈ V.

Using the constitutive relation,
∫_Ω [2 μ ε(u) : ε(v) + λ tr(ε(u)) tr(ε(v))] dx = ∫_{Γ_N} t · v ds.

This is the standard Galerkin formulation for linear elasticity.

## 4. Galerkin Discretization

- Choose a conforming finite-dimensional subspace V_h ⊂ V composed of vector-valued, first-order Lagrange (P1) basis functions on a tetrahedral or hexahedral mesh of Ω. Each basis function is continuous at nodes and has 3 components (x, y, z).
- Approximate u by u_h ∈ V_h: u_h(x) = ∑_j N_j(x) d_j, where N_j are vector basis functions and d_j the nodal displacement unknowns.

The discrete problem is: find u_h ∈ V_h such that
a(u_h, v_h) = l(v_h) for all v_h ∈ V_h,
with
a(u, v) = ∫_Ω σ(u) : ε(v) dx,
l(v) = ∫_{Γ_N} t · v ds.

## 5. Matrix Formulation

Let {N_i} be the vector basis functions spanning V_h. The linear system K u = f arises with:

- Stiffness matrix K = [K_ij], where
  K_ij = ∫_Ω σ(N_j) : ε(N_i) dx
       = ∫_Ω [2 μ ε(N_j) : ε(N_i) + λ tr[ε(N_j)] tr[ε(N_i)]] dx.

Equivalently, using Voigt notation,
K_ij = ∫_Ω B_iᵀ C B_j dx,
where:
- B_j is the strain-displacement matrix for basis function N_j,
- C is the 6×6 isotropic elasticity matrix constructed from λ, μ.

- Load vector f = [f_i], where
  f_i = ∫_{Γ_N} t · N_i ds.

- Dirichlet boundary conditions u = 0 on Γ_D are imposed by constraining the corresponding degrees of freedom.

## 6. Assembly and Boundary Operations (Linear Algebra View)

Given the forms a(·,·) and l(·), the assembled matrix and vector are:

1) K = Assemble(a) = ∑_elements ∫_Ω_e Bᵀ C B dx, accumulated into the global sparse matrix.
2) f = Assemble(l) = ∑_boundary_facets ∫_{Γ_N∩∂Ω_e} Nᵀ t ds, accumulated into the global vector.

To incorporate Dirichlet conditions u = 0 on Γ_D (clamped face):
- Identify the constrained degrees of freedom (DOFs) associated with Γ_D.
- Apply the constraints to the system. In many FEM codes this is equivalent to:
  - Lifting: f ← f − ∑_{bcs} K[:, constrained] u_constrained, which subtracts known contributions to the RHS.
  - Row/column modifications or equivalent operations so that the constrained DOFs are fixed to their prescribed values (here, zero) and the modified system remains consistent.

Final linear system:
K u = f,
with appropriate treatment of constrained DOFs. For linear isotropic elasticity on a clamped domain, K is symmetric positive definite (in the absence of rigid-body modes).

## 7. Notes on Measures and Loading

- The traction is applied only on the end face Γ_N = {x = L}, so the boundary integral for f is restricted to that face. In practice, this is done by tagging the facets on Γ_N and integrating the Neumann term over that tagged set.

## 8. Solver

- The resulting linear system is solved with a Krylov or direct method. In the demo, a PETSc KSP is used with a direct factorization (preonly + LU), so the solution step is:
  - Solve K u = f for u using the chosen solver/preconditioner.

This completes the mathematical formulation that underpins the demo’s assembly and solve steps.
