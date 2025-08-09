# Cantilever Beam Stress-Strain Analysis: Mathematical Formulation

This document provides the complete mathematical formulation for the cantilever beam stress-strain analysis implemented in the DOLFINx demos.

## Problem Description

We solve the linear elasticity problem for a cantilever beam under end loading. The beam has:
- Length: L = 2.0 m
- Width: w = 0.2 m  
- Height: h = 0.1 m
- Material: Steel (E = 200 GPa, ν = 0.3)
- Loading: F = 1000 N downward at free end

## 1. Governing Partial Differential Equations

### Strong Form

The linear elasticity problem in 3D is governed by the equilibrium equation:

```
∇ · σ + f = 0    in Ω
```

where:
- `Ω ⊂ ℝ³` is the beam domain
- `σ` is the Cauchy stress tensor
- `f` is the body force vector (gravity, distributed loads)

### Constitutive Relations

For linear elastic materials, stress and strain are related by Hooke's law:

```
σ = C : ε = 2μ ε + λ tr(ε) I
```

where:
- `ε = ½(∇u + ∇uᵀ)` is the strain tensor (symmetric gradient)
- `u` is the displacement vector field
- `μ` and `λ` are the Lamé parameters
- `I` is the identity tensor
- `tr(·)` denotes the trace operator

### Lamé Parameters

The Lamé parameters are expressed in terms of Young's modulus E and Poisson's ratio ν:

```
μ = E / (2(1 + ν))                    (shear modulus)
λ = Eν / ((1 + ν)(1 - 2ν))           (first Lamé parameter)
```

### Expanded Form

Substituting the constitutive relation into the equilibrium equation:

```
∇ · (2μ ε + λ tr(ε) I) + f = 0
```

In index notation:
```
∂/∂xⱼ [2μ εᵢⱼ + λ δᵢⱼ εₖₖ] + fᵢ = 0
```

where `δᵢⱼ` is the Kronecker delta and `εₖₖ = ∇ · u` is the volumetric strain.

## 2. Boundary Conditions

### Dirichlet Boundary Conditions (Essential)

Fixed support at the cantilever root (x = 0):
```
u = 0    on Γ_D = {x ∈ ∂Ω : x₁ = 0}
```

This constrains all three displacement components:
```
u₁ = u₂ = u₃ = 0    at x₁ = 0
```

### Neumann Boundary Conditions (Natural)

Free surfaces with applied tractions:
```
σ · n = t    on Γ_N = ∂Ω \ Γ_D
```

where:
- `n` is the outward unit normal
- `t` is the prescribed traction vector

For our cantilever beam:
- **Free end (x = L)**: Applied load `t = [0, 0, -F/(w·h)]` (distributed over cross-section)
- **Other surfaces**: Traction-free `t = 0`

## 3. Weak Formulation (Galerkin Method)

### Function Spaces

Define the solution and test function spaces:
```
V = {v ∈ [H¹(Ω)]³ : v = 0 on Γ_D}
```

where `H¹(Ω)` is the Sobolev space of functions with square-integrable first derivatives.

### Weak Form Derivation

Multiply the strong form by a test function `v ∈ V` and integrate over the domain:

```
∫_Ω (∇ · σ) · v dΩ + ∫_Ω f · v dΩ = 0
```

Apply integration by parts (divergence theorem):
```
∫_Ω σ : ∇v dΩ - ∫_{∂Ω} (σ · n) · v dS + ∫_Ω f · v dΩ = 0
```

Using the boundary conditions:
- On `Γ_D`: `v = 0` (test functions vanish)
- On `Γ_N`: `σ · n = t`

This gives the weak formulation:

**Find `u ∈ V` such that:**
```
a(u, v) = L(v)    ∀v ∈ V
```

where:
```
a(u, v) = ∫_Ω σ(u) : ε(v) dΩ = ∫_Ω [2μ ε(u) + λ tr(ε(u)) I] : ε(v) dΩ

L(v) = ∫_Ω f · v dΩ + ∫_{Γ_N} t · v dS
```

### Expanded Bilinear Form

The bilinear form can be written as:
```
a(u, v) = ∫_Ω [2μ ε(u) : ε(v) + λ (∇ · u)(∇ · v)] dΩ
```

In index notation:
```
a(u, v) = ∫_Ω [2μ εᵢⱼ(u) εᵢⱼ(v) + λ εₖₖ(u) εₗₗ(v)] dΩ
```

## 4. Finite Element Discretization

### Finite Element Spaces

Discretize using vector-valued Lagrange P1 elements:
```
Vₕ = {vₕ ∈ [C⁰(Ω)]³ : vₕ|_K ∈ [P₁(K)]³ ∀K ∈ Tₕ, vₕ = 0 on Γ_D}
```

where:
- `Tₕ` is a triangulation of `Ω` into tetrahedra
- `P₁(K)` is the space of linear polynomials on element `K`
- `h` is the mesh parameter (maximum element diameter)

### Basis Functions

Express the discrete solution as:
```
uₕ(x) = Σᵢ₌₁ᴺ Uᵢ φᵢ(x)
```

where:
- `{φᵢ}ᵢ₌₁ᴺ` are the vector-valued basis functions
- `{Uᵢ}ᵢ₌₁ᴺ` are the unknown coefficients (DOF values)
- `N` is the total number of degrees of freedom

### Discrete Weak Form

**Find `uₕ ∈ Vₕ` such that:**
```
a(uₕ, vₕ) = L(vₕ)    ∀vₕ ∈ Vₕ
```

## 5. Matrix Formulation

### Linear System

The discrete weak form leads to the linear system:
```
A U = b
```

where:
- `A ∈ ℝᴺˣᴺ` is the stiffness matrix
- `U ∈ ℝᴺ` is the vector of unknown DOF values
- `b ∈ ℝᴺ` is the load vector

### Stiffness Matrix Assembly

The stiffness matrix entries are:
```
Aᵢⱼ = a(φⱼ, φᵢ) = ∫_Ω [2μ ε(φⱼ) : ε(φᵢ) + λ (∇ · φⱼ)(∇ · φᵢ)] dΩ
```

### Load Vector Assembly

The load vector entries are:
```
bᵢ = L(φᵢ) = ∫_Ω f · φᵢ dΩ + ∫_{Γ_N} t · φᵢ dS
```

### Element-wise Assembly

Both matrix and vector are assembled element by element:
```
A = Σ_{K∈Tₕ} A^K,    b = Σ_{K∈Tₕ} b^K
```

where `A^K` and `b^K` are the element contributions computed using numerical quadrature.

## 6. Boundary Condition Enforcement

### Dirichlet Boundary Conditions

Essential boundary conditions are enforced by:

1. **Matrix modification**: Set rows corresponding to constrained DOFs to identity
2. **Vector modification**: Set corresponding entries in `b` to prescribed values (zero for homogeneous BC)

Mathematically:
```
Aᵢᵢ = 1,    Aᵢⱼ = 0 (j ≠ i),    bᵢ = 0    for constrained DOF i
```

### Constraint Application

For our cantilever beam, all DOFs at nodes on `x = 0` are constrained:
```
U₃ₖ₊₀ = U₃ₖ₊₁ = U₃ₖ₊₂ = 0    for node k on Γ_D
```

## 7. Solution Method

### Direct Solver

The linear system is solved using LU factorization:
```
A = L U    (factorization)
L y = b    (forward substitution)
U x = y    (backward substitution)
```

### Solver Properties

- **Symmetry**: `A` is symmetric positive definite (SPD)
- **Sparsity**: `A` has a sparse structure due to local support of basis functions
- **Conditioning**: Well-conditioned for reasonable mesh aspect ratios

## 8. Post-Processing

### Stress Recovery

Compute stress tensor from displacement solution:
```
σₕ = 2μ ε(uₕ) + λ tr(ε(uₕ)) I
```

### Von Mises Stress

The Von Mises stress (scalar failure criterion) is:
```
σᵥₘ = √(3/2 σ'ᵢⱼ σ'ᵢⱼ)
```

where `σ'ᵢⱼ = σᵢⱼ - ⅓δᵢⱼσₖₖ` is the deviatoric stress tensor.

### Strain Energy

The total strain energy is:
```
U = ½ ∫_Ω σ : ε dΩ = ½ Uᵀ A U
```

## 9. Analytical Validation

### Euler-Bernoulli Beam Theory

For comparison, the classical beam theory gives:

**Maximum deflection** (at free end):
```
δₘₐₓ = FL³/(3EI)
```

**Maximum stress** (at fixed end, top/bottom fiber):
```
σₘₐₓ = FL(h/2)/I
```

where `I = wh³/12` is the second moment of area.

### Expected Results

For our parameters:
- `I = 0.2 × 0.1³/12 = 1.667 × 10⁻⁶ m⁴`
- `δₘₐₓ = 1000 × 2³/(3 × 200×10⁹ × 1.667×10⁻⁶) ≈ 0.80 mm`
- `σₘₐₓ = 1000 × 2 × 0.05/(1.667×10⁻⁶) ≈ 60 MPa`

The FEM solution should converge to these values as the mesh is refined.

## 10. Implementation Notes

### Numerical Integration

Element integrals are computed using Gaussian quadrature:
```
∫_K f(x) dx ≈ Σᵢ wᵢ f(xᵢ) |det(J)|
```

where `{xᵢ, wᵢ}` are quadrature points and weights, and `J` is the Jacobian of the element transformation.

### Mesh Quality

For accurate results, ensure:
- **Aspect ratio**: Elements should not be too elongated
- **Boundary layer**: Sufficient resolution near stress concentrations
- **Convergence**: Refine mesh until solution stabilizes

### DOLFINx Implementation

The DOLFINx library handles:
- Automatic assembly of `A` and `b`
- Efficient sparse matrix storage
- Parallel computation (MPI)
- Integration with PETSc solvers
- XDMF output for visualization

This mathematical formulation provides the theoretical foundation for understanding the numerical implementation in the cantilever beam demos.
