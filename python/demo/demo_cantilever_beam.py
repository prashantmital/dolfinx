#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

"""
Cantilevered beam under end load (linear elasticity, 3D, steel).

This demo is structured to highlight the key concepts in a finite element
simulation for beginners:
- Geometry: shape and physical placement in 3D space
- Discretization: meshing the geometry
- Boundary conditions: constraining and loading the model
- Matrix construction: building the linear system
- Linear solver: solving for the unknown displacement field
- Analysis and results: computing strain/stress and writing outputs

How to run:
    python python/demo/demo_cantilever_beam.py

Outputs (ParaView-readable XDMF):
- out_cantilever/beam_mesh.xdmf: Mesh
- out_cantilever/displacement.xdmf: Displacement field u
- out_cantilever/strain.xdmf: Small-strain tensor field ε
- out_cantilever/stress.xdmf: Cauchy stress tensor field σ
"""

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl

from dolfinx.fem import (
    Expression,
    Function,
    dirichletbc,
    functionspace,
    form,
    locate_dofs_geometrical,
)
from dolfinx.fem import petsc as fem_petsc
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh


def make_geometry():
    """Geometry: define the physical size of the beam in 3D space (L, W, H)."""
    L = 1.0
    W = 0.1
    H = 0.1
    return L, W, H


def discretize(L, W, H, nx=30, ny=6, nz=6):
    """Discretization: create a volumetric mesh and vector-valued function space."""
    msh = dmesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([L, W, H])],
        [nx, ny, nz],
        cell_type=dmesh.CellType.hexahedron,
    )
    V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    return msh, V


def material():
    """Material: return steel properties and Lamé parameters (E, nu, lambda, mu)."""
    E = 210e9
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return E, nu, lmbda, mu


def epsilon(u):
    """Strain operator: small-strain tensor ε(u) = sym(grad(u))."""
    return ufl.sym(ufl.grad(u))


def sigma(u, lmbda, mu, gdim):
    """Stress law: linear elasticity σ(u) = 2μ ε(u) + λ tr(ε(u)) I."""
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)


def apply_boundary_conditions(msh, V):
    """Boundary conditions: clamp the x=0 face by enforcing u = 0 on that boundary."""
    def on_clamped(x):
        return np.isclose(x[0], 0.0)

    u0 = Function(V)
    u0.x.array[:] = 0.0
    dofs = locate_dofs_geometrical(V, on_clamped)
    bcs = [dirichletbc(u0, dofs)]
    return bcs


def build_measures_and_load(msh, L, traction_mag=1e6, traction_dir=(0.0, 0.0, -1.0)):
    """Measures and load: mark end face (x=L) and define the traction vector."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)

    def end_face(x):
        return np.isclose(x[0], L)

    end_facets = dmesh.locate_entities_boundary(msh, fdim, end_face)
    values = np.full(end_facets.shape, 1, dtype=np.int32)
    facet_tags = dmesh.meshtags(msh, fdim, end_facets, values)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    t_vec = traction_mag * np.array(traction_dir, dtype=float)
    return ds, t_vec


def build_forms(msh, V, lmbda, mu, ds, t_vec):
    """Matrix construction (forms): build bilinear form a and linear form L."""
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u, lmbda, mu, msh.geometry.dim), epsilon(v)) * ufl.dx
    L_form = ufl.inner(ufl.as_vector(t_vec), v) * ds(1)
    return a, L_form


def assemble_system(a, L_form, bcs):
    """Matrix construction (assembly): assemble global matrix A and RHS vector b, applying BCs."""
    A = fem_petsc.assemble_matrix(form(a), bcs=bcs)
    A.assemble()

    b = fem_petsc.assemble_vector(form(L_form))
    fem_petsc.apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)
    return A, b


def solve_system(A, b, V, ksp_type="preonly", pc_type="lu", options_prefix="cantilever_"):
    """Linear solver: configure PETSc KSP and solve for the displacement field uh."""
    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOptionsPrefix(options_prefix)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setFromOptions()

    uh = Function(V)
    uh_vec = uh.x.petsc_vec
    ksp.solve(b, uh_vec)
    uh.x.scatter_forward()
    return uh, ksp


def analyze_and_export(msh, uh, lmbda, mu, out_dir="out_cantilever"):
    """Analysis and results: compute strain/stress fields and export XDMF files for visualization."""
    TenSpace = functionspace(msh, ("DG", 0, (msh.geometry.dim, msh.geometry.dim)))

    eps_func = Function(TenSpace)
    eps_expr = ufl.as_tensor(epsilon(uh))
    eps_func.name = "strain"
    eps_E = Expression(eps_expr, TenSpace.element.interpolation_points)
    eps_func.interpolate(eps_E)

    sig_func = Function(TenSpace)
    sig_expr = ufl.as_tensor(sigma(uh, lmbda, mu, msh.geometry.dim))
    sig_func.name = "stress"
    sig_E = Expression(sig_expr, TenSpace.element.interpolation_points)
    sig_func.interpolate(sig_E)

    if msh.comm.rank == 0:
        import os
        os.makedirs(out_dir, exist_ok=True)
    msh.comm.barrier()

    with XDMFFile(MPI.COMM_WORLD, f"{out_dir}/beam_mesh.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(msh)

    with XDMFFile(MPI.COMM_WORLD, f"{out_dir}/displacement.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(uh)

    with XDMFFile(MPI.COMM_WORLD, f"{out_dir}/strain.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(eps_func)

    with XDMFFile(MPI.COMM_WORLD, f"{out_dir}/stress.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(sig_func)

    if msh.comm.rank == 0:
        disp_norm = np.linalg.norm(uh.x.array)
        print(f"Solved cantilever beam. ||u||_2 (global vector) = {disp_norm:.6e}")
        print(f"Results written to: {out_dir}/")


def main():
    """Pipeline: geometry → discretization → BCs → forms → assembly → solver → analysis/output."""
    L, W, H = make_geometry()
    msh, V = discretize(L, W, H)
    E, nu, lmbda, mu = material()
    bcs = apply_boundary_conditions(msh, V)
    ds, t_vec = build_measures_and_load(msh, L)
    a, L_form = build_forms(msh, V, lmbda, mu, ds, t_vec)
    A, b = assemble_system(a, L_form, bcs)
    uh, ksp = solve_system(A, b, V, ksp_type="preonly", pc_type="lu", options_prefix="demo_cantilever_")
    assert ksp.getConvergedReason() > 0
    analyze_and_export(msh, uh, lmbda, mu)


if __name__ == "__main__":
    main()
