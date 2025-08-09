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

This demo sets up a rectangular cantilever beam clamped at x=0 with a traction
load applied on the free end face at x=L. It solves the linear elasticity
problem for small strains, computes strain and stress fields, and writes
outputs to XDMF for visualization (e.g., ParaView).

Geometry:
- Beam length L in x-direction, width W in y-direction, height H in z-direction
- Rectangular cross section (W x H)

Material (steel, typical values):
- Young's modulus E = 210e9 Pa
- Poisson's ratio nu = 0.3

Boundary conditions:
- Clamped (u = 0) at x = 0 plane
- End load as uniform traction on face x = L

How to run:
    python python/demo/demo_cantilever_beam.py

Outputs:
- out_cantilever/beam_mesh.xdmf: Mesh
- out_cantilever/displacement.xdmf: Displacement field u
- out_cantilever/strain.xdmf: Small-strain tensor field ε
- out_cantilever/stress.xdmf: Cauchy stress tensor field σ

Notes:
- This script is written for clarity and education. Mesh and load values can be
  adjusted via the parameters section below.
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
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh


L = 1.0          # beam length (m)
W = 0.1          # beam width (m)
H = 0.1          # beam height (m)
nx, ny, nz = 30, 6, 6  # mesh resolution

E = 210e9        # Young's modulus (Pa) for steel
nu = 0.3         # Poisson's ratio
traction_mag = 1e6  # traction magnitude (Pa) applied on end face x=L
traction_dir = np.array([0.0, 0.0, -1.0])  # direction of traction (unit vector)


msh = dmesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, W, H])],
    [nx, ny, nz],
    cell_type=dmesh.CellType.hexahedron,
)


V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(msh.geometry.dim)


def on_clamped(x):
    return np.isclose(x[0], 0.0)


u_clamp = Function(V)
u_clamp.x.array[:] = 0.0
dofs_clamped = locate_dofs_geometrical(V, on_clamped)
bcs = [dirichletbc(u_clamp, dofs_clamped)]

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

t_vec = traction_mag * traction_dir


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.inner(ufl.as_vector(t_vec), v) * ds(1)


uh = Function(V)
uh.name = "u"

problem = LinearProblem(
    a,
    L_form,
    bcs=bcs,
    u=uh,
    petsc_options_prefix="demo_cantilever_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)
_ = problem.solve()
assert problem.solver.getConvergedReason() > 0


TenSpace = functionspace(msh, ("DG", 0, (msh.geometry.dim, msh.geometry.dim)))

eps_func = Function(TenSpace)
eps_expr = ufl.as_tensor(epsilon(uh))
eps_func.name = "strain"

eps_E = Expression(eps_expr, TenSpace.element.interpolation_points)
eps_func.interpolate(eps_E)

sig_func = Function(TenSpace)
sig_expr = ufl.as_tensor(sigma(uh))
sig_func.name = "stress"
sig_E = Expression(sig_expr, TenSpace.element.interpolation_points)
sig_func.interpolate(sig_E)


out_dir = "out_cantilever"
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
