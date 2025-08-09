"""
Simple Cantilever Beam Demo for DOLFINx

A minimal, easy-to-understand demonstration of cantilever beam stress analysis
using DOLFINx finite element library.

This script demonstrates:
- 3D cantilever beam with rectangular cross-section
- Steel material properties (E=200 GPa, ν=0.3)
- Fixed boundary condition at one end
- Point load at free end
- Stress and displacement computation

To run: python simple_cantilever_demo.py
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import (
    Expression,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary

LENGTH = 2.0    # 2 meters
WIDTH = 0.2     # 20 cm
HEIGHT = 0.1    # 10 cm
LOAD = 1000.0   # 1000 N downward

E = 200e9       # Young's modulus (Pa)
nu = 0.3        # Poisson's ratio
mu = E / (2.0 * (1.0 + nu))                    # Shear modulus
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Lamé parameter

def main():
    print("Cantilever Beam Stress Analysis Demo")
    print("====================================")
    
    print("Creating mesh...")
    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([LENGTH, WIDTH, HEIGHT])],
        (20, 4, 2),  # Coarse mesh for demo
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )
    
    V = functionspace(mesh, ("Lagrange", 1, (3,)))
    print(f"DOFs: {V.dofmap.index_map.size_global}")
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    def stress(u):
        return 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3)
    
    a = form(ufl.inner(stress(u), ufl.grad(v)) * ufl.dx)
    
    f = ufl.as_vector([0.0, 0.0, -LOAD/(WIDTH*HEIGHT)])
    L = form(ufl.inner(f, v) * ufl.dx)
    
    print("Applying boundary conditions...")
    fixed_facets = locate_entities_boundary(
        mesh, dim=2, marker=lambda x: np.isclose(x[0], 0.0)
    )
    bc = dirichletbc(
        np.zeros(3, dtype=PETSc.ScalarType),
        locate_dofs_topological(V, entity_dim=2, entities=fixed_facets),
        V=V
    )
    
    print("Solving...")
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bc.set(b.array_w)
    
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    uh = Function(V)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    
    print("Computing stress...")
    sigma = stress(uh)
    
    sigma_dev = sigma - (1/3) * ufl.tr(sigma) * ufl.Identity(3)
    von_mises = ufl.sqrt((3/2) * ufl.inner(sigma_dev, sigma_dev))
    
    W = functionspace(mesh, ("Discontinuous Lagrange", 0))
    von_mises_expr = Expression(von_mises, W.element.interpolation_points)
    von_mises_func = Function(W)
    von_mises_func.interpolate(von_mises_expr)
    
    print("Saving results...")
    import os
    os.makedirs("results", exist_ok=True)
    
    with XDMFFile(mesh.comm, "results/displacement.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)
    
    with XDMFFile(mesh.comm, "results/von_mises_stress.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(von_mises_func)
    
    max_disp = np.max(np.abs(uh.x.array))
    disp_norm = la.norm(uh.x)
    
    if mesh.comm.rank == 0:
        print("\nResults:")
        print(f"Maximum displacement: {max_disp*1000:.2f} mm")
        print(f"Displacement norm: {disp_norm:.6e} m")
        
        I = WIDTH * HEIGHT**3 / 12  # Second moment of area
        analytical_disp = LOAD * LENGTH**3 / (3 * E * I)
        print(f"Analytical displacement: {analytical_disp*1000:.2f} mm")
        
        print("\nOutput files:")
        print("- results/displacement.xdmf")
        print("- results/von_mises_stress.xdmf")
        print("Open in ParaView for visualization")
    
    print("Demo completed!")

if __name__ == "__main__":
    main()
