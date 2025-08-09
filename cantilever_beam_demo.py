"""
Cantilever Beam Stress-Strain Analysis Demo

This demo script simulates the stress-strain distribution in a cantilevered beam
with an end load using DOLFINx. The beam has a rectangular cross-section and is
made of steel.

Key features:
- 3D cantilever beam with rectangular cross-section
- Steel material properties
- Fixed boundary condition at one end
- Point load at the free end
- Visualization of displacement, stress, and strain distributions

Author: Devin AI
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

dtype = PETSc.ScalarType

def create_cantilever_mesh(length=2.0, width=0.2, height=0.1, nx=40, ny=8, nz=4):
    """
    Create a rectangular cantilever beam mesh.
    
    Parameters:
    - length: beam length (m)
    - width: beam width (m) 
    - height: beam height (m)
    - nx, ny, nz: number of elements in each direction
    """
    return create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([length, width, height])],
        (nx, ny, nz),
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )

def steel_material_properties():
    """
    Return material properties for structural steel.
    """
    E = 200e9  # Young's modulus (Pa) - typical for steel
    nu = 0.3   # Poisson's ratio
    return E, nu

def compute_lame_parameters(E, nu):
    """
    Compute Lamé parameters from Young's modulus and Poisson's ratio.
    """
    mu = E / (2.0 * (1.0 + nu))  # Shear modulus
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # First Lamé parameter
    return mu, lmbda

def stress_tensor(u, mu, lmbda):
    """
    Compute the stress tensor for linear elasticity.
    """
    return 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(len(u))

def strain_tensor(u):
    """
    Compute the strain tensor.
    """
    return ufl.sym(ufl.grad(u))

def von_mises_stress(sigma):
    """
    Compute Von Mises stress from stress tensor.
    """
    sigma_dev = sigma - (1/3) * ufl.tr(sigma) * ufl.Identity(len(sigma))
    return ufl.sqrt((3/2) * ufl.inner(sigma_dev, sigma_dev))

def main():
    """
    Main function to run the cantilever beam analysis.
    """
    print("Starting cantilever beam stress-strain analysis...")
    
    length = 2.0    # 2 meters long
    width = 0.2     # 20 cm wide  
    height = 0.1    # 10 cm high
    
    print("Creating mesh...")
    mesh = create_cantilever_mesh(length, width, height, nx=40, ny=8, nz=4)
    
    E, nu = steel_material_properties()
    mu, lmbda = compute_lame_parameters(E, nu)
    print(f"Material properties - E: {E/1e9:.1f} GPa, nu: {nu}")
    
    V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    print(f"Function space created with {V.dofmap.index_map.size_global} DOFs")
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    sigma_u = stress_tensor(u, mu, lmbda)
    a = form(ufl.inner(sigma_u, ufl.grad(v)) * ufl.dx)
    
    load_magnitude = 1000.0  # 1000 N downward force
    f = ufl.as_vector([0.0, 0.0, -load_magnitude/(width*height)])  # Distributed over cross-section
    
    L = form(ufl.inner(f, v) * ufl.dx)
    
    print("Applying boundary conditions...")
    fixed_facets = locate_entities_boundary(
        mesh, dim=2, marker=lambda x: np.isclose(x[0], 0.0)
    )
    
    bc = dirichletbc(
        np.zeros(3, dtype=dtype), 
        locate_dofs_topological(V, entity_dim=2, entities=fixed_facets), 
        V=V
    )
    
    print("Assembling system...")
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bc.set(b.array_w)
    
    print("Setting up solver...")
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    print("Solving linear system...")
    uh = Function(V)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    
    print("Computing stress and strain...")
    
    sigma_h = stress_tensor(uh, mu, lmbda)
    
    epsilon_h = strain_tensor(uh)
    
    sigma_vm = von_mises_stress(sigma_h)
    
    epsilon_vm = ufl.sqrt((2/3) * ufl.inner(epsilon_h, epsilon_h))
    
    W_tensor = functionspace(mesh, ("Discontinuous Lagrange", 0, (3, 3)))
    W_scalar = functionspace(mesh, ("Discontinuous Lagrange", 0))
    
    sigma_expr = Expression(sigma_h, W_tensor.element.interpolation_points)
    sigma_func = Function(W_tensor)
    sigma_func.interpolate(sigma_expr)
    
    epsilon_expr = Expression(epsilon_h, W_tensor.element.interpolation_points)
    epsilon_func = Function(W_tensor)
    epsilon_func.interpolate(epsilon_expr)
    
    sigma_vm_expr = Expression(sigma_vm, W_scalar.element.interpolation_points)
    sigma_vm_func = Function(W_scalar)
    sigma_vm_func.interpolate(sigma_vm_expr)
    
    epsilon_vm_expr = Expression(epsilon_vm, W_scalar.element.interpolation_points)
    epsilon_vm_func = Function(W_scalar)
    epsilon_vm_func.interpolate(epsilon_vm_expr)
    
    print("Saving results...")
    
    import os
    os.makedirs("cantilever_results", exist_ok=True)
    
    with XDMFFile(mesh.comm, "cantilever_results/displacement.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)
    
    with XDMFFile(mesh.comm, "cantilever_results/von_mises_stress.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(sigma_vm_func)
    
    with XDMFFile(mesh.comm, "cantilever_results/von_mises_strain.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(epsilon_vm_func)
    
    with XDMFFile(mesh.comm, "cantilever_results/stress_tensor.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(sigma_func)
    
    with XDMFFile(mesh.comm, "cantilever_results/strain_tensor.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(epsilon_func)
    
    displacement_norm = la.norm(uh.x)
    max_displacement = np.max(np.abs(uh.x.array))
    
    if mesh.comm.rank == 0:
        print("\n" + "="*60)
        print("CANTILEVER BEAM ANALYSIS RESULTS")
        print("="*60)
        print(f"Beam dimensions: {length}m × {width}m × {height}m")
        print(f"Applied load: {load_magnitude} N (downward at free end)")
        print(f"Material: Steel (E = {E/1e9:.1f} GPa, ν = {nu})")
        print(f"Mesh: {mesh.topology.index_map(3).size_global} elements")
        print(f"DOFs: {V.dofmap.index_map.size_global}")
        print(f"Displacement norm: {displacement_norm:.6e} m")
        print(f"Maximum displacement: {max_displacement:.6e} m")
        print("\nOutput files saved to 'cantilever_results/' directory:")
        print("- displacement.xdmf: Displacement field")
        print("- von_mises_stress.xdmf: Von Mises stress distribution")
        print("- von_mises_strain.xdmf: Von Mises strain distribution")
        print("- stress_tensor.xdmf: Full stress tensor")
        print("- strain_tensor.xdmf: Full strain tensor")
        print("\nUse ParaView or similar software to visualize the results.")
        print("="*60)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
