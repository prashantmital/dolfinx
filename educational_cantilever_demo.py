"""
Educational Cantilever Beam Demo for DOLFINx

A well-structured demonstration of cantilever beam stress analysis that clearly
separates the key concepts in numerical methods for finite element analysis.

This script demonstrates the 6 key stages of numerical methods:
1. GEOMETRY - Define the physical domain and its properties
2. DISCRETIZATION - Create a computational mesh
3. BOUNDARY CONDITIONS - Apply physical constraints
4. MATRIX CONSTRUCTION - Build the linear system
5. LINEAR SOLVER - Solve the system of equations
6. ANALYSIS & RESULTS - Post-process and visualize

Author: Devin AI
Educational refactor for learning numerical methods
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


class BeamGeometry:
    """
    STAGE 1: GEOMETRY
    
    Defines the physical domain - the shape, size, and material properties
    of our cantilever beam in 3D space.
    """
    
    def __init__(self, length=2.0, width=0.2, height=0.1):
        """
        Create a rectangular cantilever beam geometry.
        
        Args:
            length (float): Beam length in meters (x-direction)
            width (float): Beam width in meters (y-direction) 
            height (float): Beam height in meters (z-direction)
        """
        self.length = length
        self.width = width
        self.height = height
        
        self.corner_min = np.array([0.0, 0.0, 0.0])
        self.corner_max = np.array([length, width, height])
        
        print(f"📐 GEOMETRY: Cantilever beam {length}m × {width}m × {height}m")
        print(f"   Domain: [{self.corner_min[0]:.1f}, {self.corner_max[0]:.1f}] × "
              f"[{self.corner_min[1]:.1f}, {self.corner_max[1]:.1f}] × "
              f"[{self.corner_min[2]:.1f}, {self.corner_max[2]:.1f}]")
    
    def get_cross_sectional_area(self):
        """Calculate the cross-sectional area (width × height)."""
        return self.width * self.height
    
    def get_second_moment_of_area(self):
        """Calculate second moment of area for beam bending theory."""
        return self.width * self.height**3 / 12


class MaterialProperties:
    """
    Material properties for the beam - defines how the material responds
    to stress and strain.
    """
    
    def __init__(self, youngs_modulus=200e9, poissons_ratio=0.3):
        """
        Steel material properties.
        
        Args:
            youngs_modulus (float): Young's modulus in Pa (stiffness)
            poissons_ratio (float): Poisson's ratio (lateral contraction)
        """
        self.E = youngs_modulus
        self.nu = poissons_ratio
        
        self.mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        
        print(f"🔧 MATERIAL: Steel (E = {self.E/1e9:.0f} GPa, ν = {self.nu})")
        print(f"   Lamé parameters: μ = {self.mu/1e9:.1f} GPa, λ = {self.lmbda/1e9:.1f} GPa")


def create_mesh(geometry, mesh_density="coarse"):
    """
    STAGE 2: DISCRETIZATION
    
    Convert the continuous physical domain into a discrete computational mesh
    of finite elements (tetrahedra in 3D).
    
    Args:
        geometry (BeamGeometry): The beam geometry to discretize
        mesh_density (str): "coarse", "medium", or "fine"
    
    Returns:
        dolfinx.mesh.Mesh: The computational mesh
    """
    print(f"\n🔲 DISCRETIZATION: Creating {mesh_density} tetrahedral mesh...")
    
    density_map = {
        "coarse": (20, 4, 2),   # Fast for demos
        "medium": (40, 8, 4),   # Balanced accuracy/speed
        "fine": (80, 16, 8)     # High accuracy
    }
    
    nx, ny, nz = density_map.get(mesh_density, density_map["coarse"])
    
    mesh = create_box(
        MPI.COMM_WORLD,
        [geometry.corner_min, geometry.corner_max],
        (nx, ny, nz),
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )
    
    num_cells = mesh.topology.index_map(3).size_global
    print(f"   Mesh created: {nx}×{ny}×{nz} = {num_cells} tetrahedral elements")
    
    return mesh


def create_function_space(mesh):
    """
    Create the finite element function space for displacement.
    
    We use vector-valued Lagrange P1 elements (linear functions) with
    3 components for 3D displacement (u_x, u_y, u_z).
    """
    print(f"\n🎯 FUNCTION SPACE: Vector Lagrange P1 elements")
    
    V = functionspace(mesh, ("Lagrange", 1, (3,)))  # 3D displacement
    num_dofs = V.dofmap.index_map.size_global
    
    print(f"   Degrees of freedom: {num_dofs} (3 displacement components per node)")
    
    return V


def apply_boundary_conditions(mesh, function_space, geometry):
    """
    STAGE 3: BOUNDARY CONDITIONS
    
    Apply physical constraints to the mathematical problem:
    - Fixed support at x=0 (cantilever root)
    - Free boundary everywhere else
    
    Args:
        mesh: The computational mesh
        function_space: The finite element function space
        geometry: The beam geometry
    
    Returns:
        list: Boundary condition objects
    """
    print(f"\n🔒 BOUNDARY CONDITIONS: Applying physical constraints...")
    
    fixed_facets = locate_entities_boundary(
        mesh, dim=2, marker=lambda x: np.isclose(x[0], 0.0)
    )
    
    print(f"   Fixed support: {len(fixed_facets)} boundary facets at x = 0")
    print(f"   Constraint: u_x = u_y = u_z = 0 (no displacement)")
    
    bc = dirichletbc(
        np.zeros(3, dtype=PETSc.ScalarType),  # Zero displacement vector
        locate_dofs_topological(function_space, entity_dim=2, entities=fixed_facets),
        V=function_space
    )
    
    return [bc]


def construct_linear_system(function_space, material, load_force, geometry):
    """
    STAGE 4: MATRIX CONSTRUCTION
    
    Build the linear system Ax = b from the weak form of linear elasticity:
    - A: Stiffness matrix (relates forces to displacements)
    - b: Load vector (external forces)
    
    Args:
        function_space: Finite element function space
        material: Material properties
        load_force: Applied load in Newtons
        geometry: Beam geometry
    
    Returns:
        tuple: (bilinear_form, linear_form) for the weak formulation
    """
    print(f"\n🔨 MATRIX CONSTRUCTION: Building linear elasticity system...")
    
    u = ufl.TrialFunction(function_space)  # Unknown displacement
    v = ufl.TestFunction(function_space)   # Virtual displacement (test function)
    
    def stress_tensor(displacement):
        """
        Compute stress tensor from displacement using linear elasticity.
        σ = 2μ ε + λ tr(ε) I, where ε = ½(∇u + ∇u^T)
        """
        strain = ufl.sym(ufl.grad(displacement))  # Symmetric gradient (strain)
        return (2.0 * material.mu * strain + 
                material.lmbda * ufl.tr(strain) * ufl.Identity(3))
    
    a = form(ufl.inner(stress_tensor(u), ufl.grad(v)) * ufl.dx)
    
    distributed_load = load_force / geometry.get_cross_sectional_area()
    f = ufl.as_vector([0.0, 0.0, -distributed_load])  # Downward force
    L = form(ufl.inner(f, v) * ufl.dx)
    
    print(f"   Weak form: Find u such that a(u,v) = L(v) for all v")
    print(f"   Applied load: {load_force} N distributed over {geometry.get_cross_sectional_area():.4f} m²")
    print(f"   Load density: {distributed_load:.0f} N/m²")
    
    return a, L


def solve_linear_system(bilinear_form, linear_form, boundary_conditions, function_space):
    """
    STAGE 5: LINEAR SOLVER
    
    Solve the linear system Ax = b using PETSc solvers:
    1. Assemble the stiffness matrix A and load vector b
    2. Apply boundary conditions
    3. Solve using direct LU factorization
    
    Args:
        bilinear_form: The bilinear form a(u,v)
        linear_form: The linear form L(v)
        boundary_conditions: List of boundary conditions
        function_space: The finite element function space
    
    Returns:
        Function: The computed displacement field
    """
    print(f"\n⚙️ LINEAR SOLVER: Assembling and solving the system...")
    
    print("   Assembling stiffness matrix A...")
    A = assemble_matrix(bilinear_form, bcs=boundary_conditions)
    A.assemble()
    
    print("   Assembling load vector b...")
    b = assemble_vector(linear_form)
    apply_lifting(b, [bilinear_form], bcs=[boundary_conditions])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    for bc in boundary_conditions:
        bc.set(b.array_w)
    
    print("   Setting up LU direct solver...")
    solver = PETSc.KSP().create(function_space.mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)  # Direct solver
    solver.getPC().setType(PETSc.PC.Type.LU)  # LU factorization
    
    print("   Solving Ax = b...")
    uh = Function(function_space)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    
    print("   ✅ Linear system solved successfully!")
    
    return uh


def analyze_results(displacement_solution, material, geometry, load_force):
    """
    STAGE 6: ANALYSIS & RESULTS
    
    Post-process the solution to extract meaningful engineering quantities:
    - Displacement field
    - Stress and strain tensors
    - Von Mises stress (failure criterion)
    - Comparison with analytical beam theory
    
    Args:
        displacement_solution: The computed displacement field
        material: Material properties
        geometry: Beam geometry
        load_force: Applied load
    
    Returns:
        dict: Analysis results and computed fields
    """
    print(f"\n📊 ANALYSIS & RESULTS: Post-processing the solution...")
    
    max_displacement = np.max(np.abs(displacement_solution.x.array))
    displacement_norm = la.norm(displacement_solution.x)
    
    print(f"   Maximum displacement: {max_displacement*1000:.2f} mm")
    print(f"   Displacement norm: {displacement_norm:.6e} m")
    
    def stress_tensor(u):
        strain = ufl.sym(ufl.grad(u))
        return (2.0 * material.mu * strain + 
                material.lmbda * ufl.tr(strain) * ufl.Identity(3))
    
    sigma = stress_tensor(displacement_solution)
    
    sigma_dev = sigma - (1/3) * ufl.tr(sigma) * ufl.Identity(3)  # Deviatoric stress
    von_mises = ufl.sqrt((3/2) * ufl.inner(sigma_dev, sigma_dev))
    
    W = functionspace(displacement_solution.function_space.mesh, ("Discontinuous Lagrange", 0))
    
    von_mises_expr = Expression(von_mises, W.element.interpolation_points)
    von_mises_func = Function(W)
    von_mises_func.interpolate(von_mises_expr)
    
    I = geometry.get_second_moment_of_area()
    analytical_displacement = load_force * geometry.length**3 / (3 * material.E * I)
    analytical_stress = load_force * geometry.length * (geometry.height/2) / I
    
    print(f"\n📐 ANALYTICAL COMPARISON (Euler-Bernoulli beam theory):")
    print(f"   Analytical max displacement: {analytical_displacement*1000:.2f} mm")
    print(f"   Analytical max stress: {analytical_stress/1e6:.1f} MPa")
    print(f"   FEM vs Analytical displacement ratio: {max_displacement/analytical_displacement:.3f}")
    
    return {
        'displacement': displacement_solution,
        'von_mises_stress': von_mises_func,
        'max_displacement': max_displacement,
        'analytical_displacement': analytical_displacement,
        'analytical_stress': analytical_stress
    }


def save_results(results, output_dir="educational_results"):
    """
    Save the computed fields to files for visualization in ParaView.
    
    Args:
        results: Dictionary containing computed fields
        output_dir: Directory to save results
    """
    print(f"\n💾 SAVING RESULTS: Writing to {output_dir}/...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    mesh = results['displacement'].function_space.mesh
    
    with XDMFFile(mesh.comm, f"{output_dir}/displacement.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(results['displacement'])
    
    with XDMFFile(mesh.comm, f"{output_dir}/von_mises_stress.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(results['von_mises_stress'])
    
    print(f"   ✅ Results saved:")
    print(f"   - {output_dir}/displacement.xdmf")
    print(f"   - {output_dir}/von_mises_stress.xdmf")
    print(f"   📈 Open in ParaView for 3D visualization")


def main():
    """
    Main educational demo showing the 6 stages of numerical methods
    for finite element analysis of a cantilever beam.
    """
    print("=" * 70)
    print("🎓 EDUCATIONAL CANTILEVER BEAM DEMO")
    print("   Learning the 6 Stages of Numerical Methods")
    print("=" * 70)
    
    geometry = BeamGeometry(length=2.0, width=0.2, height=0.1)
    material = MaterialProperties(youngs_modulus=200e9, poissons_ratio=0.3)
    load_force = 1000.0  # 1000 N downward
    
    mesh = create_mesh(geometry, mesh_density="coarse")
    function_space = create_function_space(mesh)
    
    boundary_conditions = apply_boundary_conditions(mesh, function_space, geometry)
    
    bilinear_form, linear_form = construct_linear_system(
        function_space, material, load_force, geometry
    )
    
    displacement_solution = solve_linear_system(
        bilinear_form, linear_form, boundary_conditions, function_space
    )
    
    results = analyze_results(displacement_solution, material, geometry, load_force)
    save_results(results)
    
    print("\n" + "=" * 70)
    print("🎉 EDUCATIONAL DEMO COMPLETED SUCCESSFULLY!")
    print("   You have learned the 6 key stages of numerical methods:")
    print("   1. ✅ Geometry definition")
    print("   2. ✅ Domain discretization") 
    print("   3. ✅ Boundary condition application")
    print("   4. ✅ Matrix construction")
    print("   5. ✅ Linear system solution")
    print("   6. ✅ Results analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
