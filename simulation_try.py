from mpi4py import MPI
import numpy as np
import pygmsh
import gmsh
import meshio
from dolfinx import mesh, fem, io, default_scalar_type, geometry
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import dirichletbc, locate_dofs_topological, Constant, locate_dofs_topological
from dolfinx.mesh import CellType, locate_entities_boundary, create_mesh
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from dolfinx_mpc import MultiPointConstraint
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from scipy.spatial import cKDTree
from dolfinx.plot import vtk_mesh
import pyvista
import matplotlib.pyplot as plt


# Physical parameters
# Optical fiber
'''
Length of left optical fiber: L_optical_L = 13.51e-3 [m]
Radius of right optical fiber: radius_L = 125e-6/2 [m]
Length of right optical fiber: L_optical_R = 13.72e-3 [m]
Radius of right optical fiber: radius_R = 80e-6/2 [m]
Young's modulus: E_optical = 70e9 [Pa]
Possion ratio: nu_optical = 0.17 
Left end displacement (for stretching): displacement = 15e-3 [m]
'''
L_optical_L = 13.51e-3  
radius_L = 125e-6 / 2   
L_optical_R = 13.72e-3  
radius_R = 80e-6 / 2    
E_optical = 70e9        
nu_optical = 0.17
displacement = 15e-3    
# Elastin fiber
'''
Elastin initial length: L_elastin = 360e-6 [m]
Elastin initial radius: radius_elastin = 2.5e-6/2 [m]
Young's modulus: E_elastin = 0.75e6 [Pa]
Poisson's ratio (assume nearly incompresibe;): nu_elastin = 0.49
'''
L_elastin = 360e-6      
radius_elastin = 2.5e-6 / 2  
E_elastin = 0.75e6        
nu_elastin = 0.49

# Mesh generation with pygmsh
'''
Create a 3D mesh of columns:
Dimensions: add_cylinder(start point=[], axis=[], radius) 
'''
with pygmsh.occ.Geometry() as geom:
    min_size = min(radius_elastin / 50, radius_L / 30, radius_R / 30)
    max_size = max(radius_L / 5, radius_R / 5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for robustness
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D
    
    # Left optical fiber
    left_fiber = geom.add_cylinder(
        [0, 0, 0], [L_optical_L, 0, 0], radius_L)
    geom.add_physical(left_fiber, label="left_fiber")
    # Elastin fiber
    elastin = geom.add_cylinder(
        [L_optical_L, radius_L/2, 0], [0, -L_elastin, 0], radius_elastin)
    geom.add_physical(elastin, label="elastin")
    # Right optical fiber
    right_fiber = geom.add_cylinder(
        [L_optical_L, -radius_L/2 - L_elastin + radius_R/2, 0], [L_optical_R, 0, 0], radius_R)
    geom.add_physical(right_fiber, label="right_fiber")
    
    geom.in_volume(left_fiber, left_fiber)
    geom.in_volume(elastin, elastin)
    geom.in_volume(right_fiber, right_fiber)
    
    # Generate mesh
    mesh = geom.generate_mesh(dim=3, verbose=True)
    # Save mesh
    gmsh.write("fiber_assembly_cylindrical.msh")

# Convert mesh to XDMF
mesh_from_file = meshio.read("fiber_assembly_cylindrical.msh")
cells = mesh_from_file.cells_dict.get("tetra", None)
if cells is None:
    raise ValueError("No tetrahedral cells found in the mesh. Check Gmsh output.")

tetra_mesh = meshio.Mesh(
    points=mesh_from_file.points,
    cells={"tetra": cells},
    cell_data={"gmsh:physical": [mesh_from_file.cell_data_dict["gmsh:physical"]["tetra"]]},
)
meshio.write("fiber_assembly_cylindrical.xdmf", tetra_mesh)

# Import mesh to DOLFINx
with XDMFFile(MPI.COMM_WORLD, "fiber_assembly_cylindrical.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")

# Vector function space
V = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim,)))
# Subdomains for material regions
left_marker, elastin_marker, right_marker = 1, 2, 3
tdim = domain.topology.dim
subdomain_cells = {
    left_marker: np.where(ct.values == left_marker)[0],
    elastin_marker: np.where(ct.values == elastin_marker)[0],
    right_marker: np.where(ct.values == right_marker)[0]
}

# Material model
ds = ufl.Measure("ds", domain=domain)
dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
dx_left = dx(left_marker)
dx_elastin = dx(elastin_marker)
dx_right = dx(right_marker)

mu_optical = E_optical / (2 * (1 + nu_optical))
lambda_optical = E_optical * nu_optical / ((1 + nu_optical) * (1 - 2 * nu_optical))
mu_elastin = E_elastin / (2 * (1 + nu_elastin))
kappa_elastin = E_elastin / (3 * (1 - 2 * nu_elastin))
# Linear elastic model: optical fibers
def linear_elastic(u, E, nu):
    I = ufl.Identity(3)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    eps = ufl.sym(ufl.grad(u))
    return 2 * mu * eps + lam * ufl.tr(eps) * I
# Neo-Hookean model (hyperelasticity): elastin
def neo_hookean(u, E, nu):
    I = ufl.Identity(3)
    F = I + ufl.grad(u)
    C = F.T * F
    J = ufl.det(F)
    J_safe = ufl.conditional(ufl.ge(J, 1e-12), J, 1e-12)
    lnJ = ufl.ln(J_safe)
    mu = E / (2 * (1 + nu))
    kappa = E / (3 * (1 - 2 * nu))
    return (mu/J) * (C - I) + (kappa/J) * lnJ * I

# Boundary conditions
'''
Fixed the right end of right optical fiber.
Set a y-displacement at the left end of the left optical fiber for stretching elastin.
'''
def fixed_boundary(x):
    return np.isclose(x[0], L_optical_L + L_optical_R, atol=1e-6)

facets_fixed = locate_entities_boundary(domain, domain.topology.dim - 1, fixed_boundary)
dofs_fixed = locate_dofs_topological(V, domain.topology.dim - 1, facets_fixed)
bc_fixed = dirichletbc(np.zeros(domain.topology.dim), dofs_fixed, V)

def top_displacement_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)

facets_top = locate_entities_boundary(domain, domain.topology.dim - 1, top_displacement_boundary)
dofs_top = locate_dofs_topological(V, domain.topology.dim - 1, facets_top)
# Tie constrains for elastin and optical fiber
'''
Conncet elastin with the right end of left optical fiber (top) 
and the left end of right optical fiber (bottom)
'''
mpc = MultiPointConstraint(V)
def find_matching_dofs(dofs_slave, dofs_master, x_slave, x_master, tol=1e-8):
    tree = cKDTree(x_master.T)
    matched = []
    for i, x in enumerate(x_slave.T):
        dist, j = tree.query(x, distance_upper_bound=tol)
        if dist != np.inf:
            matched.append((dofs_slave[i], dofs_master[j]))
    return matched

def top_connected_boundary(x):
    return np.isclose(x[1], radius_L/2, atol=radius_L * 2)

facets_top_connect = locate_entities_boundary(domain, domain.topology.dim - 1, top_connected_boundary)
dofs_top_connect = locate_dofs_topological(V, domain.topology.dim - 1, facets_top_connect)
x = domain.geometry.x.T
slave_coords_top = x[:, dofs_top_connect]
master_coords_top = x[:, dofs_top]
matched_top = find_matching_dofs(dofs_top_connect, dofs_top, slave_coords_top, master_coords_top)
rank = V.mesh.comm.rank
for slave, master in matched_top:
    mpc.add_constraint(V, slaves=np.array([slave], dtype=np.int32),
        masters=np.array([master], dtype=np.int64),
        coeffs=np.array([1.0], dtype=np.float64),
        owners=np.array([rank], dtype=np.int32),
        offsets=np.array([0, 1], dtype=np.int32)
    )
    
def bottom_connected_boundary(x):
    return np.isclose(x[1], -radius_L/2 - L_elastin + radius_R/2, atol=radius_R * 2)

facets_bottom_connect = locate_entities_boundary(domain, domain.topology.dim - 1, bottom_connected_boundary)
dofs_bottom_connect = locate_dofs_topological(V, domain.topology.dim - 1, facets_bottom_connect)
slave_coords_bottom = x[:, dofs_bottom_connect]
master_coords_bottom = x[:, dofs_fixed]
matched_bottom = find_matching_dofs(dofs_bottom_connect, dofs_fixed, slave_coords_bottom, master_coords_bottom)
for slave, master in matched_bottom:
    mpc.add_constraint(V, slaves=np.array([slave], dtype=np.int32),
        masters=np.array([master], dtype=np.int64),
        coeffs=np.array([1.0], dtype=np.float64),
        owners=np.array([rank], dtype=np.int32),
        offsets=np.array([0, 1], dtype=np.int32)
    )

mpc.finalize()
 
# Displacement steps
displacement_steps = np.linspace(0, displacement, 10)
solutions = []

# Variational problem (Nonlinear)
u = fem.Function(V)
v = ufl.TestFunction(V)
def F(u, v):   
    # Stress tensors
    sigma_left = linear_elastic(u, E_optical, nu_optical)
    sigma_elastin = neo_hookean(u, E_elastin, nu_elastin)
    sigma_right = linear_elastic(u, E_optical, nu_optical)
    
    F_left = ufl.inner(sigma_left, ufl.grad(v)) * dx_left
    F_elastin = ufl.inner(sigma_elastin, ufl.grad(v)) * dx_elastin
    F_right = ufl.inner(sigma_right, ufl.grad(v)) * dx_right
    
    return F_left + F_elastin + F_right

for i, disp in enumerate(displacement_steps):
    print(f"\nStep {i+1}/{len(displacement_steps)}: Displacement = {disp:.2e} m")

    bc_top = dirichletbc(np.array([0, disp, 0]), dofs_top, V)
    problem = NonlinearProblem(F(u, v), u, bcs=[bc_fixed, bc_top])     
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = 1e-6
    solver.atol = 1e-8
    solver.max_it = 50
    
    try:
        n, converged = solver.solve(u)
        if converged:
            print(f"Converged in {n} iterations")
        else:
            print(f"Warning: Did not converge")
        solutions.append(u.copy())
        
        with io.XDMFFile(domain.comm, f"step_{i}.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u)
            
    except Exception as e:
        print(f"Error at displacement {disp}: {e}")
        if solutions:
            solutions.append(solutions[-1].copy())
        else:
            solutions.append(u.copy())  
        
# Analytical solution
# Tip displacement
def get_tip_displacement(uh, tip_loc):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, np.array(tip_loc, dtype=np.float64))
    cells = geometry.compute_colliding_cells(domain, cell_candidates, np.array(tip_loc, dtype=np.float64))
    if len(cells.array) > 0:
        return uh.eval(tip_loc, cells.array[0])
    else:
        print(f"Warning: No cell found at point {tip_loc}")
        return np.zeros(domain.topology.dim, dtype=default_scalar_type)

print("\nFinal tip displacement (simulation):")
top_connection = [L_optical_L - 1e-6, 0.0, 0.0]
elastin_connection = [L_optical_L, -L_elastin + 1e-6, 0.0]
left_tip_disp = get_tip_displacement(u, top_connection)
right_tip_disp = get_tip_displacement(u, elastin_connection)
print(f"Left optical fiber tip: {left_tip_disp[1]:.2e} m")
print(f"Right optical fiber tip: {right_tip_disp[1]:.2e} m")
# Elastin strain
elastin_top = get_tip_displacement(solutions[-1], [L_optical_L, 0.0, 0.0])
elastin_bottom = get_tip_displacement(solutions[-1], [L_optical_L, -L_elastin, 0.0])
strain = (((elastin_top[1] - elastin_bottom[1]) - L_elastin) / L_elastin) * 100
print(f"Elastin strain: {strain:.2f} % (Original length: {L_elastin:.2e} m)")  
# Experimental data
left_tip_disp_experiment = 9.22e-4 # [m]
right_tip_disp_experiment = 6.86e-4 # [m]
elastin_elong = displacement - (left_tip_disp_experiment + right_tip_disp_experiment)
strain_experiment = ((elastin_elong - L_elastin)/L_elastin) * 100
print("\nExperimental data:")
print(f"Left optical fiber tip: {9.22e-4:.2e} m")
print(f"Right optical fiber tip: {6.86e-4:.2e} m")
print(f"Elastin strain: {strain_experiment:.2f} %")
# Compare experimental data and simulation result
left_error = ((left_tip_disp_experiment - left_tip_disp[1])/left_tip_disp_experiment) * 100
right_error = ((right_tip_disp_experiment - right_tip_disp[1])/right_tip_disp_experiment) * 100
strain_error = ((strain_experiment - strain)/strain_experiment) *100
print("\nError (%):")
print(f"Left optical fiber: {left_error:.2f}")
print(f"Right optical fiber: {right_error:.2f}")
print(f"Elastin strain: {strain_error:.2f}")

# Create gif file
if MPI.COMM_WORLD.rank == 0:
    pyvista.start_xvfb()

pyvista.start_xvfb()
plotter = pyvista.Plotter()
plotter.open_gif("simulation.gif", fps=3)

topology, cells, geometry = vtk_mesh(V)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

values = np.zeros((geometry.shape[0], 3))
function_grid["u"] = values
function_grid.set_active_vectors("u")

warped = function_grid.warp_by_vector("u", factor=1.0)
warped.set_active_vectors("u")
actor = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, displacement])

Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
for i, u in enumerate(solutions):
    u.x.scatter_forward()

    values[:, :3] = u.x.array.reshape((geometry.shape[0], 3))
    function_grid.point_data["u"] = values[:, :3]
    warped_n = function_grid.warp_by_vector("u", factor=1.0)
    warped.points[:, :] = warped_n.points
    
    mag_interp = np.linalg.norm(values[:, :3], axis=1)
    warped.point_data["mag"] = mag_interp
    warped.set_active_scalars("mag")

    plotter.update_scalar_bar_range([0, displacement])
    plotter.write_frame()

plotter.close()

# Visualization   
with io.XDMFFile(domain.comm, "final_deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u) 
