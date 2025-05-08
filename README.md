# ME700_FinalProject
# Simulation of Elastin Fiber Stretching  
This code simulates the elastin fiber stretching process. I used pygmsh and gmsh to mesh the 3 columns, one is left glass optical fiber, connecting the elastin fiber from the top, another is elastin fiber, the other one is the right glass optical fiber, which is conecting to the bottom of the elastin fiber. See below schemetic for the device set up.  
After meshing them, I imported the .xdml file to FENicsX for finite element analysis, including setting up the boundary condition, tying the three columns together, and applying both linear elasicity model and Neo-Hookean model.  
Finally, pyvista show the whole simulation process with a .gif file.


Let's start with setting up FENicsX environment
```bash
module load miniconda
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install gmsh
pip install pygmsh
pip install h5py
```
To run the code
```bash
python simulation.py
```
See process.gif for the simulation result.  
The comparison between this simulation and the experimental data:  
