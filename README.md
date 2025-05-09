# ME700_FinalProject
## Simulation of Elastin Fiber Stretching  
This code simulates the elastin fiber stretching process. Pygmsh and gmsh were used to mesh structure. The structure consist of 3 columns, one is left glass optical fiber, connecting the elastin fiber from the top, another is elastin fiber, and the other one is the right glass optical fiber, which is conecting to the bottom of the elastin fiber. See below schematic for the device set up.  
![schematic of stretching device.jpg](https://github.com/Esther918/ME700_FinalProject/blob/main/schematic%20of%20stretching%20device.jpg)
The resulting .xdmf file was imported into DOLFINx, where boundary conditions were defined, the columns were tied together, and different material models were applied, that is linear elasticity model (for optical fiber) and a Neo-Hookean hyperelastic model (for elastin fiber).
Finally, PyVista was used to visualize the simulation and create an animated .gif showing the deformation process.   

### Reproducibility  
Let's start with setting up FENicsX environment  
```bash
module load miniconda
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
pip install pygmsh
pip install h5py
```
To run the code
```bash
python simulation.py
```
See simulation.gif for the simulation result.  
The comparison between this simulation and the experimental data:  
Final tip displacement (simulation):
Left optical fiber tip: 1.50e-02 m  
Right optical fiber tip: 0.00e+00 m  
Elastin strain: 4066.66 % (Original length: 3.60e-04 m)  
Experimental data:  
Left optical fiber tip: 9.22e-04 m  
Right optical fiber tip: 6.86e-04 m  
Elastin strain: 3620.00 %  
elastin_curve.jpg is a experimental stress strain curve of a elastin fiber.  

### ME700 Applied Skills
**Technical Knowledge**  
* Implement finite element analysis techniques for solving complex mechanical problems  
* Analyze and solve problems involving material nonlinearity and geometric nonlinearity (large deformation in this project)

**Software Development & Implementation**  
* Use GitHub for code management  
* Python programming including implenting open sources code

**Integration & Application**
* Design and implement comprehensive solutions that integrate mechanical theory with practical software implementation
* Create modular, reusable code that follows software engineering best practices while solving mechanics problems
* Develop and execute verification and validation strategies for computational mechanics implementations  
