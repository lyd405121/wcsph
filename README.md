# Weakly compressed liqiud simulation

### The code is mainly code transplanted from [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)

### [Taichi](https://github.com/taichi-dev/taichi) programming language

---


## Current implement algorithm
* [SESPH] 
* [PCISPH]
* [IISPH]
* [DFSPH]
---

## Some image produced by this project

### Possion disk sample(full parallel) for boundry handling

![image](boundtry.gif)
---

###  Implicit viscosity solver 

![image](super_nian.gif)

![image](rock.gif)

###  Use precondition cg to solve viscosity:

![image](average_iter_num.png)


###  Surface tension (without & with):

![image](no_tension.gif) ![image](tension.gif)


###  anistropic mesh restruction :

- marching cube 

![image](mc.png)

- using anistropic kernel to build the volume field

![image](ani-mc.png)


###  Other algorithm:

- hash grid

- cfl time step
---
# Referrence

- M Weiler 2018: A physically consistent implicit viscosity solver for SPH fluids

- Nadir Akinci 2013: Versatile surface tension and adhesion for SPH fluids

- JIHUN YU 2013: Reconstructing Surfaces of Particle-Based Fluids. Using Anisotropic Kernels. 

