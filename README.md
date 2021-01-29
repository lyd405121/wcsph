# Weakly compressed liqiud simulation

- The code is mainly code transplanted from [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)

- The code use [Taichi](https://github.com/taichi-dev/taichi) programming language

---


## Current implement algorithm
* [SESPH] 
* [PCISPH]
* [IISPH]
* [DFSPH]
---

## How to run 
* First config your anaconda workspace, and open the anaconda prompt
* Second you need install [taichi](https://github.com/taichi-dev/taichi) language, **pip install taichi**
* Last you type **ti dfsph.py**, that's all
---

## Some image produced by this project

### Possion disk sample(full parallel) for boundry handling

![image](image/boundtry.gif)
---

###  Implicit viscosity solver 

<img src="image/super_nian.gif" width="25%" height="25%" />

<img src="image/rock.gif" width="25%" height="25%" />

---

###  Use precondition cg to solve viscosity:

<img src="image/average_iter_num.png" width="25%" height="25%" />

---

###  Surface tension (without & with):

![image](image/no_tension.gif) ![image](image/tension.gif)

---

###  anistropic mesh restruction :

- marching cube 

<img src="image/mc.png" width="25%" height="25%" />

- using anistropic kernel to build the volume field

<img src="image/ani-mc.png" width="25%" height="25%" />

- render image

<img src="out/rendering.png" width="25%" height="25%" />
---

###  Other algorithm:

- hash grid

- cfl time step
---
# Referrence

- M Weiler 2018: A physically consistent implicit viscosity solver for SPH fluids

- Nadir Akinci 2013: Versatile surface tension and adhesion for SPH fluids

- JIHUN YU 2013: Reconstructing Surfaces of Particle-Based Fluids. Using Anisotropic Kernels. 

