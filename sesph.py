import sys
import os
import taichi as ti
import time
import math
import numpy as np
from Canvas import Canvas
from HashGrid import HashGrid


#gui param
imgSize = 512
current_time = 0.0
total_time   = 5.0
eps = 1e-5
test_id = 1000
deltaT    = ti.field( dtype=ti.f32, shape=(1))


#particle param
particleRadius = 0.025
gridR       = particleRadius * 2.0
searchR     = gridR*2.0
invGridR    = 1.0 / gridR
boundary    = 2.0
blockSize   = int(boundary * invGridR)
doublesize  = blockSize*blockSize
gridSize    = blockSize*blockSize*blockSize

particleDimX = 20
particleDimY = 20
particleDimZ = 20
particleLiquidNum  = particleDimX*particleDimY*particleDimZ
particleSolidNum   = doublesize * 2 + (blockSize-2)*blockSize*2 + (blockSize-2)*(blockSize-2)*2 
particleNum        = particleLiquidNum + particleSolidNum

rho_0 = 1000.0
VL0    = particleRadius * particleRadius * particleRadius * 0.8 * 8.0
VS0    = VL0 * 2.0
liqiudMass = VL0 * rho_0

#kernel param
searchR     = gridR*2.0
pi    = 3.1415926
h3    = searchR*searchR*searchR
m_k   = 8.0  / (pi*h3)
m_l   = 48.0 / (pi*h3)

#advetion param
vel         = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel       = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
gravity    = ti.Vector([0.0, -9.81, 0.0])
global hash_grid

#pressure param
rho         = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
pressure    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
stiffness   = 50000.0

#viscorcity 
dim_coff  = 10.0
viscosity     = 0.1
viscosity_b   = 0.0



def init_particle():
    global hash_grid
    global particleLiquidNum
    global particleNum
    global particleSolidNum


    maxboundarynp = np.ones(shape=(1,3), dtype=np.float32)
    minboundarynp = np.ones(shape=(1,3), dtype=np.float32)
    for j in range(3):
        maxboundarynp[0, j] = boundary*0.5-gridR*0.5
        minboundarynp[0, j] = -boundary*0.5+gridR*0.5


    arrV = np.ones(shape=(particleNum, 3), dtype=np.float32)


    for i in range(particleLiquidNum):
        aa = particleDimZ*particleDimY
        arrV[i, 0]  = float(i//aa) * particleRadius*2.0
        arrV[i, 1]  = float(i%aa//particleDimZ) * particleRadius*2.0 - 0.9
        arrV[i, 2]  = float(i%particleDimZ) * particleRadius*2.0

    #y = Ax + B  
    A = boundary  / ( float(blockSize)-1.0 )
    B = -0.5 * boundary
    shrink = 1.0
    index = 0
    for i in range(gridSize):
        indexX      = i//doublesize
        indexY      = (i%doublesize)//blockSize
        indexZ      = i%blockSize
        if indexX== 0 or indexY ==0 or indexZ == 0 or\
        indexX == blockSize-1 or indexY ==blockSize-1 or indexZ == blockSize-1 :
            arrV[index+particleLiquidNum, 0] = (A * float(indexX)  + B) * shrink
            arrV[index+particleLiquidNum, 1] = (A * float(indexY)  + B) * shrink
            arrV[index+particleLiquidNum, 2] = (A * float(indexZ)  + B) * shrink
            index += 1

    hash_grid = HashGrid(particleNum, particleLiquidNum, maxboundarynp, minboundarynp, gridR)
    hash_grid.pos.from_numpy(arrV)
    print("grid szie:", hash_grid.gridSize, "liqiud particle num:", particleLiquidNum, "solid particle num:", particleSolidNum)



@ti.func
def gradW(r):
    res = ti.Vector([0.0, 0.0, 0.0])
    rl = r.norm()
    q = rl / searchR
    if ((rl > 1.0e-5) and (q <= 1.0)):
    	gradq = r / ( rl*searchR)
    	if (q <= 0.5):
    		res = m_l*q*(3.0*q - 2.0)*gradq
    	else:
    		factor = 1.0 - q
    		res = -m_l*(factor*factor)*gradq
    return res


@ti.func
def W_norm(v):
    res = 0.0
    q = v / searchR
    
    if q <= 1.0:
        if (q <= 0.5):
            qq = q*q
            qqq = qq*q
            res = m_k*(6.0*qqq - 6.0*qq+1.0)
        else:
            factor = 1.0 - q
            res = m_k*2.0*factor*factor*factor
    return res

@ti.func
def W(v):
    return W_norm(v.norm())

@ti.kernel
def reset_particle():
    for i in vel:
        vel[i]      = ti.Vector([0.0, 0.0, 0.0])
        pressure[i] = 0.0
        deltaT[0] = 0.001


@ti.kernel
def update_advection_density():
    for i in rho:
        rho[i]           = VL0 *W_norm(0.0)
        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            d_den = W(r)
            if j < particleLiquidNum:
                rho[i] += VL0 * d_den  
            else:
                rho[i] += VS0 * d_den
            k += 1
        rho[i] *= rho_0
        


@ti.kernel
def update_pressure():
    for i in rho:
        rho[i] = ti.max(rho[i], rho_0)

        q = rho[i] / rho_0
        qq = q*q
        qqqq = qq*qq
        pressure[i] = stiffness * (qqqq*qq*q - 1.0)

@ti.kernel
def compute_force():
    for i in d_vel:
        d_vel[i]         = gravity
        cur_neighbor = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            pi = hash_grid.pos[i]
            pj = hash_grid.pos[j]
            r = pi - pj
            grad  = gradW(r)

            if j < particleLiquidNum:
                d_vel[i] +=  dim_coff * viscosity * liqiudMass / rho[j] * (vel[i] - vel[j]).dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
                d_vel[i] +=  -rho_0 * VL0 * (pressure[i] / (rho[i]*rho[i]) + pressure[j] / (rho[j]*rho[j])) * grad
            else:
                d_vel[i] += dim_coff * viscosity_b * VS0 * (rho[i] / rho_0) * vel[i].dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
                d_vel[i] +=  -rho_0 * VS0 * (pressure[i] / (rho[i]*rho[i])+ pressure[i] / (rho_0*rho_0))  * grad

            k += 1

    
@ti.kernel
def integrator_sesph():
    for i in hash_grid.pos:
        if i < particleLiquidNum:
            vel[i]  += d_vel[i]  * deltaT[0]
            hash_grid.pos[i]  += vel[i]  * deltaT[0]

    
            
@ti.kernel
def draw_particle():
    for i in hash_grid.pos:
        if i < particleLiquidNum:
            #draw_solid_sphere(hash_grid.pos[i], ti.Vector([1.0,1.0,1.0]))
            sph_canvas.draw_sphere(hash_grid.pos[i], ti.Vector([1.0,1.0,1.0]))
        elif hash_grid.pos[i][2] < boundary*0.1 and hash_grid.pos[i][2] > -boundary*0.1 :
            sph_canvas.draw_sphere(hash_grid.pos[i], ti.Vector([0.3,0.3,0.3]))


            

gui = ti.GUI('sesph', res=(imgSize, imgSize))
sph_canvas = Canvas(imgSize, imgSize)
init_particle()
reset_particle()

sph_canvas.set_fov(2.0)
sph_canvas.set_target(0.0, 0.0, 0.0)
sph_canvas.ortho = 1

while gui.running:
    
    sph_canvas.set_view_point(0.0, 0.0, 0.0, 3.0)
    hash_grid.update_grid()

    update_advection_density()
    update_pressure()
    compute_force()
    integrator_sesph()

    
    sph_canvas.clear_canvas()
    draw_particle()
        
    gui.set_image(sph_canvas.img.to_numpy())
    gui.show()

    dt = deltaT.to_numpy()[0]
    current_time += dt
    print("time:%.3f"%current_time, "step:%.4f"%dt)



