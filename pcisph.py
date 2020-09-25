import sys
import os
import taichi as ti
import time
import math
import numpy as np

from Canvas import Canvas
from HashGrid import HashGrid

ti.init(arch=ti.gpu,advanced_optimization=True)

#gui param
imgSize = 512
current_time = 0.0
total_time   = 5.0
eps = 1e-5
test_id = 1000
deltaT     = ti.field( dtype=ti.f32, shape=(1))


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
gravity    = ti.Vector([0.0, -9.81, 0.0])
vel         = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel       = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel_pre   = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
pos_star    = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
vel_star    = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
global hash_grid

#pressure param
rho_err     = ti.field( dtype=ti.f32, shape=(1))
rho         = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
adv_rho       = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
pressure    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))




#viscorcity cg sovler
dim_coff  = 10.0
viscosity     = 0.1
viscosity_b   = 0.0
pr_iter = 0



def CpuGradW(r):
    res = np.array([0.0, 0.0, 0.0])
    rl =np.linalg.norm(r)
    q = rl / searchR
    if ((rl > 1.0e-5) and (q <= 1.0)):
    	gradq = r / ( rl*searchR)
    	if (q <= 0.5):
    		res = m_l*q*(3.0*q - 2.0)*gradq
    	else:
    		factor = 1.0 - q
    		res = -m_l*(factor*factor)*gradq
    return res

def GetPciCoff():
    supportRadius = searchR
    diam = 2.0 * particleRadius
    sumGradW         = np.array([0.0, 0.0, 0.0])
    sumGradW2     = 0.0
    V00 = particleRadius * particleRadius * particleRadius * 0.8 * 8.0

    xi = np.array([0.0, 0.0, 0.0])
    xj = np.array([-supportRadius, -supportRadius, -supportRadius])
    while (xj[0] <= supportRadius):
        while (xj[1] <= supportRadius):
            while (xj[2] <= supportRadius):
                r = xi-xj
                dist = np.linalg.norm(r)
                if(dist < supportRadius):
                    grad = CpuGradW(r)
                    sumGradW += grad
                    dist_grad = np.linalg.norm(grad)
                    sumGradW2 += dist_grad*dist_grad
                xj[2] += diam
            xj[1] += diam
            xj[2] = -supportRadius
        xj[0] += diam
        xj[1] = -supportRadius
        xj[2] = -supportRadius

    beta = 2.0 * V00*V00
    dist_sumgrad = np.linalg.norm(sumGradW)
    return 1.0 / (beta * (dist_sumgrad*dist_sumgrad  + sumGradW2))


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


def sovel_pressure():
    global pr_iter
    pr_iter = 0
    err  = 0.0

    init_iter_info()
    while (err >  0.01 or pr_iter < 3) and (pr_iter < 50):
        update_iter_info()
        predict_density()
        err = rho_err.to_numpy()[0] / float(particleLiquidNum)
        pr_iter += 1    

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
        deltaT[0] = 0.001
 
@ti.kernel
def compute_nonpressure_force():
    for i in d_vel:
        d_vel[i] = gravity
        rho[i]  = VL0 * W_norm(0.0) * rho_0 

        cur_neighbor     = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]

            if j < particleLiquidNum:
                rho[i]     += VL0 * W(r) * rho_0 
                d_vel[i]   += dim_coff * viscosity * liqiudMass / rho[j] * (vel[i] - vel[j]).dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
            else:
                rho[i]     += VS0 * W(r) * rho_0
                d_vel[i]   += dim_coff * viscosity_b * VS0 * (rho[i] / rho_0) * vel[i].dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
            
            k += 1

@ti.kernel
def init_iter_info():
    for i in vel_star:
        vel_star[i]  = vel[i]
        pos_star[i]  = hash_grid.pos[i]
        pressure[i]  = 0.0
        d_vel_pre[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def update_iter_info():
    for i in vel_star:
        vel_star[i]  = vel[i] + (d_vel[i]+d_vel_pre[i])  * deltaT[0]
        pos_star[i]  = hash_grid.pos[i] + vel_star[i]  * deltaT[0]

        rho_err[i]   = 0.0
        pressure[i]  = 0.0

@ti.kernel
def predict_density():
    for i in rho:
        adv_rho[i] = VL0 * W_norm(0.0)

        cur_neighbor     = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            WW = W(r)
            if j < particleLiquidNum:
                adv_rho[i]     += VL0 * WW
            else:
                adv_rho[i]     += VS0 * WW
            k += 1

        adv_rho[i] = ti.max(adv_rho[i], 1.0)
        pressure[i] += pci_coff * (adv_rho[i]-1.0) / (deltaT[0] * deltaT[0])
        rho_err[0]  += adv_rho[i]-1.0

    for i in d_vel_pre:
        d_vel_pre[i]      = ti.Vector([0.0, 0.0, 0.0])

        cur_neighbor = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]

            pi = hash_grid.pos[i]
            pj = hash_grid.pos[j]
            if j < particleLiquidNum:
                pj = pos_star[j]

            gradV       = gradW(pi - pj)
            dpi = pressure[i] 
            if j < particleLiquidNum:
                dpj = pressure[j] 
                d_vel_pre[i]   +=  - VL0 * (dpi + dpj) * gradV
            else:
                d_vel_pre[i]   +=  - VS0 * dpi  * gradV
            k += 1


@ti.kernel
def update_pos():
    for i in vel:
        vel[i]  += (d_vel[i]+d_vel_pre[i]) * deltaT[0]
        hash_grid.pos[i]  += vel[i]  * deltaT[0]

@ti.kernel
def draw_particle():
    for i in hash_grid.pos:
        if i < particleLiquidNum:
            #draw_solid_sphere(hash_grid.pos[i], ti.Vector([1.0,1.0,1.0]))
            sph_canvas.draw_sphere(hash_grid.pos[i], ti.Vector([1.0,1.0,1.0]))
        elif hash_grid.pos[i][2] < boundary*0.1 and hash_grid.pos[i][2] > -boundary*0.1 :
            sph_canvas.draw_sphere(hash_grid.pos[i], ti.Vector([0.3,0.3,0.3]))




gui = ti.GUI('pcisph', res=(imgSize, imgSize))
sph_canvas = Canvas(imgSize, imgSize)
init_particle()

pci_coff = GetPciCoff()
reset_particle()

sph_canvas.set_fov(2.0)
sph_canvas.set_target(0.0, 0.0, 0.0)
sph_canvas.ortho = 1


while gui.running:
    sph_canvas.set_view_point(0.0, 0.0, 0.0, 3.0)
    hash_grid.update_grid()

    compute_nonpressure_force()
    sovel_pressure()
    update_pos()

    sph_canvas.clear_canvas()
    draw_particle()
    #ti.imwrite(img, str(frame//iterNum)+ ".png")

    gui.set_image(sph_canvas.img.to_numpy())
    gui.show()

    
    dt = deltaT.to_numpy()[0]
    current_time += dt

    print("time:%.3f"%current_time, "step:%.4f"%dt,  "pressure:", pr_iter)
    if math.isnan(hash_grid.pos.to_numpy()[test_id, 0]) or current_time >= total_time:
        print(adv_rho.to_numpy()[test_id], hash_grid.pos.to_numpy()[test_id], d_vel.to_numpy()[test_id])
        sys.exit()



