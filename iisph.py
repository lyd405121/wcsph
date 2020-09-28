import sys
import os
import taichi as ti
import time
import math
import numpy as np
from Canvas import Canvas
from HashGrid import HashGrid

#ti.init(arch=ti.gpu,advanced_optimization=False)
ti.init(arch=ti.gpu,advanced_optimization=True)

#gui param
imgSizeX = 512
imgSizeY = 512
current_time = 0.0
total_time   = 5.0
eps = 1e-5
test_id = 0
eps = 1e-5


#particle param
particleRadius = 0.025
gridR       = particleRadius * 2.0
invGridR    = 1.0 / gridR
particleDimX = 20
particleDimY = 20
particleDimZ = 20
particleLiquidNum  = particleDimX*particleDimY*particleDimZ

rho_L0 = 1000.0
rho_S0 = rho_L0
VL0    = particleRadius * particleRadius * particleRadius * 0.8 * 8.0
VS0    = VL0 
liqiudMass = VL0 * rho_L0
boundary    = 2.0

#kernel param
searchR     = gridR*2.0
pi    = 3.1415926
h3    = searchR*searchR*searchR
m_k   = 8.0  / (pi*h3)
m_l   = 48.0 / (pi*h3)


#advetion param
gravity    = ti.Vector([0.0, -9.81, 0.0])
vel_guess   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
vel         = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
vel_max      = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
d_vel       = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))


#pressure param
a_ii        = ti.field(dtype=ti.f32, shape=(particleLiquidNum))
d_ii        = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
dij_pj      = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
pressure_pre    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
pressure    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
rho         = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
adv_rho       = ti.field( dtype=ti.f32, shape=(particleLiquidNum))


#CFL time step
vs_iter = 0
dv_iter = 0
pr_iter = 0

user_max_t     = 0.005
user_min_t     = 0.00005
deltaT     = ti.field( dtype=ti.f32, shape=(1))


#viscorcity cg sovler
dim_coff  = 10.0
omega = 0.5
viscosity     = 2.0
viscosity_b   = 3.0
viscosity_err = 0.05

avg_density_err = ti.field( dtype=ti.f32, shape=(1))
cg_delta     = ti.field( dtype=ti.f32, shape=(1))
cg_delta_old     = ti.field( dtype=ti.f32, shape=(1))
cg_delta_zero     = ti.field( dtype=ti.f32, shape=(1))

cg_Minv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(particleLiquidNum))
cg_r = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
cg_dir   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
cg_Ad   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
cg_s   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))



global hash_grid


def init_particle(filename):
    global hash_grid
    hash_grid = HashGrid(gridR)
    
    ZxY = particleDimZ*particleDimY
    dis = particleRadius * 2.0
    for i in range(particleLiquidNum):
        hash_grid.add_liquid_point([float(i//ZxY)* dis - particleRadius ,
                                    float((i%ZxY)//particleDimZ)* dis + 0.1, 
                                    float(i%particleDimZ)* dis - particleRadius])

    hash_grid.add_obj(filename)
    hash_grid.setup_grid()

def compute_nonpressure_force():
    init_viscosity_para()

    global vs_iter 
    vs_iter = 0

    while vs_iter < 100:
        compute_viscosity_force()
        vs_iter+=1

        if cg_delta[0] <= viscosity_err * cg_delta_zero[0] or cg_delta_zero[0] < eps:
            break
    combine_nonpressure()



def solve_pressure():
    global pr_iter 

    pr_iter = 0
    err  = 0.0
    while (err > 0.001 or pr_iter < 2) and (pr_iter < 100):
        update_iter_info()
        update_pressure_force()
        err = avg_density_err.to_numpy()[0] / float(particleLiquidNum)
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
        pressure[i] = 0.0
        deltaT[0] = 0.001

@ti.func
def get_viscosity_Ax(x: ti.template(), i):
    ret = ti.Vector([0.0,0.0,0.0])
    cur_neighbor     = hash_grid.neighborCount[i]
    k=0
    while k < cur_neighbor:
        j = hash_grid.neighbor[i, k]
        r = hash_grid.pos[i] - hash_grid.pos[j]     
        
        if j < particleLiquidNum:
            ret += dim_coff*viscosity *  liqiudMass / rho[j]  * (x[i] - x[j]).dot(r)  / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r) / rho[i] * deltaT[0]
        else:
            ret += dim_coff*viscosity_b  * rho_S0 / rho[i] * VS0  * x[i].dot(r)  / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r) / rho[i] * deltaT[0]    
        k+=1
    return x[i] - ret

@ti.kernel
def init_viscosity_para():
    for i in vel_guess:
        vel_guess[i] += vel[i] 

    for i in cg_Minv:
        m = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        cur_neighbor     = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]     
            grad_xij = gradW(r).outer_product(r)
            if j < particleLiquidNum:
                m += dim_coff * viscosity  * liqiudMass / rho[j]  / (r.norm_sqr() + 0.01*searchR*searchR) * grad_xij 
            else:
                m += dim_coff * viscosity_b  * rho_S0 / rho[i] * VS0  / (r.norm_sqr() + 0.01*searchR*searchR) * grad_xij   
            k+=1
        cg_Minv[i] = (ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) - m  * (deltaT[0]/rho[i]) ).inverse()
        #cg_Minv[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])



    cg_delta_zero[0] = 0.0
    for i in cg_r:
        cg_r[i]   = vel[i] - get_viscosity_Ax(vel_guess, i)
        cg_dir[i] = cg_Minv[i] @ cg_r[i]

        cg_delta_zero[0] += cg_r[i].dot(cg_dir[i])
    cg_delta[0] = cg_delta_zero[0]

@ti.kernel
def compute_viscosity_force():
    cg_dAd  = eps
    for i in cg_r:
        cg_Ad[i] = get_viscosity_Ax(cg_dir, i)
        cg_dAd += cg_dir[i].dot(cg_Ad[i]) 
    
    
    alpha = cg_delta[0] / cg_dAd
    cg_delta_old[0] = cg_delta[0]
    cg_delta[0] = 0.0
    for i in cg_r:
        vel_guess[i] += alpha * cg_dir[i]
        cg_r[i] = cg_r[i] - alpha * cg_Ad[i]
        cg_s[i] = cg_Minv[i] @ cg_r[i]
        cg_delta[0] += cg_r[i].dot(cg_s[i])

        
    beta = cg_delta[0] / cg_delta_old[0]

    for i in cg_r:
        cg_dir[i] = cg_s[i] + beta * cg_dir[i]

@ti.kernel
def compute_density():
    for i in rho:
        rho[i]  = VL0 * W_norm(0.0) * rho_L0 

        cur_neighbor     = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            if j < particleLiquidNum:
                rho[i]     += VL0 * W(r) * rho_L0 
            else:
                rho[i]     += VS0 * W(r) * rho_S0
            k += 1

@ti.kernel
def combine_nonpressure():
    for i in d_vel:
        d_vel[i]  = gravity + (vel_guess[i] - vel[i]) / deltaT[0]
        vel_guess[i]  = vel_guess[i]-vel[i]

@ti.kernel
def compute_advection():
    for i in d_ii:
        d_ii[i] = ti.Vector([0.0, 0.0, 0.0])
        vel[i] += deltaT[0] * d_vel[i]

        cur_neighbor = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j  = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV       = gradW(r)

            inv_den     = rho_L0 / rho[i] 
            d_ii[i]     +=  -VL0 * inv_den * inv_den * gradV
            k += 1

    for i in a_ii:
        a_ii[i] = 0.0
        density = rho[i] / rho_L0
        adv_rho[i] = density
        pressure_pre[i] = 0.5 * pressure[i]

        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j  = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV       = gradW(r)
            

            if j < particleLiquidNum:
                adv_rho[i]     +=  deltaT[0] *  VL0 * ((vel[i] - vel[j]).dot(gradV))
            else:
                adv_rho[i]     +=  deltaT[0] *  VS0 * (vel[i] .dot(gradV))


            d_ji = VL0 / (density*density) * gradV
            a_ii[i]     +=   VL0 * (d_ii[i] -  d_ji).dot(gradV)
            k += 1

@ti.kernel
def update_iter_info():
    for i in dij_pj:

        avg_density_err[0] = 0.0
        dij_pj[i] = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            if j < particleLiquidNum:
                r = hash_grid.pos[i] - hash_grid.pos[j]
                gradV=gradW(r)
                densityj = rho[j] / rho_L0
                dij_pj[i] += -VL0/(densityj*densityj)*pressure_pre[j]*gradV
            k += 1

@ti.kernel
def update_pressure_force():
    for i in pressure:
        sum=0.0
        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV       = gradW(r)

            if j < particleLiquidNum:
                density = rho[i] / rho_L0
                dji = VL0 / (density*density) * gradV
                d_ji_pi = dji *  pressure_pre[i]
                d_jk_pk = dij_pj[j] 
                sum +=  VL0 * ( dij_pj[i] - d_ii[j]*pressure_pre[j] - (d_jk_pk - d_ji_pi)).dot(gradV)
            else:
                sum +=  VS0 * dij_pj[i].dot(gradV)
            k += 1


        b = 1.0 - adv_rho[i]
        h2     = deltaT[0] * deltaT[0]

        denom = a_ii[i]*h2
        if (ti.abs(denom) > eps):
            pressure[i] = ti.max(    (1.0 - omega) *pressure_pre[i] + omega / denom * (b - h2*sum), 0.0)
            #print( adv_rho[i],rho[i] / rho_L0, h2*sum, omega / denom)
        else:
            pressure[i] = 0.0
        
        if pressure[i] != 0.0:
            avg_density_err[0] += (a_ii[i]*pressure[i]  + sum)*h2 - b
        
@ti.kernel
def update_pos():

    for i in d_vel:
        d_vel[i]   = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor = hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV       = gradW(r)

            density_i = rho[i]/ rho_L0
            dpi = pressure[i] / (density_i*density_i)
            if j < particleLiquidNum:
                density_j = rho[j]/ rho_L0
                dpj = pressure[j] / (density_j*density_j)
                d_vel[i]   +=  - VL0 * (dpi + dpj) * gradV
            else:
                d_vel[i]   +=  - VS0 * dpi  * gradV
            k += 1

    for i in vel:
        vel[i]  += d_vel[i] * deltaT[0]
        hash_grid.pos[i]  += vel[i]  * deltaT[0]



@ti.kernel
def draw_particle():
    for i in hash_grid.pos:
        if i < particleLiquidNum:
            sph_canvas.draw_sphere(hash_grid.pos[i], ti.Vector([1.0,1.0,1.0]))
        else:
            sph_canvas.draw_point(hash_grid.pos[i], ti.Vector([0.3,0.3,0.3]))


gui = ti.GUI('iisph', res=(imgSizeX, imgSizeY))
sph_canvas = Canvas(imgSizeX, imgSizeY)
init_particle("box_boundry.obj")
reset_particle()


while gui.running:

    sph_canvas.yaw_cam(0.0,1.0,0.0)
    #sph_canvas.static_cam(0.0,1.0,0.0)
    hash_grid.update_grid()

    compute_density()
    compute_nonpressure_force()
    compute_advection()
    

    solve_pressure()
    update_pos()

    sph_canvas.clear_canvas()
    draw_particle()
    gui.set_image(sph_canvas.img.to_numpy())
    gui.show()

    dt = deltaT.to_numpy()[0]
    current_time += dt
    print("time:%.3f"%current_time, "step:%.4f"%dt, "viscorcity:", vs_iter, "pressure:", pr_iter)
    #print(rho.to_numpy()[test_id], d_ii.to_numpy()[test_id], a_ii.to_numpy()[test_id], dij_pj.to_numpy()[test_id])

    if math.isnan(hash_grid.pos.to_numpy()[test_id, 0]) or current_time >= total_time:
        print(adv_rho.to_numpy()[test_id], hash_grid.pos.to_numpy()[test_id], d_vel.to_numpy()[test_id])
        sys.exit()




