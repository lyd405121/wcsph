import sys
import os
import time
import math
import numpy as np
import taichi as ti
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


#particle param
particleRadius = 0.025
gridR       = particleRadius * 2.0
invGridR    = 1.0 / gridR
particleDimX = 20
particleDimY = 10
particleDimZ = 10
particleLiquidNum  = particleDimX*particleDimY*particleDimZ

rho_L0 = 1000.0
rho_S0 = rho_L0
VL0    = particleRadius * particleRadius * particleRadius * 0.8 * 8.0
VS0    = VL0 
liqiudMass = VL0 * rho_L0


#kernel param
searchR     = gridR*2.0
pi    = 3.1415926
h3    = searchR*searchR*searchR
m_k   = 8.0  / (pi*h3)
m_l   = 48.0 / (pi*h3)


#advetion param
gravity    = ti.Vector([0.0, -9.81, 0.0])
vel_guess  = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
vel        = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
omega      = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
vel_max    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
d_vel      = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
d_omega    = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))

#pressure param
alpha_coff  = ti.field(dtype=ti.f32, shape=(particleLiquidNum))
kappa        = ti.field(dtype=ti.f32, shape=(particleLiquidNum))
kappa_v        = ti.field(dtype=ti.f32, shape=(particleLiquidNum))

pressure    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
rho         = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
adv_rho       = ti.field( dtype=ti.f32, shape=(particleLiquidNum))


#CFL time step
vs_iter = 0
dv_iter = 0
pr_iter = 0

user_max_t     = 0.005
user_min_t     = 0.0001
deltaT     = ti.field( dtype=ti.f32, shape=(1))


#viscorcity cg sovler
dim_coff          = 10.0
viscosity         = 5.0
viscosity_b       = 3.0
viscosity_err     = 0.05

avg_density_err = ti.field( dtype=ti.f32, shape=(1))
cg_delta     = ti.field( dtype=ti.f32, shape=(1))
cg_delta_old     = ti.field( dtype=ti.f32, shape=(1))
cg_delta_zero     = ti.field( dtype=ti.f32, shape=(1))

cg_Minv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(particleLiquidNum))
cg_r = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
cg_dir   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
cg_Ad   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))
cg_s   = ti.Vector.field(3, dtype=ti.f32, shape=(particleLiquidNum))

#tension
tension_coff     = 0.02
tension_coff_b   = 0.1 * tension_coff 
tension_c      = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
tension_gradc  = ti.field( dtype=ti.f32, shape=(particleLiquidNum))

#vorcity_coff
viscosity_omega   = 0.1
vorticity_coff  = 0.01
vorticity_init  = 0.5

global sph_canvas
global hash_grid

def init_particle(filename):
    global hash_grid
    hash_grid = HashGrid(gridR)
    
    ZxY = particleDimZ*particleDimY
    dis = particleRadius * 2.0
    for i in range(particleLiquidNum):
        hash_grid.add_liquid_point([float(i//ZxY - particleDimX /2)* dis ,
                                    float((i%ZxY)//particleDimZ)* dis + 0.7, 
                                    float(i%particleDimZ-particleDimZ /2)* dis])
    '''
    for i in range(particleLiquidNum):
        hash_grid.add_liquid_point([float(i//ZxY)* dis - particleRadius ,
                                    float((i%ZxY)//particleDimZ)* dis + 0.1, 
                                    float(i%particleDimZ)* dis - particleRadius])
    '''

    hash_grid.add_obj(filename)
    hash_grid.setup_grid()
    
def compute_nonpressure_force():

    global vs_iter
    clear_nonpressure()

    #surface tension
    compute_tension()

    #pre cg for viscorcity
    init_viscosity_para()
    vs_iter = 0
    while vs_iter < 100:
        compute_viscosity_force()
        vs_iter+=1
        if cg_delta[0] <= viscosity_err * cg_delta_zero[0] or cg_delta_zero[0] < eps:
            break
    end_viscosity()

    #compute vorticity
    compute_vorticity()



def optimize_time_step():
    size = 1
    while size < particleLiquidNum:
        cfl_time_step(size)
        size = size*2

    deltaT_np = deltaT.to_numpy()
    vel_max_np = vel_max.to_numpy()
  
    if vel_max_np[0] >eps:
        cfl_factor = 0.5
        time_step =  cfl_factor * 0.4 * particleRadius * 2.0 / math.sqrt(vel_max_np[0])
        time_step = min(time_step, user_max_t)
        time_step = max(time_step, user_min_t)
 
        iter = max(vs_iter, max(pr_iter, vs_iter))
        if iter > 10:
            deltaT_np[0] *= 0.9
        elif iter <5:
            deltaT_np[0] *= 1.1

        deltaT_np[0] = min(deltaT_np[0], time_step)
        deltaT.from_numpy(deltaT_np)

def solve_vel_divergence():
    global dv_iter 
    dv_iter = 0

    warmstart_divergence_vel()
    err  = -0.1
    begin_divergence_iter()

    deltaT_np = deltaT.to_numpy()

    while (avg_density_err.to_numpy()[0] > err) and (dv_iter < 10):
        divergence_iter()        
        err = 0.001 * float(particleLiquidNum) / deltaT_np[0]
        dv_iter += 1

    end_divergence_iter()



def solve_pressure():
    global pr_iter 


    warmstart_pressure() 
    pr_iter = 0
    err  = 0.0
    begin_pressure_iter()
    

    while (err > 0.001 or pr_iter < 2 ) and (pr_iter < 100):
        pressure_iter()
        err = avg_density_err.to_numpy()[0] / float(particleLiquidNum)
        pr_iter += 1
    end_pressure_iter()
           
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
def reset_param():

    for i in vel:
        vel[i]      = ti.Vector([0.0, 0.0, 0.0])
        omega[i] = ti.Vector([0.0, 0.0, 0.0])
        
        pressure[i] = 0.0
        kappa_v[i]  = 0.0
        kappa[i] = 0.0

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
            ret += dim_coff*viscosity * liqiudMass / rho[j]  * (x[i] - x[j]).dot(r)  / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r) / rho[i] * deltaT[0]
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
                m += dim_coff * viscosity * liqiudMass    / rho[j]  / (r.norm_sqr() + 0.01*searchR*searchR) * grad_xij 
            else:
                m += dim_coff * viscosity_b  * rho_S0 / rho[i] * VS0  / (r.norm_sqr() + 0.01*searchR*searchR) * grad_xij   
            k+=1
        cg_Minv[i] = (ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) - m  * (deltaT[0]/rho[i]) ).inverse()

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

        cur_neighbor     =  hash_grid.neighborCount[i]
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
def compute_tension():
    for i in tension_c:
        tension_c[i]  = liqiudMass / rho[i] *  W_norm(0.0)

 
        cur_neighbor     =  hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            Wr = W(r)

            if j < particleLiquidNum:
                tension_c[i]     += liqiudMass / rho[j] * Wr  
            else:
                tension_c[i]     += VS0 * Wr 
            k += 1


    for i in tension_gradc:
        gradc = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor     =  hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]

            if j < particleLiquidNum:
                gradc     +=  liqiudMass / rho[j] * tension_c[j] * gradW(r) 
            k += 1

        gradc *= 1.0 / tension_c[i]
        tension_gradc[i]  = gradc.norm_sqr()

    for i in d_vel:
        factor = 0.25 * tension_coff / rho[i]
        factorb = 0.25 * tension_coff_b / rho[i]

        cur_neighbor     =  hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gardV = gradW(r) 

            if j < particleLiquidNum:
                d_vel[i]     +=  factor * liqiudMass / rho[j] * (tension_gradc[i] + tension_gradc[j]) * gardV 
            else:
                d_vel[i]     +=  factorb * VS0 * tension_gradc[i]  * gardV 
            k += 1



@ti.kernel
def compute_vorticity():
    for i in omega:
        d_omega[i]  = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor     =  hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV = gradW(r)

            if j < particleLiquidNum:
                d_omega[i] += -1.0 / deltaT[0] * vorticity_init* viscosity_omega *(liqiudMass/rho[j])*  (omega[i] - omega[j]) * W(r)
                d_vel[i] += vorticity_coff / rho[i] * liqiudMass * (omega[i] - omega[j]).cross(gradV)
                d_omega[i] += vorticity_coff / rho[i]* vorticity_init * liqiudMass * (vel[i]-vel[j]).cross(gradV)
            else:
                d_vel[i] += vorticity_coff / rho[i] * rho_L0 * VS0 * (omega[i] - omega[j]).cross(gradV)
                d_omega[i] += vorticity_coff / rho[i]* vorticity_init * rho_L0* VL0* (vel[i]-vel[j]).cross(gradV)
            d_omega[i] += -2.0 * vorticity_init * vorticity_coff * omega[i]
            k += 1

    for i in omega:
        omega[i] += d_omega[i] * deltaT[0]


@ti.kernel
def clear_nonpressure():
    for i in d_vel:
        d_vel[i]  = gravity


@ti.kernel
def end_viscosity():
    for i in vel:
        d_vel[i]  += (vel_guess[i] - vel[i]) / deltaT[0]
        vel_guess[i]    = vel_guess[i] - vel[i]

@ti.kernel
def compute_dfsph_coff():
    for i in alpha_coff:
        alpha_coff[i] = 0.0
        sum_grad = ti.Vector([0.0, 0.0, 0.0])
        sum_grad_square = 0.0

        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j  = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV       = gradW(r)
            

            if j < particleLiquidNum:
                temp = VL0 * gradV
                sum_grad_square += temp.norm_sqr()
                sum_grad += temp
            else:
                sum_grad +=  VS0 * gradV
            k += 1

        sum_grad_square += sum_grad.norm_sqr()

        if sum_grad_square > eps:
            alpha_coff[i] = -1.0 / sum_grad_square
        else:
            alpha_coff[i] = 0.0

@ti.func
def update_drho_divergence(i):
    cur_neighbor = hash_grid.neighborCount[i]
    k=0
    adv_rho[i] = 0.0
    while k < cur_neighbor:
        j = hash_grid.neighbor[i, k]
        r = hash_grid.pos[i] - hash_grid.pos[j]
        gradV = gradW(r)

        if j < particleLiquidNum:
            adv_rho[i] += VL0 * ( (vel[i]-vel[j]).dot(gradV))
        else:
            adv_rho[i] += VS0 * ( vel[i].dot(gradV) )
        k += 1
    adv_rho[i] = max(adv_rho[i], 0.0)

    if cur_neighbor < 20:
        adv_rho[i]  = 0.0

@ti.func
def update_drho_pressure(i):
    cur_neighbor = hash_grid.neighborCount[i]
    k=0
    temp = 0.0

    while k < cur_neighbor:
        j = hash_grid.neighbor[i, k]
        r = hash_grid.pos[i] - hash_grid.pos[j]
        gradV = gradW(r)

        if j < particleLiquidNum:
            temp += VL0 * ((vel[i]-vel[j]).dot(gradV))
        else:
            temp += VL0 * (vel[i].dot(gradV))
        k += 1

    adv_rho[i] = rho[i] / rho_L0 + deltaT[0] * temp
    adv_rho[i] = ti.max(1.0, adv_rho[i])


@ti.kernel
def warmstart_divergence_vel():
    for i in kappa_v:
        kappa_v[i] = 0.5 * max(kappa_v[i] / deltaT[0] , -0.5*rho_L0*rho_L0)
        update_drho_divergence(i)

    for i in vel:
        if adv_rho[i] > 0.0:
            cur_neighbor = hash_grid.neighborCount[i]
            k=0

            while k < cur_neighbor:
                j = hash_grid.neighbor[i, k]
                r = hash_grid.pos[i] - hash_grid.pos[j]
                gradV=gradW(r)

                ki = kappa_v[i]
                if j < particleLiquidNum:
                    kj = kappa_v[j]
                    sum = ki + kj
                    if abs(sum) > eps:
                        vel[i] += deltaT[0] * sum * VL0*gradV
                elif abs(ki) > eps:
                    vel[i] += deltaT[0] * ki * VS0 * gradV
                k += 1

@ti.kernel
def  begin_divergence_iter():
    for i in kappa_v:
        update_drho_divergence(i)
        alpha_coff[i] =  alpha_coff[i] /  deltaT[0] 
        kappa_v[i] = 0.0
        

@ti.kernel
def  divergence_iter():
    for i in vel:
        avg_density_err[0] = 0.0
        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        bi = adv_rho[i]
        ki = bi * alpha_coff[i]
        kappa_v[i] += ki

        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] - hash_grid.pos[j]
            gradV=gradW(r)

            if j < particleLiquidNum:
                sum = ki + alpha_coff[j] *adv_rho[j]
                if abs(sum) > eps:
                    vel[i] += deltaT[0] * sum * VL0*gradV
            else:
                if ti.abs(ki) > eps:
                    vel[i] += deltaT[0] * ki * VS0 * gradV

            k += 1 
    
    for i in adv_rho:
        update_drho_divergence(i)
        avg_density_err[0] += adv_rho[i]


@ti.kernel
def  end_divergence_iter():
    for i in kappa_v:
        kappa_v[i] *= deltaT[0]
        alpha_coff[i] *= deltaT[0]


@ti.kernel
def warmstart_pressure():
    for i in kappa:
        kappa[i] = max(kappa[i] / deltaT[0] / deltaT[0] , -0.5*rho_L0*rho_L0)

    for i in adv_rho:
        if adv_rho[i] > rho_L0:
            cur_neighbor = hash_grid.neighborCount[i]
            k=0

            while k < cur_neighbor:
                j = hash_grid.neighbor[i, k]
                r = hash_grid.pos[i] - hash_grid.pos[j]
                gradV=gradW(r)

                if j < particleLiquidNum:
                    sum = kappa[i] + kappa[j]
                    if abs(sum) > eps:
                        vel[i] += deltaT[0] * sum * VL0 * gradV 
                elif abs(kappa[i]) > eps:
                    vel[i] += deltaT[0] * kappa_v[i] * VS0 * gradV
                k += 1


@ti.kernel
def begin_pressure_iter():
    for i in kappa:
        update_drho_pressure(i)
        alpha_coff[i] = alpha_coff[i] / deltaT[0] / deltaT[0]
        kappa[i] = 0.0

@ti.kernel
def  pressure_iter():
    for i in vel:
        avg_density_err[0] = 0.0
        cur_neighbor = hash_grid.neighborCount[i]
        k=0

        bi = adv_rho[i] - 1.0
        ki = bi * alpha_coff[i]
        kappa[i] += ki

        while k < cur_neighbor:
            j = hash_grid.neighbor[i, k]
            r = hash_grid.pos[i] -hash_grid.pos[j]
            gradV=gradW(r)

            if j < particleLiquidNum:
                bj = adv_rho[j] - 1.0
                kj = bj * alpha_coff[j]
                sum = ki + kj
                if ti.abs(sum) > eps:
                    vel[i] += deltaT[0] * sum * VL0 * gradV
            elif ti.abs(ki) > eps:
                vel[i] += deltaT[0] * ki * VS0 * gradV

            k += 1 
    
    for i in vel:
        update_drho_pressure(i)
        avg_density_err[0] += adv_rho[i]-1.0

@ti.kernel
def  end_pressure_iter():
    for i in kappa:
        kappa[i] *= deltaT[0] * deltaT[0]


@ti.kernel
def cfl_time_step(index: ti.i32):
    for i in vel_max:
        if index == 1:
            vel_max[i] = ti.max( (vel[i] + d_vel[i] * deltaT[0]).norm_sqr(), 0.1)
        elif i % index == 0:
            offcet = int(index / 2)
            vi = vel_max[i]
            vj = vel_max[i+offcet]

            if vi > vj:
                vel_max[i] =   vi
            else:
                vel_max[i] =   vj

 

@ti.kernel
def  update_vel():
    for i in vel:
        vel[i]  += d_vel[i]  * deltaT[0]

@ti.kernel
def  update_pos():
    for i in vel:
        hash_grid.pos[i]  += vel[i]  * deltaT[0]


@ti.kernel
def draw_particle():
    
    for i in hash_grid.pos:
        if i < particleLiquidNum:
            sph_canvas.draw_sphere(hash_grid.pos[i], ti.Vector([1.0,1.0,1.0]))
    
    for i in hash_grid.pos:
        posi = hash_grid.pos[i]
        if i> particleLiquidNum and posi.z < 0.3 and posi.z > -0.3 and posi.x < 0.99 and posi.x > -0.99 and posi.y < 1.99 and posi.y > 0.01:
            sph_canvas.draw_sphere(posi, ti.Vector([1.0,0.0,0.0]))
        else:
            sph_canvas.draw_point(posi, ti.Vector([0.3,0.3,0.3]))


gui = ti.GUI('dfsph', res=(imgSizeX, imgSizeY))
sph_canvas = Canvas(imgSizeX, imgSizeY)
init_particle("boundry.obj")
reset_param()

while gui.running:

    sph_canvas.pitch_cam(0.0, 1.0, 0.0)
    #sph_canvas.yaw_cam(0.0,1.0,0.0)
    #sph_canvas.static_cam(0.0,1.0,0.0)

    hash_grid.update_grid()

    compute_density()
    compute_dfsph_coff()
    solve_vel_divergence()

    compute_nonpressure_force()
    optimize_time_step()
    update_vel()

    solve_pressure()
    update_pos()

    sph_canvas.clear_canvas()
    draw_particle()
    gui.set_image(sph_canvas.img.to_numpy())
    gui.show()

    dt = deltaT.to_numpy()[0]
    current_time += dt

    print("time:%.3f"%current_time, "step:%.4f"%dt, "viscorcity:", vs_iter, "divergence:", dv_iter, "pressure:", pr_iter)
    #sph_canvas.export_png(current_time)

    if math.isnan(hash_grid.pos.to_numpy()[test_id, 0]) or current_time >= total_time:
        print(adv_rho.to_numpy()[test_id], hash_grid.pos.to_numpy()[test_id], d_vel.to_numpy()[test_id])
        sys.exit()