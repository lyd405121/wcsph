import sys
import os
import time
import math
import numpy as np
import taichi as ti
from Canvas import Canvas
from ParticleData import ParticleData

from kernels.CubicKernel import CubicKernel
from kernels.CohesionKernel import CohesionKernel
from kernels.AdhesionKernel import AdhesionKernel

#ti.init(arch=ti.gpu,advanced_optimization=False)
ti.init(arch=ti.gpu,advanced_optimization=True)


#gui param
imgSizeX     = 512
imgSizeY     = 512
current_time = 0.0
total_time   = 1.5
eps          = 1e-5
test_id      = 0


#particle param
particleRadius = 0.025
particleDimX   = 20
particleDimY   = 20
particleDimZ   = 20
particleLiquidNum  = particleDimX*particleDimY*particleDimZ


#CFL time step
vs_iter = 0
dv_iter = 0
pr_iter = 0

user_max_t     = 0.005
user_min_t     = 0.0001
deltaT     = ti.field( dtype=ti.f32, shape=(1))


#particle_data.pressure param dfsph only
alpha_coff  = ti.field(dtype=ti.f32, shape=(particleLiquidNum))
kappa       = ti.field(dtype=ti.f32, shape=(particleLiquidNum))
kappa_v     = ti.field(dtype=ti.f32, shape=(particleLiquidNum))




global sph_canvas
global kernel_c
global kernel_adh
global kernel_coh
global particle_data

def init_particle(filename):
    global kernel_c
    global kernel_adh
    global kernel_coh
    global particle_data


    particle_data = ParticleData(particleRadius)
    ZxY = particleDimZ*particleDimY
    dis = particleRadius * 2.0
    
    for i in range(particleLiquidNum):
        particle_data.add_liquid_point([float(i//ZxY - particleDimX /2)* dis + dis*0.5 ,
                                    float((i%ZxY)//particleDimZ)* dis + 0.2, 
                                    float(i%particleDimZ-particleDimZ /2)* dis+ dis*0.5 ])

    
    kernel_c      = CubicKernel(particle_data.hash_grid.searchR)
    kernel_adh    = AdhesionKernel(particle_data.hash_grid.searchR)
    kernel_coh    = CohesionKernel(particle_data.hash_grid.searchR)

    particle_data.add_obj(filename)
    particle_data.setup_data_gpu()
    particle_data.setup_data_cpu()

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
        if particle_data.cg_delta[0] <= particle_data.viscosity_err * particle_data.cg_delta_zero[0] or particle_data.cg_delta_zero[0] < eps:
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
    vel_max_np = particle_data.vel_max.to_numpy()
  
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

    while (particle_data.avg_density_err.to_numpy()[0] > err) and (dv_iter < 10):
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
        err = particle_data.avg_density_err.to_numpy()[0] / float(particleLiquidNum)
        pr_iter += 1
    end_pressure_iter()


@ti.kernel
def reset_param():

    for i in particle_data.vel:
        particle_data.vel[i]      = ti.Vector([0.0, 0.0, 0.0])
        particle_data.omega[i] = ti.Vector([0.0, 0.0, 0.0])
        
        particle_data.pressure[i] = 0.0
        kappa_v[i]  = 0.0
        kappa[i] = 0.0

        deltaT[0] = 0.001

    
@ti.func
def get_viscosity_Ax(x: ti.template(), i):
    ret = ti.Vector([0.0,0.0,0.0])
    cur_neighbor     = particle_data.hash_grid.neighborCount[i]
    k=0
    while k < cur_neighbor:
        j = particle_data.hash_grid.neighbor[i, k]
        r = particle_data.pos[i] - particle_data.pos[j]     
        
        if j < particleLiquidNum:
            ret += particle_data.dim_coff*particle_data.viscosity * particle_data.liqiudMass / particle_data.rho[j]  * (x[i] - x[j]).dot(r)  / (r.norm_sqr() + 0.01*particle_data.hash_grid.searchR*particle_data.hash_grid.searchR) * kernel_c.CubicGradW(r) / particle_data.rho[i] * deltaT[0]
        else:
            ret += particle_data.dim_coff*particle_data.viscosity_b  * particle_data.rho_S0 / particle_data.rho[i] * particle_data.VS0  * x[i].dot(r)  / (r.norm_sqr() + 0.01*particle_data.hash_grid.searchR*particle_data.hash_grid.searchR) * kernel_c.CubicGradW(r) / particle_data.rho[i] * deltaT[0]    
        k+=1
    return x[i] - ret

@ti.kernel
def init_viscosity_para():
    for i in particle_data.vel_guess:
        particle_data.vel_guess[i] += particle_data.vel[i] 

    for i in particle_data.cg_Minv:
        m = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        cur_neighbor     = particle_data.hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]     
            grad_xij = kernel_c.CubicGradW(r).outer_product(r)
            if j < particleLiquidNum:
                m += particle_data.dim_coff * particle_data.viscosity * particle_data.liqiudMass    / particle_data.rho[j]  / (r.norm_sqr() + 0.01*particle_data.hash_grid.searchR*particle_data.hash_grid.searchR) * grad_xij 
            else:
                m += particle_data.dim_coff * particle_data.viscosity_b  * particle_data.rho_S0 / particle_data.rho[i] * particle_data.VS0  / (r.norm_sqr() + 0.01*particle_data.hash_grid.searchR*particle_data.hash_grid.searchR) * grad_xij   
            k+=1
        particle_data.cg_Minv[i] = (ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) - m  * (deltaT[0]/particle_data.rho[i]) ).inverse()

    particle_data.cg_delta_zero[0] = 0.0
    for i in particle_data.cg_r:
        particle_data.cg_r[i]   = particle_data.vel[i] - get_viscosity_Ax(particle_data.vel_guess, i)
        particle_data.cg_dir[i] = particle_data.cg_Minv[i] @ particle_data.cg_r[i]

        particle_data.cg_delta_zero[0] += particle_data.cg_r[i].dot(particle_data.cg_dir[i])
    particle_data.cg_delta[0] = particle_data.cg_delta_zero[0]

@ti.kernel
def compute_viscosity_force():
    cg_dAd  = eps
    for i in particle_data.cg_r:
        particle_data.cg_Ad[i] = get_viscosity_Ax(particle_data.cg_dir, i)
        cg_dAd += particle_data.cg_dir[i].dot(particle_data.cg_Ad[i]) 
    
    
    alpha = particle_data.cg_delta[0] / cg_dAd
    particle_data.cg_delta_old[0] = particle_data.cg_delta[0]
    particle_data.cg_delta[0] = 0.0
    for i in particle_data.cg_r:
        particle_data.vel_guess[i] += alpha * particle_data.cg_dir[i]
        particle_data.cg_r[i] = particle_data.cg_r[i] - alpha * particle_data.cg_Ad[i]
        particle_data.cg_s[i] = particle_data.cg_Minv[i] @ particle_data.cg_r[i]
        particle_data.cg_delta[0] += particle_data.cg_r[i].dot(particle_data.cg_s[i])

        
    beta = particle_data.cg_delta[0] / particle_data.cg_delta_old[0]

    for i in particle_data.cg_r:
        particle_data.cg_dir[i] = particle_data.cg_s[i] + beta * particle_data.cg_dir[i]

@ti.kernel
def compute_density():
    for i in particle_data.rho:
        particle_data.rho[i]  = particle_data.VL0 * kernel_c.Cubic_W_norm(0.0) * particle_data.rho_L0 
        
        cur_neighbor     =  particle_data.hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]
            if j < particleLiquidNum:
                particle_data.rho[i]     += particle_data.VL0 * kernel_c.Cubic_W(r) * particle_data.rho_L0 
            else:
                particle_data.rho[i]     += particle_data.VS0 * kernel_c.Cubic_W(r) * particle_data.rho_S0
            k += 1

@ti.kernel
def compute_tension():
    for i in particle_data.normal:
        particle_data.normal[i] = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor     =  particle_data.hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]
            gardV = kernel_c.CubicGradW(r) 

            if j < particleLiquidNum:
                particle_data.normal[i]      +=  particle_data.liqiudMass / particle_data.rho[j]  * gardV 
            particle_data.normal[i] *= particle_data.hash_grid.searchR

            k += 1

    for i in particle_data.d_vel:
        accel = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor     =  particle_data.hash_grid.neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]
            len2 = r.norm_sqr()

            if j < particleLiquidNum:
                k_ij = 2.0 * particle_data.rho_L0 / (particle_data.rho[i] + particle_data.rho[j])
                if len2 > eps:
                    xixj = r / ti.sqrt(len2)
                    accel     +=  -particle_data.tension_coff * particle_data.liqiudMass * xixj * kernel_coh.Cubic_W(r)
                accel = -particle_data.tension_coff*(particle_data.normal[i] - particle_data.normal[j])
                particle_data.d_vel[i] += k_ij * accel

            else:
                # the box particle are not included
                centre = ti.Vector([0.0, 0.5, 0.0])
                dis = (particle_data.pos[j]-centre).norm()
                if len2 > eps and dis < 0.26:
                    xixj = r / ti.sqrt(len2)
                    particle_data.d_vel[i]     +=  -particle_data.tension_coff_b * particle_data.rho_S0*particle_data.VS0 * xixj * kernel_adh.Cubic_W(r)
            k += 1

@ti.kernel
def compute_vorticity():
    for i in particle_data.omega:
        particle_data.d_omega[i]  = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor     =  particle_data.hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]
            gradV = kernel_c.CubicGradW(r)

            if j < particleLiquidNum:
                particle_data.d_omega[i] += -1.0 / deltaT[0] * particle_data.vorticity_init* particle_data.viscosity_omega *(particle_data.liqiudMass/particle_data.rho[j])*  (particle_data.omega[i] - particle_data.omega[j]) * kernel_c.Cubic_W(r)
                particle_data.d_vel[i] += particle_data.vorticity_coff / particle_data.rho[i] * particle_data.liqiudMass * (particle_data.omega[i] - particle_data.omega[j]).cross(gradV)
                particle_data.d_omega[i] += particle_data.vorticity_coff / particle_data.rho[i]* particle_data.vorticity_init * particle_data.liqiudMass * (particle_data.vel[i]-particle_data.vel[j]).cross(gradV)
            else:
                particle_data.d_vel[i] += particle_data.vorticity_coff / particle_data.rho[i] * particle_data.rho_L0 * particle_data.VS0 * (particle_data.omega[i] - particle_data.omega[j]).cross(gradV)
                particle_data.d_omega[i] += particle_data.vorticity_coff / particle_data.rho[i]* particle_data.vorticity_init * particle_data.rho_L0* particle_data.VL0* (particle_data.vel[i]-particle_data.vel[j]).cross(gradV)
            particle_data.d_omega[i] += -2.0 * particle_data.vorticity_init * particle_data.vorticity_coff * particle_data.omega[i]
            k += 1

    for i in particle_data.omega:
        particle_data.omega[i] += particle_data.d_omega[i] * deltaT[0]


@ti.kernel
def clear_nonpressure():
    for i in particle_data.d_vel:
        particle_data.d_vel[i]  = particle_data.gravity


@ti.kernel
def end_viscosity():
    for i in particle_data.vel:
        particle_data.d_vel[i]  += (particle_data.vel_guess[i] - particle_data.vel[i]) / deltaT[0]
        particle_data.vel_guess[i]    = particle_data.vel_guess[i] - particle_data.vel[i]

@ti.kernel
def compute_dfsph_coff():
    for i in alpha_coff:
        alpha_coff[i] = 0.0
        sum_grad = ti.Vector([0.0, 0.0, 0.0])
        sum_grad_square = 0.0

        cur_neighbor = particle_data.hash_grid.neighborCount[i]
        k=0

        while k < cur_neighbor:
            j  = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]
            gradV       = kernel_c.CubicGradW(r)
            if j < particleLiquidNum:
                temp = particle_data.VL0 * gradV
                sum_grad_square += temp.norm_sqr()
                sum_grad += temp
            else:
                sum_grad +=  particle_data.VS0 * gradV
            k += 1

        sum_grad_square += sum_grad.norm_sqr()

        if sum_grad_square > eps:
            alpha_coff[i] = -1.0 / sum_grad_square
        else:
            alpha_coff[i] = 0.0

@ti.func
def update_drho_divergence(i):
    cur_neighbor = particle_data.hash_grid.neighborCount[i]
    k=0
    particle_data.adv_rho[i] = 0.0
    while k < cur_neighbor:
        j = particle_data.hash_grid.neighbor[i, k]
        r = particle_data.pos[i] - particle_data.pos[j]
        gradV = kernel_c.CubicGradW(r)

        if j < particleLiquidNum:
            particle_data.adv_rho[i] += particle_data.VL0 * ( (particle_data.vel[i]-particle_data.vel[j]).dot(gradV))
        else:
            particle_data.adv_rho[i] += particle_data.VS0 * ( particle_data.vel[i].dot(gradV) )
        k += 1
    particle_data.adv_rho[i] = max(particle_data.adv_rho[i], 0.0)

    if cur_neighbor < 20:
        particle_data.adv_rho[i]  = 0.0

@ti.func
def update_drho_pressure(i):
    cur_neighbor = particle_data.hash_grid.neighborCount[i]
    k=0
    temp = 0.0

    while k < cur_neighbor:
        j = particle_data.hash_grid.neighbor[i, k]
        r = particle_data.pos[i] - particle_data.pos[j]
        gradV = kernel_c.CubicGradW(r)

        if j < particleLiquidNum:
            temp += particle_data.VL0 * ((particle_data.vel[i]-particle_data.vel[j]).dot(gradV))
        else:
            temp += particle_data.VL0 * (particle_data.vel[i].dot(gradV))
        k += 1

    particle_data.adv_rho[i] = particle_data.rho[i] / particle_data.rho_L0 + deltaT[0] * temp
    particle_data.adv_rho[i] = ti.max(1.0, particle_data.adv_rho[i])


@ti.kernel
def warmstart_divergence_vel():
    for i in kappa_v:
        kappa_v[i] = 0.5 * max(kappa_v[i] / deltaT[0] , -0.5*particle_data.rho_L0*particle_data.rho_L0)
        update_drho_divergence(i)

    for i in particle_data.vel:
        if particle_data.adv_rho[i] > 0.0:
            cur_neighbor = particle_data.hash_grid.neighborCount[i]
            k=0

            while k < cur_neighbor:
                j = particle_data.hash_grid.neighbor[i, k]
                r = particle_data.pos[i] - particle_data.pos[j]
                gradV=kernel_c.CubicGradW(r)

                ki = kappa_v[i]
                if j < particleLiquidNum:
                    kj = kappa_v[j]
                    sum = ki + kj
                    if abs(sum) > eps:
                        particle_data.vel[i] += deltaT[0] * sum * particle_data.VL0*gradV
                elif abs(ki) > eps:
                    particle_data.vel[i] += deltaT[0] * ki * particle_data.VS0 * gradV
                k += 1

@ti.kernel
def  begin_divergence_iter():
    for i in kappa_v:
        update_drho_divergence(i)
        alpha_coff[i] =  alpha_coff[i] /  deltaT[0] 
        kappa_v[i] = 0.0
        

@ti.kernel
def  divergence_iter():
    for i in particle_data.vel:
        particle_data.avg_density_err[0] = 0.0
        cur_neighbor = particle_data.hash_grid.neighborCount[i]
        k=0

        bi = particle_data.adv_rho[i]
        ki = bi * alpha_coff[i]
        kappa_v[i] += ki

        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] - particle_data.pos[j]
            gradV=kernel_c.CubicGradW(r)

            if j < particleLiquidNum:
                sum = ki + alpha_coff[j] *particle_data.adv_rho[j]
                if abs(sum) > eps:
                    particle_data.vel[i] += deltaT[0] * sum * particle_data.VL0*gradV
            else:
                if ti.abs(ki) > eps:
                    particle_data.vel[i] += deltaT[0] * ki * particle_data.VS0 * gradV

            k += 1 
    
    for i in particle_data.adv_rho:
        update_drho_divergence(i)
        particle_data.avg_density_err[0] += particle_data.adv_rho[i]


@ti.kernel
def  end_divergence_iter():
    for i in kappa_v:
        kappa_v[i] *= deltaT[0]
        alpha_coff[i] *= deltaT[0]


@ti.kernel
def warmstart_pressure():
    for i in kappa:
        kappa[i] = max(kappa[i] / deltaT[0] / deltaT[0] , -0.5*particle_data.rho_L0*particle_data.rho_L0)

    for i in particle_data.adv_rho:
        if particle_data.adv_rho[i] > particle_data.rho_L0:
            cur_neighbor = particle_data.hash_grid.neighborCount[i]
            k=0

            while k < cur_neighbor:
                j = particle_data.hash_grid.neighbor[i, k]
                r = particle_data.pos[i] - particle_data.pos[j]
                gradV=kernel_c.CubicGradW(r)

                if j < particleLiquidNum:
                    sum = kappa[i] + kappa[j]
                    if abs(sum) > eps:
                        particle_data.vel[i] += deltaT[0] * sum * particle_data.VL0 * gradV 
                elif abs(kappa[i]) > eps:
                    particle_data.vel[i] += deltaT[0] * kappa_v[i] * particle_data.VS0 * gradV
                k += 1


@ti.kernel
def begin_pressure_iter():
    for i in kappa:
        update_drho_pressure(i)
        alpha_coff[i] = alpha_coff[i] / deltaT[0] / deltaT[0]
        kappa[i] = 0.0

@ti.kernel
def  pressure_iter():
    for i in particle_data.vel:
        particle_data.avg_density_err[0] = 0.0
        cur_neighbor = particle_data.hash_grid.neighborCount[i]
        k=0

        bi = particle_data.adv_rho[i] - 1.0
        ki = bi * alpha_coff[i]
        kappa[i] += ki

        while k < cur_neighbor:
            j = particle_data.hash_grid.neighbor[i, k]
            r = particle_data.pos[i] -particle_data.pos[j]
            gradV=kernel_c.CubicGradW(r)

            if j < particleLiquidNum:
                bj = particle_data.adv_rho[j] - 1.0
                kj = bj * alpha_coff[j]
                sum = ki + kj
                if ti.abs(sum) > eps:
                    particle_data.vel[i] += deltaT[0] * sum * particle_data.VL0 * gradV
            elif ti.abs(ki) > eps:
                particle_data.vel[i] += deltaT[0] * ki * particle_data.VS0 * gradV

            k += 1 
    
    for i in particle_data.vel:
        update_drho_pressure(i)
        particle_data.avg_density_err[0] += particle_data.adv_rho[i]-1.0

@ti.kernel
def  end_pressure_iter():
    for i in kappa:
        kappa[i] *= deltaT[0] * deltaT[0]


@ti.kernel
def cfl_time_step(index: ti.i32):
    for i in particle_data.vel_max:
        if index == 1:
            particle_data.vel_max[i] = ti.max( (particle_data.vel[i] + particle_data.d_vel[i] * deltaT[0]).norm_sqr(), 0.1)
        elif i % index == 0:
            offcet = int(index / 2)
            vi = particle_data.vel_max[i]
            vj = particle_data.vel_max[i+offcet]

            if vi > vj:
                particle_data.vel_max[i] =   vi
            else:
                particle_data.vel_max[i] =   vj

 

@ti.kernel
def  update_vel():
    for i in particle_data.vel:
        particle_data.vel[i]  += particle_data.d_vel[i]  * deltaT[0]

@ti.kernel
def  update_pos():
    for i in particle_data.vel:
        particle_data.pos[i]  += particle_data.vel[i]  * deltaT[0]



@ti.kernel
def draw_particle():
    
    for i in particle_data.pos:
        if i < particleLiquidNum:
            sph_canvas.draw_sphere(particle_data.pos[i], ti.Vector([1.0,1.0,1.0]))
    
    for i in particle_data.pos:
        posi = particle_data.pos[i]
        sph_canvas.draw_point(posi, ti.Vector([0.3,0.3,0.3]))

gui = ti.GUI('dfsph', res=(imgSizeX, imgSizeY))
sph_canvas = Canvas(imgSizeX, imgSizeY)
init_particle("box_boundry.obj")
reset_param()

while gui.running:

    #sph_canvas.pitch_cam(0.0, 1.0, 0.0)
    #sph_canvas.yaw_cam(0.0,1.0,0.0)
    sph_canvas.static_cam(0.0,1.0,0.0)

    particle_data.hash_grid.update_grid()

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

    print("time:%.3f"%current_time, "step:%.4f"%dt, "viscorcity:", vs_iter, "divergence:", dv_iter, "particle_data.pressure:", pr_iter)


    

    
    #particle_data.mc_grid.export_surface(current_time)

    '''
    if particle_data.mc_grid.frame == 1:
        particle_data.export_kernel()
        sys.exit()
    '''

    #sph_canvas.export_png(current_time) 
    
    if math.isnan(particle_data.pos.to_numpy()[test_id, 0]) or current_time >= total_time:
        print(particle_data.adv_rho.to_numpy()[test_id], particle_data.pos.to_numpy()[test_id], particle_data.d_vel.to_numpy()[test_id])
        sys.exit()