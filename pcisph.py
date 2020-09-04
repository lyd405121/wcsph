import sys
import os
import taichi as ti
import time
import math
import numpy as np


#ti.init(arch=ti.gpu,advanced_optimization=False)
ti.init(arch=ti.gpu,advanced_optimization=True)
imgSize = 512
screenRes = ti.Vector([imgSize, imgSize])
img = ti.Vector(3, dt=ti.f32, shape=[imgSize,imgSize])
depth = ti.field(dtype=ti.f32, shape=[imgSize,imgSize])
gui = ti.GUI('wcsph', res=(imgSize,imgSize))



test_id = 0
maxNeighbour = 32

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

pos         = ti.Vector(3, dt=ti.f32, shape=(particleNum))
inedxInGrid = ti.field( dtype=ti.i32, shape=(particleNum))

pressure    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
vel         = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel       = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel_pre   = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))

debug_value = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
rho_err     = ti.field( dtype=ti.f32, shape=(1))

rho         = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
d_rho       = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
pos_star    = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
vel_star    = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))




neighborCount = ti.field(dtype=ti.i32, shape=(particleLiquidNum))
neighbor      = ti.field(dtype=ti.i32, shape=(particleLiquidNum, maxNeighbour))

gridCount     = ti.field(dtype=ti.i32, shape=(gridSize))
grid          = ti.field(dtype=ti.i32, shape=(gridSize, maxNeighbour))

print("gridsize:", gridSize, "gridR:", gridR, "liqiud particle num:", particleLiquidNum, "solid particle num:", particleSolidNum)



@ti.func
def clamp(v, low_limit, up_limit):
    res = v
    if res < low_limit:
        res = low_limit
        
    if res > up_limit:
        res = up_limit
    return res
    
@ti.func
def clampV(v, low_limit, up_limit):
    res = v
    res.x = clamp(res.x, low_limit, up_limit)
    res.y = clamp(res.y, low_limit, up_limit)
    res.z = clamp(res.z, low_limit, up_limit)
    
    return res
    
@ti.func
def get_proj(fovY, ratio, zn, zf):
    #  d3d perspective https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixperspectiverh 
    # rember it is col major
    # xScale     0          0              0
    # 0        yScale       0              0
    # 0        0        zf/(zn-zf)        -1
    # 0        0        zn*zf/(zn-zf)      0
    # where:
    # yScale = cot(fovY/2)  
    # xScale = yScale / aspect ratio
    
    #yScale = 1.0    / ti.tan(fovY/2)
    #xScale = yScale / ratio
    #return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, zf/(zn-zf), zn*zf/(zn-zf)], [0.0, 0.0, -1.0, 0.0] ])
    
    #d3d ortho https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixorthorh
    yScale = 1.0    / ti.tan(fovY/2)
    xScale = yScale / ratio
    return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, 1.0/(zn-zf), zn/(zn-zf)], [0.0, 0.0, 0.0, 1.0] ])
    
    
@ti.func
def get_view(eye, target, up):
    #  d3d lookat https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixlookatrh
    # rember it is col major
    #zaxis = normal(Eye - At)
    #xaxis = normal(cross(Up, zaxis))
    #yaxis = cross(zaxis, xaxis)
     
    # xaxis.x           yaxis.x           zaxis.x          0
    # xaxis.y           yaxis.y           zaxis.y          0
    # xaxis.z           yaxis.z           zaxis.z          0
    # dot(xaxis, eye)   dot(yaxis, eye)   dot(zaxis, eye)  1    ←  there is something wrong with it，  it should be '-'
    
    zaxis = eye - target
    zaxis = zaxis.normalized()
    xaxis = up.cross( zaxis)
    xaxis = xaxis.normalized()
    yaxis = zaxis.cross( xaxis)
    return ti.Matrix([ [xaxis.x, xaxis.y, xaxis.z, -xaxis.dot(eye)], [yaxis.x, yaxis.y, yaxis.z, -yaxis.dot(eye)], [zaxis.x, zaxis.y, zaxis.z, -zaxis.dot(eye)], [0.0, 0.0, 0.0, 1.0] ])

@ti.func
def transform(v):
    proj = get_proj(fov, 1.0, near, far)
    view = get_view(eye, target, up )
    
    screenP  = proj @ view @ ti.Vector([v.x, v.y, v.z, 1.0])
    screenP /= screenP.w
    
    return ti.Vector([(screenP.x+1.0)*0.5*screenRes.x, (screenP.y+1.0)*0.5*screenRes.y, screenP.z])

@ti.func
def fill_pixel(v, z, c):
    if (v.x >= 0) and  (v.x <screenRes.x) and (v.y >=0 ) and  (v.y < screenRes.y):
        if depth[v] > z:
            img[v] = max(img[v], c)
            depth[v] = z

@ti.func
def draw_sphere(v, c):

    v  = transform(v)
    xc = ti.cast(v.x, ti.i32)
    yc = ti.cast(v.y, ti.i32)

    r=3
    x=0
    y = r
    d = 3 - 2 * r
    
    while x<=y:
        fill_pixel(ti.Vector([ xc + x, yc + y]), v.z, c)
        fill_pixel(ti.Vector([ xc - x, yc + y]), v.z, c)
        fill_pixel(ti.Vector([ xc + x, yc - y]), v.z, c)
        fill_pixel(ti.Vector([ xc - x, yc - y]), v.z, c)
        fill_pixel(ti.Vector([ xc + y, yc + x]), v.z, c)
        fill_pixel(ti.Vector([ xc - y, yc + x]), v.z, c)
        fill_pixel(ti.Vector([ xc + y, yc - x]), v.z, c)
        fill_pixel(ti.Vector([ xc - y, yc - x]), v.z, c)


        if d<0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y = y-1
        x +=1


@ti.func
def draw_solid_sphere(v, c):
    v = transform(v)
    r = 6
    Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])

    for i in range(-r, r+1):
        for j in range(-r, r+1):
            dis = i*i + j*j
            if (dis < r*r):
                fill_pixel(Centre+ti.Vector([i,j]), v.z, c)


@ti.func
def draw_point(v, c):
    v = transform(v)
    Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])
    fill_pixel(Centre, 0.0, c)
                


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
        aa = particleDimZ*particleDimY
        x = i//aa
        y = (i%aa)//particleDimZ
        z = i%particleDimZ
        pos[i]      = ti.cast(ti.Vector([x, y, z]),ti.f32) * particleRadius*2.0 + ti.Vector([0.0, -0.9, 0.0])



    for i in gridCount:
        gridCount[i]=0

        index      = ti.Vector([i//doublesize, (i%doublesize)//blockSize, i%blockSize])
        xyz        = (float(index) * gridR - boundary/2.0  + gridR * 0.5) * ( boundary / (boundary- gridR*1.5) )
        
        if index.x == 0:
            solidIndex  = particleLiquidNum + index.y * blockSize + index.z
            pos[solidIndex]     = xyz

        elif index.x == blockSize-1:
            solidIndex  = particleLiquidNum + doublesize + index.y * blockSize + index.z
            pos[solidIndex]     = xyz

        elif index.y ==0:
            solidIndex  = particleLiquidNum + doublesize*2 + (index.x-1) * blockSize  + index.z
            pos[solidIndex]     = xyz

        elif index.y == blockSize-1:
            solidIndex  = particleLiquidNum + doublesize*2 + (blockSize-2)*blockSize + (index.x-1) * blockSize + index.z
            pos[solidIndex]     = xyz

        elif index.z == 0:
            solidIndex  = particleLiquidNum +  doublesize*2 + (blockSize-2)*blockSize*2 + (index.x-1) * (blockSize-2) + index.y-1
            pos[solidIndex]     = xyz
        elif index.z == blockSize-1:
            solidIndex  = particleLiquidNum + doublesize*2 + (blockSize-2)*blockSize*2 + (blockSize-2) * (blockSize-2) + (index.x-1) * (blockSize-2) + index.y-1
            pos[solidIndex]     = xyz

        
        
@ti.kernel
def clear_canvas():
    for i, j in img:
        img[i, j]=ti.Vector([0, 0, 0])
        depth[i, j] = 1.0
        

@ti.kernel
def clear_grid():
    for i,j in grid:
        grid[i,j] = -1
        gridCount[i]=0


@ti.kernel
def update_grid():
    for i in pos:
        indexV         = clampV( ti.cast((pos[i]+boundary/2.0)*invGridR, ti.i32), 0, blockSize-1)
        index          = indexV.x*doublesize + indexV.y * blockSize + indexV.z
        inedxInGrid[i] = index

        old = ti.atomic_add(gridCount[index] , 1)
        if old > maxNeighbour-1:
            #print("exceed grid", old)
            gridCount[index] = maxNeighbour
        else:
            grid[index, old] = i


@ti.kernel
def reset_neighbor():
    for i,j in neighbor:
        neighbor[i,j]    = -1
        neighborCount[i] = 0

@ti.func
def insert_neighbor(i, colj):
    if colj >= 0:
        k=0
        while k < gridCount[colj]:
            j = grid[colj, k]
            if j >= 0 and (i != j):
                #print( "posi:", pos[i],  "posj:", pos[j])

                r = pos[i] - pos[j]
                r_mod = r.norm()
                if r_mod < searchR:
                    old = ti.atomic_add(neighborCount[i] , 1)
                    if old > maxNeighbour-1:
                        old = old
                        #print("exceed neighbor", old)
                    else:
                        neighbor[i, old] = j
            k += 1

@ti.kernel
def find_neighbour():
    for i in neighborCount:
        index = inedxInGrid[i]

        insert_neighbor(i, index)
        insert_neighbor(i, index-1)
        insert_neighbor(i, index+1) 
        insert_neighbor(i, index+doublesize)
        insert_neighbor(i, index-doublesize)
        insert_neighbor(i, index+blockSize)
        insert_neighbor(i, index-blockSize)     

        insert_neighbor(i, index+doublesize+blockSize)
        insert_neighbor(i, index+doublesize-blockSize)
        insert_neighbor(i, index-doublesize+blockSize)
        insert_neighbor(i, index-doublesize-blockSize)  

        insert_neighbor(i, index+doublesize+1)
        insert_neighbor(i, index-doublesize+1)
        insert_neighbor(i, index+blockSize+1)
        insert_neighbor(i, index-blockSize+1)       
        insert_neighbor(i, index+doublesize-1)
        insert_neighbor(i, index-doublesize-1)
        insert_neighbor(i, index+blockSize-1)
        insert_neighbor(i, index-blockSize-1)     

        insert_neighbor(i, index+doublesize+blockSize+1)
        insert_neighbor(i, index+doublesize-blockSize+1)
        insert_neighbor(i, index-doublesize+blockSize+1)
        insert_neighbor(i, index-doublesize-blockSize+1)
        insert_neighbor(i, index+doublesize+blockSize-1)
        insert_neighbor(i, index+doublesize-blockSize-1)
        insert_neighbor(i, index-doublesize+blockSize-1)
        insert_neighbor(i, index-doublesize-blockSize-1)



@ti.kernel
def compute_nonpressure_force():
    for i in d_vel:
        d_vel[i] = gravity
        rho[i]  = VL0 * W_norm(0.0) * rho_0 

        cur_neighbor     = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]

            if j < particleLiquidNum:
                rho[i]     += VL0 * W(r) * rho_0 
                d_vel[i]   += visorcity / rho_0 * (vel[i] - vel[j]).dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
            else:
                rho[i]     += VS0 * W(r) * rho_0
                d_vel[i]   += visorcity / rho_0 * vel[i] .dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
            
            k += 1
        vel_star[i]  = vel[i]
        pos_star[i]  = pos[i]
        d_vel_pre[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def update_iter_info():
    for i in vel_star:
        vel_star[i]  = vel[i] + (d_vel[i]+d_vel_pre[i])  * deltaT
        pos_star[i]  = pos[i] + vel_star[i]  * deltaT

        rho_err[i]   = 0.0
        pressure[i]  = 0.0

@ti.kernel
def predict_density():
    for i in rho:
        d_rho[i] = VL0 * W_norm(0.0)

        cur_neighbor     = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]
            WW = W(r)
            if j < particleLiquidNum:
                d_rho[i]     += VL0 * WW
            else:
                d_rho[i]     += VS0 * WW
            k += 1

        d_rho[i] = ti.max(d_rho[i], 1.0)
        pressure[i] += pci_coff * (d_rho[i]-1.0) / (deltaT * deltaT)
        rho_err[0]  += d_rho[i]-1.0

    for i in d_vel_pre:
        d_vel_pre[i]      = ti.Vector([0.0, 0.0, 0.0])

        cur_neighbor = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            pi = pos_star[i]
            pj = pos[j]
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
        vel[i]  += (d_vel[i]+d_vel_pre[i]) * deltaT
        pos[i]  += vel[i]  * deltaT

@ti.kernel
def draw_particle():
    for i in pos:
        if i < particleLiquidNum:
            #draw_solid_sphere(pos[i], ti.Vector([1.0,1.0,1.0]))
            draw_sphere(pos[i], ti.Vector([1.0,1.0,1.0]))
        elif i < particleLiquidNum + doublesize *2 + blockSize*(blockSize-2)*2:
            draw_sphere(pos[i], ti.Vector([0.3,0.3,0.3]))


eye        = ti.Vector([0.0, 0.0, 3.0])
target     = ti.Vector([0.0, 0.0, 0.0])
up         = ti.Vector([0.0, 1.0, 0.0])
gravity    = ti.Vector([0.0, -9.81, 0.0])
collisionC = ti.Vector([0.0, 3.0, 0.0])

deltaT     = 0.001
fov        = 2.0
near       = 1.0
far        = 1000.0

visorcity = 0.02
rho_0 = 1000.0
VL0    = particleRadius * particleRadius * particleRadius * 0.8 * 8.0
VS0    = VL0 * 2.0

pi = 3.1415926
h3 = searchR*searchR*searchR
m_k = 8.0  / (pi*h3)
m_l = 48.0 / (pi*h3)
    
frame = 0
iterNum = 0.03 /  deltaT
totalFrame = 120

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

pci_coff = GetPciCoff()
reset_particle()
clear_canvas()

while gui.running:
    clear_grid()
    update_grid()
    reset_neighbor()
    find_neighbour()

    compute_nonpressure_force()

    iter = 0
    err  = 0.0
    while (err >  0.001 or iter < 3) and (iter < 10):
        update_iter_info()
        predict_density()
        err = rho_err.to_numpy()[0] / float(particleLiquidNum)
        iter += 1

    update_pos()
    

    if frame % iterNum == 0:
        clear_canvas()
        draw_particle()
        #ti.imwrite(img, str(frame//iterNum)+ ".png")

    gui.set_image(img.to_numpy())
    gui.show()
    frame += 1
    

    #ti.imwrite(img, str(frame)+ ".png")

    # create a PLYWriter
    #np_pos = np.reshape(pos.to_numpy(), (particleLiquidNum, 3))
    #writer = ti.PLYWriter(num_vertices=particleLiquidNum)
    #writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    #writer.export_frame(frame, "wcsph.ply")


    if math.isnan(pos.to_numpy()[test_id, 0]) or frame >= totalFrame * iterNum:
        print(d_rho.to_numpy()[test_id], pos.to_numpy()[test_id], d_vel.to_numpy()[test_id])
        sys.exit()



