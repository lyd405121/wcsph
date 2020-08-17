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
depth = ti.var(dt=ti.f32, shape=[imgSize,imgSize])
gui = ti.GUI('wcsph', res=(imgSize,imgSize))



test_id = 0
maxNeighbour = 32


gridR       = 0.05
searchR     = gridR*2.0
invGridR    = 1.0 / gridR
boundary    = 2.0
blockSize   = int(boundary * invGridR)
doublesize  = blockSize*blockSize
gridSize    = blockSize*blockSize*blockSize

particleDimX = 16
particleDimY = 16
particleDimZ = 16
particleLiquidNum  = particleDimX*particleDimY*particleDimZ
particleSolidNum   = doublesize * 2 + (blockSize-2)*blockSize*2 + (blockSize-2)*(blockSize-2)*2 
particleNum        = particleLiquidNum + particleSolidNum

pos         = ti.Vector(3, dt=ti.f32, shape=(particleNum))
inedxInGrid = ti.var( dt=ti.i32, shape=(particleNum))

vel         = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel       = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))

debug_value = ti.var( dt=ti.f32, shape=(particleLiquidNum))
rho_err     = ti.var( dt=ti.f32, shape=(1))

d_rho       = ti.var( dt=ti.f32, shape=(particleLiquidNum))
pci_coff    = ti.var( dt=ti.f32, shape=(particleLiquidNum))
pos_star    = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
vel_star    = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))




neighborCount = ti.var(dt=ti.i32, shape=(particleLiquidNum))
neighbor      = ti.var(dt=ti.i32, shape=(particleLiquidNum, maxNeighbour))

gridCount     = ti.var(dt=ti.i32, shape=(gridSize))
grid          = ti.var(dt=ti.i32, shape=(gridSize, maxNeighbour))

print("gridsize:", gridSize, "gridR:", gridR, "liqiud particle num:", particleLiquidNum, "solid particle num:", particleSolidNum)


@ti.func
def get_length3(v):
    return ti.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

@ti.func
def get_length2(v):
    return ti.sqrt(v.x*v.x+ v.y*v.y)

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
    
    yScale = 1.0    / ti.tan(fovY/2)
    xScale = yScale / ratio
    return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, zf/(zn-zf), zn*zf/(zn-zf)], [0.0, 0.0, -1.0, 0.0] ])
    
    #d3d ortho https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixorthorh
    #yScale = 1.0    / ti.tan(fovY/2)
    #xScale = yScale / ratio
    #return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, 1.0/(zn-zf), zn/(zn-zf)], [0.0, 0.0, 0.0, 1.0] ])
    
    
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
        aa = particleDimZ*particleDimY
        x = i//aa
        y = (i%aa)//particleDimZ
        z = i%particleDimZ
        pos[i]     = ti.cast(ti.Vector([x, y, z]),ti.f32)
        percent    = ti.cast(ti.Vector([particleDimX, particleDimY, particleDimZ]), ti.f32)
        pos[i]     = (pos[i] / percent  * 2.0-1.0 + 1.0 / percent) * ti.Vector([float(particleDimX)* gridR, float(particleDimY)* gridR, float(particleDimZ)* gridR]) * 0.7
        vel[i]     = ti.Vector([0.0, -1.0, 0.0])
        
        
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
            solidIndex  = particleLiquidNum +  doublesize*2 + (blockSize-2)*blockSize*2 + (index.x-1) * (blockSize-2) + index.y
            pos[solidIndex]     = xyz
        elif index.z == blockSize-1:
            solidIndex  = particleLiquidNum + doublesize*2 + (blockSize-2)*blockSize*2 + (blockSize-2) * (blockSize-2) + (index.x-1) * (blockSize-2) + index.y
            pos[solidIndex]     = xyz


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
        d_rho[i]         = 0.0

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
        cur_neighbor     = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]

            d_vel[i] += 0.01 / rho_0 * (vel[i] - vel[j]).dot(r) / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r)
            k += 1
        
        vel_star[i]  = vel[i]
        pos_star[i]  = pos[i]


@ti.kernel
def compute_pci_coff():
    for i in pci_coff:
        Wij    =  ti.Vector([0.0, 0.0, 0.0])
        WijWij = 0.0

        cur_neighbor = neighborCount[i]
        k=0
        while k < cur_neighbor:

            j  = neighbor[i, k]
            r = pos[i] - pos[j]
            gradV       = gradW(r)

            Wij     +=  gradV
            WijWij  +=  gradV.dot(gradV)
            k += 1

        if k > 0:
            pci_coff[i] = 0.5 * (rho_0*rho_0) / (deltaT * deltaT * (Wij.norm_sqr() + WijWij) )
        else:
            pci_coff[i] = rho_0 * 50.0
            print("Liquad too sparse")



@ti.kernel
def update_iter_info():
    for i in vel_star:
        vel_star[i]  += d_vel[i]  * deltaT
        pos_star[i]  += vel_star[i]  * deltaT
        d_vel[i]    = ti.Vector([0.0, 0.0, 0.0])
        rho_err[i]  = 0.0


@ti.kernel
def update_density():
    for i in d_rho:

        d_rho[i]   =  0.0
        cur_neighbor = neighborCount[i]
        k=0
        while k < cur_neighbor:

            j = neighbor[i, k]
            pi = pos_star[i]
            pj = pos[j]
            vi = vel_star[i]
            vj = vel[j]

            if j < particleLiquidNum:
                pj = pos_star[j]
                vj = vel_star[j]

            r = pi - pj
            gradV       = gradW(r)
            d_rho[i]   +=  W(r) + (vi - vj).dot(gradV) * deltaT 
            k += 1
        rho_err[0]  += d_rho[i]
        




@ti.kernel
def update_pressure_force():
    for i in d_vel:
        
        cur_neighbor = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            pi = pos_star[i]
            pj = pos[j]
            if j < particleLiquidNum:
                pj = pos_star[j]
            r = pi - pj
            gradV       = gradW(r)

            pressure         = pci_coff[i] * d_rho[i] / rho_0
            d_vel[i]        +=  -2.0 * pressure / (rho_0 * rho_0 )  * gradV
            
            k += 1


@ti.kernel
def update_pos():
    for i in vel:
        vel[i]   = vel_star[i]
        pos[i]  += vel[i]  * deltaT

@ti.kernel
def draw_particle():
    for i in pos:
        if i < particleLiquidNum:
            draw_solid_sphere(pos[i], ti.Vector([1.0,1.0,1.0]))
        elif i < particleLiquidNum + doublesize *2 + blockSize*(blockSize-2)*2:
            draw_sphere(pos[i], ti.Vector([0.3,0.3,0.3]))


eye        = ti.Vector([0.0, 0.0, 3.0])
target     = ti.Vector([0.0, 0.0, 0.0])
up         = ti.Vector([0.0, 1.0, 0.0])
gravity    = ti.Vector([0.0, -9.81, 0.0])
collisionC = ti.Vector([0.0, 3.0, 0.0])

deltaT     = 0.001
fov        = 1.0
near       = 1.0
far        = 1000.0

rho_0 = 1000.0
pi = 3.1415926
h3 = searchR*searchR*searchR
m_k = 8.0  / (pi*h3)
m_l = 48.0 / (pi*h3)
    
frame = 0

reset_particle()
clear_canvas()
while gui.running:
    
    clear_grid()
    update_grid()
    reset_neighbor()
    find_neighbour()

    compute_nonpressure_force()

    if frame == 0:
        compute_pci_coff()

    iter = 0
    err  = 0.0
    while (err > (rho_0 * 0.001) or iter < 3) and (iter < 10):

        update_iter_info()
        update_density()
        update_pressure_force()
        err = rho_err.to_numpy()[test_id]
        iter += 1
        

    update_pos()
    

    if frame % 10 == 0:
        clear_canvas()
        draw_particle()
    

    gui.set_image(img.to_numpy())
    gui.show()
    frame += 1
    ti.imwrite(img, str(frame)+ ".png")

    # create a PLYWriter
    #np_pos = np.reshape(pos.to_numpy(), (particleLiquidNum, 3))
    #writer = ti.PLYWriter(num_vertices=particleLiquidNum)
    #writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    #writer.export_frame(frame, "wcsph.ply")


    if math.isnan(pos.to_numpy()[test_id, 0]) or frame >= 1500:
        print(d_rho.to_numpy()[test_id], pos.to_numpy()[test_id], d_vel.to_numpy()[test_id])
        ti.imwrite(img, "done.png")
        sys.exit()



