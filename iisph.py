import sys
import os
import taichi as ti
import time
import math
import numpy as np


#ti.init(arch=ti.gpu,advanced_optimization=False)
ti.init(arch=ti.gpu,advanced_optimization=True)
imgSizeX = 720
imgSizeY = 480
screenRes = ti.Vector([imgSizeX, imgSizeY])
img = ti.Vector(3, dt=ti.f32, shape=[imgSizeX, imgSizeY])
depth = ti.field(dtype=ti.f32, shape=[imgSizeX, imgSizeY])
gui = ti.GUI('iisph', res=(imgSizeX, imgSizeY))

frame = 0
eps = 1e-5
test_id = 0
maxNeighbour = 128
average_iter = 0

particleRadius = 0.025
gridR       = particleRadius * 2.0
searchR     = gridR*2.0
invGridR    = 1.0 / gridR


particleDimX = 16
particleDimY = 8
particleDimZ = 8
particleLiquidNum  = particleDimX*particleDimY*particleDimZ

particleSolidNum   = 0
particleNum        = 0

pos         = ti.Vector(3, dt=ti.f32)
inedxInGrid = ti.field( dtype=ti.i32)

vel_guess   = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
vel         = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
d_vel       = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
a_ii        = ti.field(dtype=ti.f32, shape=(particleLiquidNum))
d_ii        = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
dij_pj      = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))


avg_density_err = ti.field( dtype=ti.f32, shape=(1))
cg_delta     = ti.field( dtype=ti.f32, shape=(1))
cg_delta_old     = ti.field( dtype=ti.f32, shape=(1))
cg_delta_zero     = ti.field( dtype=ti.f32, shape=(1))

cg_Minv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(particleLiquidNum))
cg_r = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
cg_dir   = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
cg_Ad   = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))
cg_s   = ti.Vector(3, dt=ti.f32, shape=(particleLiquidNum))

min_boundary   = ti.Vector(3, dt=ti.f32, shape=(1))
max_boundary   = ti.Vector(3, dt=ti.f32, shape=(1))
blockSize  = ti.Vector(3, dt=ti.i32, shape=(1))
eye        = ti.Vector(3, dt=ti.f32, shape=(1))
target     = ti.Vector(3, dt=ti.f32, shape=(1))

pressure_pre    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
pressure    = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
rho         = ti.field( dtype=ti.f32, shape=(particleLiquidNum))
d_rho       = ti.field( dtype=ti.f32, shape=(particleLiquidNum))

neighborCount = ti.field(dtype=ti.i32, shape=(particleLiquidNum))
neighbor      = ti.field(dtype=ti.i32, shape=(particleLiquidNum, maxNeighbour))


global hash_grid
@ti.data_oriented
class HashMap:
    def __init__(self, n):
        self.count        = n

        self.gridCount = ti.field(dtype=ti.i32)
        self.grid       = ti.field(dtype=ti.i32)
        
        ti.root.dense(ti.i, self.count).place(self.gridCount)
        ti.root.dense(ti.ij, (self.count, maxNeighbour)).place(self.grid)
    
    @ti.kernel
    def update_grid(self):
        for i,j in self.grid:
            self.grid[i,j] = -1
            self.gridCount[i]=0

        for i in pos:
            indexV         = clampV( ti.cast((pos[i] - min_boundary[0])*invGridR, ti.i32), ti.Vector([0,0,0]), blockSize[0]-1)
            hash_index     = self.get_cell_hash(indexV)

            old = ti.atomic_add(self.gridCount[hash_index] , 1)
            if old > maxNeighbour-1:
                print("exceed grid", old)
                self.gridCount[hash_index] = maxNeighbour
            else:
                self.grid[hash_index, old] = i
        

    @ti.func
    def get_cell_hash(self, a):
        p1 = 73856093 * a.x
        p2 = 19349663 * a.y
        p3 = 83492791 * a.z

        return ((p1^p2^p3) % self.count  +  self.count ) % self.count  


def load_boundry(filename):
    global hash_grid
    global particleLiquidNum
    global particleNum
    global particleSolidNum

    vertices = []
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertices.append(v)

    particleSolidNum = len(vertices)
    particleNum = particleSolidNum + particleLiquidNum

    maxboundarynp = np.ones(shape=(1,3), dtype=np.float32)
    minboundarynp = np.ones(shape=(1,3), dtype=np.float32)
    for i in range(3):
        maxboundarynp[0, i] = -10000.0
        minboundarynp[0, i] = 10000.0

    arrV = np.ones(shape=(particleNum, 3), dtype=np.float32)
    for i in range(particleNum):
        if i < particleLiquidNum:

            aa = particleDimZ*particleDimY
            x = float(i//aa - particleDimX / 2)
            y = float((i%aa)//particleDimZ)
            z = float(i%particleDimZ - particleDimZ/2)
            arrV[i, 0]  = x * particleRadius*2.0
            arrV[i, 1]  = y * particleRadius*2.0 + 0.7
            arrV[i, 2]  = z * particleRadius*2.0
        else:
            arrV[i, 0] = vertices[i-particleLiquidNum][0]
            arrV[i, 1] = vertices[i-particleLiquidNum][1]
            arrV[i, 2] = vertices[i-particleLiquidNum][2]

        for j in range(3):
            maxboundarynp[0, j] = max(maxboundarynp[0, j], arrV[i,j])
            minboundarynp[0, j] = min(minboundarynp[0, j], arrV[i,j])
    

    blocknp   = np.ones(shape=(1,3), dtype=np.int32)
    for i in range(3):
        blocknp[0, i]    = int((maxboundarynp[0, i] - minboundarynp[0, i]) * invGridR)

    gridSize     = int(blocknp[0, 0]*blocknp[0, 1]*blocknp[0, 2])

    ti.root.dense(ti.i, particleNum ).place(pos)
    ti.root.dense(ti.i, particleNum ).place(inedxInGrid)
    hash_grid = HashMap(particleNum)

    max_boundary.from_numpy(maxboundarynp)
    min_boundary.from_numpy(minboundarynp)
    blockSize.from_numpy(blocknp)
    pos.from_numpy(arrV)

    print("gridsize:", gridSize, "gridR:", gridR, "liqiud particle num:", particleLiquidNum, "solid particle num:", particleSolidNum)

def compute_nonpressure_force():
    global average_iter
    global frame

    init_viscosity_para()
    iter = 0
    while iter < 100:
        compute_viscosity_force()
        iter+=1
        #if iter > average_iter:
        #    print("iter:", iter, "error:", cg_delta[0], "of", cg_delta_zero[0])

        if cg_delta[0] <= 0.01 * cg_delta_zero[0]:
            break
    combine_nonpressure()

    average_iter += iter
    print("iter:", iter, "average", average_iter / (frame+1) ) 




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
    res.x = clamp(res.x, low_limit.x, up_limit.x)
    res.y = clamp(res.y, low_limit.y, up_limit.y)
    res.z = clamp(res.z, low_limit.z, up_limit.z)
    
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
    proj = get_proj(fov, imgSizeX / imgSizeY, near, far)
    view = get_view(eye[0], target[0], up )
    
    screenP  = proj @ view @ ti.Vector([v.x, v.y, v.z, 1.0])
    screenP /= screenP.w
    
    return ti.Vector([(screenP.x+1.0)*0.5*screenRes.x, (screenP.y+1.0)*0.5*screenRes.y, screenP.z])

@ti.func
def fill_pixel(v, z, c):
    if (v.x >= 0) and  (v.x <screenRes.x) and (v.y >=0 ) and  (v.y < screenRes.y):
        if depth[v] > z:
            img[v] = c
            depth[v] = z

@ti.func
def draw_sphere(v, c):

    v  = transform(v)
    xc = ti.cast(v.x, ti.i32)
    yc = ti.cast(v.y, ti.i32)

    r=4
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
def draw_point(v,c):
    v = transform(v)
    Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])
    fill_pixel(Centre, v.z, c)

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
    fill_pixel(Centre, v.z, c)
                
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


        
        
@ti.kernel
def clear_canvas():
    for i, j in img:
        img[i, j]=ti.Vector([0, 0, 0])
        depth[i, j] = 1.0
        

@ti.func
def insert_neighbor(i, index_neigh):
    if index_neigh.x >= 0 and index_neigh.x <= blockSize[0].x and \
    index_neigh.y >= 0 and index_neigh.y <= blockSize[1].y  and \
    index_neigh.z >= 0 and index_neigh.z <= blockSize[2].z :

        hash_index = hash_grid.get_cell_hash(index_neigh)
        k=0
        while k < hash_grid.gridCount[hash_index]:
            j = hash_grid.grid[hash_index, k]
            if j >= 0 and (i != j):
                r = pos[i] - pos[j]
                r_mod = r.norm()
                if r_mod < searchR:
                    old = ti.atomic_add(neighborCount[i] , 1)
                    if old > maxNeighbour-1:
                        old = old
                        print("exceed neighbor", old)
                    else:
                        neighbor[i, old] = j
            k += 1

@ti.kernel
def find_neighbour():
    for i,j in neighbor:
        neighbor[i,j]    = -1
        neighborCount[i] = 0

    for i in neighborCount:
        indexV         = clampV( ti.cast((pos[i] - min_boundary[0])*invGridR, ti.i32), ti.Vector([0,0,0]), blockSize[0]-1)
        for m in range(-1,2):
            for n in range(-1,2):
                for q in range(-1,2):
                    insert_neighbor(i, ti.Vector([m, n, q]) + indexV)






@ti.func
def get_viscosity_Ax(x: ti.template(), i):
    ret = ti.Vector([0.0,0.0,0.0])
    cur_neighbor     = neighborCount[i]
    k=0
    while k < cur_neighbor:
        j = neighbor[i, k]
        r = pos[i] - pos[j]     
        
        if j < particleLiquidNum:
            ret += dim_coff*viscosity    / rho[j]  * (x[i] - x[j]).dot(r)  / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r) / rho[i] * deltaT
        else:
            ret += dim_coff*viscosity_b  * rho_S0 / rho[i] * VS0  * x[i].dot(r)  / (r.norm_sqr() + 0.01*searchR*searchR) * gradW(r) / rho[i] * deltaT    
        k+=1
    return x[i] - ret

@ti.kernel
def init_viscosity_para():
    for i in vel_guess:
        vel_guess[i] = vel[i]

    for i in cg_Minv:
        m = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        cur_neighbor     = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]     
            grad_xij = gradW(r).outer_product(r)
            if j < particleLiquidNum:
                m += dim_coff * viscosity    / rho[j]  / (r.norm_sqr() + 0.01*searchR*searchR) * grad_xij 
            else:
                m += dim_coff * viscosity_b  * rho_S0 / rho[i] * VS0  / (r.norm_sqr() + 0.01*searchR*searchR) * grad_xij   
            k+=1
        cg_Minv[i] = (ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) - m  * (deltaT/rho[i]) ).inverse()
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

        cur_neighbor     = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]
            if j < particleLiquidNum:
                rho[i]     += VL0 * W(r) * rho_L0 
            else:
                rho[i]     += VS0 * W(r) * rho_S0
            k += 1

@ti.kernel
def combine_nonpressure():
    for i in d_vel:
        d_vel[i]  = gravity + (vel_guess[i] - vel[i]) / deltaT


@ti.kernel
def compute_init_coff():

    for i in vel:
        vel[i]  += d_vel[i] * deltaT

    for i in d_ii:
        d_ii[i] = ti.Vector([0.0, 0.0, 0.0])

        cur_neighbor = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j  = neighbor[i, k]
            r = pos[i] - pos[j]
            gradV       = gradW(r)

            inv_den     = rho_L0 / rho[i] 
            d_ii[i]     +=  -VL0 * inv_den * inv_den * gradV
            k += 1



    for i in a_ii:
        a_ii[i] = 0.0
        density = rho[i] / rho_L0
        d_rho[i] = density
        pressure_pre[i] = 0.5 * pressure[i]

        cur_neighbor = neighborCount[i]
        k=0

        while k < cur_neighbor:
            j  = neighbor[i, k]
            r = pos[i] - pos[j]
            gradV       = gradW(r)
            

            if j < particleLiquidNum:
                d_rho[i]     +=  deltaT *  VL0 * ((vel[i] - vel[j]).dot(gradV))
            else:
                d_rho[i]     +=  deltaT *  VS0 * (vel[i] .dot(gradV))


            d_ji = VL0 / (density*density) * gradV
            a_ii[i]     +=   VL0 * (d_ii[i] -  d_ji).dot(gradV)
            k += 1

@ti.kernel
def update_iter_info():
    for i in dij_pj:

        avg_density_err[0] = 0.0
        dij_pj[i] = ti.Vector([0.0, 0.0, 0.0])
        cur_neighbor = neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = neighbor[i, k]
            if j < particleLiquidNum:
                r = pos[i] - pos[j]
                gradV=gradW(r)
                densityj = rho[j] / rho_L0
                dij_pj[i] += -VL0/(densityj*densityj)*pressure_pre[j]*gradV
            k += 1

@ti.kernel
def update_pressure_force():
    for i in pressure:
        sum=0.0
        cur_neighbor = neighborCount[i]
        k=0

        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]
            gradV       = gradW(r)

            if j < particleLiquidNum:
                density = rho[i] / rho_L0
                dji = VL0 / (density*density) * gradV
                d_ji_pi = dji *  pressure_pre[i]
                d_jk_pk = dij_pj[j] 
                sum +=  VL0 * ( dij_pj[i] - d_ii[i]*pressure_pre[i] - (d_jk_pk - d_ji_pi)).dot(gradV)
            else:
                sum +=  VS0 * dij_pj[i].dot(gradV)
            k += 1


        b = 1.0 - d_rho[i]
        h2     = deltaT * deltaT

        denom = a_ii[i]*h2
        if (ti.abs(denom) > 1.0e-9):
            pressure[i] = ti.max(    (1.0 - omega) *pressure_pre[i] + omega / denom * (b - h2*sum), 0.0)
            #print( d_rho[i],rho[i] / rho_L0, h2*sum, omega / denom)
        else:
            pressure[i] = 0.0
        
        if pressure[i] != 0.0:
            avg_density_err[0] += (a_ii[i]*pressure[i]  + sum)*h2 - b
        pressure_pre[i] = pressure[i]
        
@ti.kernel
def update_pos():

    for i in d_vel:
        d_vel[i]      = ti.Vector([0.0, 0.0, 0.0])

        cur_neighbor = neighborCount[i]
        k=0
        while k < cur_neighbor:
            j = neighbor[i, k]
            r = pos[i] - pos[j]
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
        vel[i]  += d_vel[i] * deltaT
        pos[i]  += vel[i]  * deltaT

@ti.kernel
def draw_particle():
    
    for i in pos:
        if i < particleLiquidNum:
            draw_solid_sphere(pos[i], ti.Vector([1.0,1.0,1.0]))

    for i in pos:
        if i> particleLiquidNum and pos[i].z < 0.3 and pos[i].z > -0.3 and pos[i].x < 1.12 and pos[i].x > -1.0 and pos[i].y > 0.01:
            draw_sphere(pos[i], ti.Vector([1.0,0.0,0.0]))
        elif i> particleLiquidNum and pos[i].z < 0.49:
            draw_point(pos[i], ti.Vector([0.3,0.3,0.3]))

up         = ti.Vector([0.0, 1.0, 0.0])
gravity    = ti.Vector([0.0, -9.81, 0.0])

deltaT     = 0.001
fov        = 1.0
near       = 1.0
far        = 1000.0

omega = 0.5
dim_coff  = 10.0
viscosity = 100.0
viscosity_b = 100.0

rho_L0 = 1000.0
rho_S0 = rho_L0
VL0    = particleRadius * particleRadius * particleRadius * 0.8 * 8.0
VS0    = VL0 

pi    = 3.1415926
h3    = searchR*searchR*searchR
m_k   = 8.0  / (pi*h3)
m_l   = 48.0 / (pi*h3)
iterNum = 0.02 /  deltaT
totalFrame = 50

load_boundry("boundry.obj")
reset_particle()
clear_canvas()
while gui.running:


    eye_np = eye.to_numpy()
    target_np = target.to_numpy()
    eye_np[0][0] = 0.5
    eye_np[0][1] = 1.0
    eye_np[0][2] = 2.0 
    target_np[0][0] = 0.0 
    target_np[0][1] = 0.0
    target_np[0][2] = -1.0
    eye.from_numpy(eye_np)
    target.from_numpy(target_np)


    hash_grid.update_grid()
    find_neighbour()

    compute_density()
    compute_nonpressure_force()
    compute_init_coff()
    

    iter = 0
    err  = 0.0
    while (err > 0.0001 or iter < 4) and (iter < 10):
        update_iter_info()
        update_pressure_force()
        
        err = avg_density_err.to_numpy()[0] / float(particleLiquidNum)
        iter += 1

    update_pos()

    if frame % iterNum == 0:
        clear_canvas()
        draw_particle()
        #ti.imwrite(img, str(frame//iterNum)+ ".png")

    gui.set_image(img.to_numpy())
    gui.show()
    frame += 1
    
    #print(d_rho.to_numpy()[test_id], rho.to_numpy()[test_id])

    if math.isnan(pos.to_numpy()[test_id, 0]) or frame >= totalFrame * iterNum:
        print(d_rho.to_numpy()[test_id], pos.to_numpy()[test_id], d_vel.to_numpy()[test_id])
        sys.exit()



