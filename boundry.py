import sys
import os
import taichi as ti
import time
import math
import numpy as np


ti.init(arch=ti.gpu,advanced_optimization=True)
imgSize = 512
screenRes = ti.Vector([imgSize, imgSize])
img       = ti.Vector(3, dt=ti.f32, shape=[imgSize,imgSize])
depth     = ti.field(dtype=ti.f32, shape=[imgSize,imgSize])
gui       = ti.GUI('boundry', res=(imgSize,imgSize))



particleRadius = 0.025
gridR          = particleRadius / math.sqrt(3.0)

tri_vertices = ti.Vector(3, dt=ti.f32)
tri_normal   = ti.Vector(3, dt=ti.f32)
tri_face     = ti.Vector(3, dt=ti.i32)

init_pos = ti.Vector(3, dt=ti.f32)
init_cell = ti.Vector(3, dt=ti.i32)
init_id = ti.field(dtype=ti.i32)

possion_sample = ti.Vector(3, dt=ti.f32)

phase_group = ti.Vector(3, dt=ti.i32)
phase_group_count = ti.field(dtype=ti.i32)

tri_area     = ti.field(dtype=ti.f32)
totalArea    = 0.0
hash_count_gpu    = ti.field( dtype=ti.i32, shape=(1))
hash_trace        = ti.field( dtype=ti.i32)
possion_sample_count    = ti.field( dtype=ti.i32, shape=(1))

min_point = ti.Vector([100000.0, 100000.0, 100000.0])
max_point = ti.Vector([-100000.0, -100000.0, -100000.0])


eye        = ti.Vector([0.0, 0.0, 2.0])
target     = ti.Vector([0.0, 0.0, 0.0])
up         = ti.Vector([0.0, 1.0, 0.0])

fov        = 2.0
near       = 1.0
far        = 1000.0


faceNum = 0
pi = 3.1415926
numInitialPoints = 0
padding_num = 0
maxArea = 0.0

phase_block_size = 27
phase_vec_max = 0
hash_map_size = 0
hash_sample_size = 5

global hMap

@ti.data_oriented
class HashMap:
    def __init__(self, n):
        self.count        = n
        self.cell         = ti.Vector(3, dt=ti.i32)
        self.start_index  = ti.field(dtype=ti.i32)

        self.sample_count = ti.field(dtype=ti.i32)
        self.sample       = ti.field(dtype=ti.i32)
        
        ti.root.dense(ti.i, self.count).place(self.cell)
        ti.root.dense(ti.i, self.count).place(self.start_index)
        ti.root.dense(ti.i, self.count).place(self.sample_count)
        ti.root.dense(ti.ij, (self.count, hash_sample_size)).place(self.sample)


def get_pot_num(num):
    m = 1
    while m<num:
        m=m<<1
    return m >>1

def detect_hmap():
    hash_cpu = hash_trace.to_numpy()
    hash_count_cpu = 0
    for i in range(numInitialPoints):
        if(hash_cpu[i] != 0):
            hash_count_cpu += 1
    if hash_count_cpu != hash_count_gpu[0] :
        print("hash map error!", "cpu:", hash_count_cpu, "gpu:", hash_count_gpu[0] )
    else:
        print("hash map ok!", "cpu:", hash_count_cpu, "gpu:", hash_count_gpu[0] )

def loadObj(filename):
    vertices = []
    faces = []
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertices.append(v)
        elif values[0] == 'f':
            for v in values[1:]:
                w = v.split('/')
                faces.append(int(w[0]))

    global padding_num
    global numInitialPoints
    global maxArea
    global totalArea
    global faceNum
    global phase_vec_max
    global hash_map_size
    global hMap
    
    faceNum = len(faces)//3
    verticeNum = len(vertices)
    

    arrV = np.ones(shape=(verticeNum, 3), dtype=np.float32)
    arrN = np.ones(shape=(verticeNum, 3), dtype=np.float32)
    arrF = np.ones(shape=(faceNum, 3), dtype=np.int32)
    arrArea = np.ones(shape=(faceNum), dtype=np.float32)

    for i in range(verticeNum):
        arrV[i, 0] = vertices[i][0]
        arrV[i, 1] = vertices[i][1]
        arrV[i, 2] = vertices[i][2]

        min_point[0] = min(arrV[i, 0], min_point[0])
        min_point[1] = min(arrV[i, 1], min_point[1])
        min_point[2] = min(arrV[i, 2], min_point[2])

        max_point[0] = max(arrV[i, 0], max_point[0])
        max_point[1] = max(arrV[i, 1], max_point[1])
        max_point[2] = max(arrV[i, 2], max_point[2])
    
    for i in range(faceNum):
        arrF[i, 0] = faces[3*i]-1
        arrF[i, 1] = faces[3*i+1]-1
        arrF[i, 2] = faces[3*i+2]-1

        a = arrV[arrF[i, 0]]
        b = arrV[arrF[i, 1]]
        c = arrV[arrF[i, 2]]

        d1 = b - a
        d2 = c - a


        arrN[arrF[i, 0]] = np.cross(d1, d2)
        arrArea[i] = np.linalg.norm(arrN[arrF[i, 0]])* 0.5

        arrN[arrF[i, 0]] = arrN[arrF[i, 0]] / np.linalg.norm(arrN[arrF[i, 0]])
        arrN[arrF[i, 1]] = arrN[arrF[i, 0]]
        arrN[arrF[i, 2]] = arrN[arrF[i, 0]]

        totalArea += arrArea[i]
        maxArea = max(arrArea[i], maxArea)

    
    circleArea = pi * particleRadius * particleRadius
    numInitialPoints = (int)(40.0 * (totalArea / circleArea)) 
    padding_num = get_pot_num(numInitialPoints) <<1
    phase_vec_max = numInitialPoints//8
    hash_map_size = numInitialPoints*2


    print("point num:", verticeNum, "tri num:", faceNum, )
    print("bound box:", min_point, max_point)
    print("total area:", totalArea, "max area:", maxArea)
    print("Si num:", numInitialPoints, "padding num:", padding_num)

    ti.root.dense(ti.i, verticeNum ).place(tri_vertices)
    ti.root.dense(ti.i, verticeNum ).place(tri_normal)

    ti.root.dense(ti.i, faceNum).place(tri_face)
    ti.root.dense(ti.i, faceNum).place(tri_area)

    ti.root.dense(ti.i,  padding_num).place(init_pos)
    ti.root.dense(ti.i,  padding_num).place(init_cell)
    ti.root.dense(ti.i,  padding_num).place(init_id)

    ti.root.dense(ti.i,  numInitialPoints).place(hash_trace)
    ti.root.dense(ti.i,  numInitialPoints).place(possion_sample)
    ti.root.dense(ti.i,  phase_block_size).place(phase_group_count)
    ti.root.dense(ti.ij, (phase_block_size, phase_vec_max)).place(phase_group)

    hMap = HashMap(hash_map_size)

    tri_vertices.from_numpy(arrV)
    tri_face.from_numpy(arrF)
    tri_normal.from_numpy(arrN)
    tri_area.from_numpy(arrArea)


def write_info(filemode):
    fo = open("test.txt", filemode)
    cell = init_cell.to_numpy()
    id   = init_id.to_numpy()
    pos  = init_pos.to_numpy()
    for i in range(numInitialPoints):
        print(cell[i], id[i], pos[i], file = fo)
    fo.close()


def gpu_bitonic_sort():

    #write_info("w")
    i = 2 
    while(i<= padding_num):
        j = i >> 1
        while(j>0):
            gpu_merge(j, i)
            j = j//2
        i = i << 1
    #write_info("w")

    

    
    


@ti.func
def get_proj(fovY, ratio, zn, zf):
    #yScale = 1.0    / ti.tan(fovY/2)
    #xScale = yScale / ratio
    #return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, zf/(zn-zf), zn*zf/(zn-zf)], [0.0, 0.0, -1.0, 0.0] ])
    
    yScale = 1.0    / ti.tan(fovY/2)
    xScale = yScale / ratio
    return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, 1.0/(zn-zf), zn/(zn-zf)], [0.0, 0.0, 0.0, 1.0] ])
    
    
@ti.func
def get_view(eye, target, up):
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
            img[v] =  c
            depth[v] = z

        
@ti.func
def draw_point(v,c):
    v = transform(v)
    Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])
    fill_pixel(Centre, v.z, c)

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
        
@ti.kernel
def clear_canvas():
    for i, j in img:
        img[i, j]=ti.Vector([0, 0, 0])
        depth[i, j] = 1.0
           
          
@ti.kernel
def init_point_set():
    for i in init_cell:
        if i < numInitialPoints:
            rn1 = ti.sqrt(ti.random())
            bc1 = 1.0 - rn1
            bc2 = ti.random()*rn1
            bc3 = 1.0 - bc1 - bc2

            randIndex = 0

            while 1:
                randIndex = (int)( faceNum*ti.random() )
                if (ti.random() < tri_area[randIndex] / maxArea):
                    break
                
            a = tri_vertices[tri_face[randIndex][0]]
            b = tri_vertices[tri_face[randIndex][1]]
            c = tri_vertices[tri_face[randIndex][2]]

            init_pos[i]  = bc1*a + bc2*b + bc3*c
            init_id[i]   = randIndex
            init_cell[i] = ti.cast( (init_pos[i] - min_point) / gridR, ti.i32)+1
        else:
            init_cell[i] = ti.Vector([1000000, 1000000, 1000000])
        

@ti.kernel
def build_hmap():
    for i in ti.ndrange(numInitialPoints):
        hash_trace[i] = 0

        if (compare_cell(i, i-1) != 0) or (i==0) :

            hash_index      = (get_cell_hash(init_cell[i])  % hash_map_size +  hash_map_size)% hash_map_size
            
            hMap.start_index[hash_index] = i
            hMap.cell[hash_index]        = init_cell[i]

            #detect hash error
            hash_trace[i] = hash_index
            hash_count_gpu[0] +=1 

            phase_index = init_cell[i][0]%3 + 3*(init_cell[i][1]%3) + 9*(init_cell[i][2]%3)
            old = ti.atomic_add(phase_group_count[phase_index], 1)

            if old < (phase_vec_max):
                phase_group[phase_index, old] = init_cell[i]
            else:
                print("longer phase_group is needed!")


@ti.kernel
def draw_particle():

    #for i in init_pos:
    #    draw_sphere(init_pos[i], ti.Vector([1.0,1.0,1.0]))
    
    for i in possion_sample:
        if i < possion_sample_count[0]:
            draw_sphere(possion_sample[i], ti.Vector([1.0,1.0,1.0]))

    for i in tri_vertices:
        draw_sphere(tri_vertices[i], ti.Vector([0.5,0.3,0.3]))


@ti.func
def get_cell_hash(a):
    p1 = 73856093 * a.x
    p2 = 19349663 * a.y
    p3 = 83492791 * a.z

    return p1^p2^p3

@ti.func
def compare_cell(i, j):

    ret = -1
    if (init_cell[i][0] > init_cell[j][0]):
        ret = 1
    elif (init_cell[i][0] == init_cell[j][0]) and (init_cell[i][1] > init_cell[j][1]):
        ret = 1
    elif (init_cell[i][0] == init_cell[j][0]) and (init_cell[i][1] == init_cell[j][1]) and (init_cell[i][2] > init_cell[j][2]):
        ret = 1
    elif (init_cell[i][0] == init_cell[j][0]) and (init_cell[i][1] == init_cell[j][1]) and (init_cell[i][2] == init_cell[j][2]):
        ret = 0

    return ret

@ti.func
def swap_cell(i, j):
    temp1 = init_cell[i]
    init_cell[i] = init_cell[j]
    init_cell[j] = temp1
    
    temp2 = init_pos[i]
    init_pos[i] =init_pos[j]
    init_pos[j] = temp2
    
    temp3 = init_id[i]
    init_id[i] =init_id[j]
    init_id[j] = temp3


@ti.kernel
def gpu_merge(j: ti.i32, k: ti.i32):
    for i in init_cell:
        ixj = i ^ j

        if ixj > i:
            if ((i & k) == 0):
                if compare_cell(i, ixj) == 1 :
                    swap_cell(i, ixj)

            if ((i & k) != 0) :
                if compare_cell(i, ixj) == -1:
                    swap_cell(i, ixj)



@ti.func
def check_cell_distance(neighbor_cell, cur_index):
    
    count = 0
    ret   = 0
    hash_neighbor = (  get_cell_hash(neighbor_cell)  % hash_map_size +  hash_map_size   )% hash_map_size

    while count < hMap.sample_count[hash_neighbor] and (ret == 0):
        neigh_cell_index = hMap.sample[hash_neighbor, count]

        cur_pos = init_pos[cur_index]
        neighbor_pos = init_pos[neigh_cell_index]

        cur_id = init_id[cur_index]
        neighbor_id = init_id[neigh_cell_index]

        dist = (cur_pos - neighbor_pos).norm()

        if cur_id != neighbor_id:
            v = (cur_pos - neighbor_pos).normalized()
            c1 = tri_normal[cur_id].dot(v)
            c2 = tri_normal[neighbor_id].dot(v)
            if (ti.abs(c1 - c2) > 0.00001):
                dist *= (ti.asin(c1) - ti.asin(c2)) / (c1 - c2)
            else:
                dist /= ti.sqrt(1.0 - c1*c1)
  
        if (dist < particleRadius):
            ret = 1
        count += 1

    return ret


@ti.func
def check_cell(cur_index): 
    
    ret = 0
    for i in range(-2,3):
        for j in range(-2,3):
            for k in range(-2,3):
                ret += check_cell_distance(ti.Vector([i, j, k]) + init_cell[cur_index], cur_index)
    return ret


@ti.kernel
def possion_disk_sample(pg: ti.i32, trial: ti.i32):
    for i in ti.ndrange(phase_group_count[pg]):
        cell = phase_group[pg, i]
        hash_index = (get_cell_hash(cell)  % hash_map_size +  hash_map_size)% hash_map_size
        
        start_index = hMap.start_index[hash_index]
        if start_index + trial < numInitialPoints:
            if compare_cell(start_index + trial , start_index ) == 0:
                if check_cell(start_index+trial) == 0:
     
                    old = ti.atomic_add(hMap.sample_count[hash_index], 1)
                    if old < hash_sample_size:
                        hMap.sample[hash_index, old] = start_index + trial
                    else:
                        hMap.sample_count[hash_index] = hash_sample_size-1
                        print("exceed hash sample!", old, hMap.start_index[hash_index])
                
                    old = ti.atomic_add(possion_sample_count[0], 1)
                    possion_sample[old] = init_pos[start_index + trial]

                        

loadObj("TAICHI.obj")
init_point_set()
gpu_bitonic_sort()
build_hmap()
detect_hmap()


possion_sample_count[0] = 0
trial_times = 0
phase_process = 0
trial_total = 10

while gui.running:

    if trial_times < trial_total:
        possion_disk_sample(phase_process, trial_times)
        print("possion sample trial:", trial_times, "phase:", phase_process, "point num:", possion_sample_count[0])
        ti.imwrite(img, str(trial_times*phase_block_size +phase_process)  + ".png")
        
    clear_canvas()
    draw_particle()
    
    gui.set_image(img.to_numpy())
    gui.show()

    if trial_times < trial_total:
        phase_process += 1
        if(phase_process%phase_block_size ==0):
            trial_times += 1
            phase_process = 0

    elif trial_times == trial_total:
        print("write obj")

        fo = open("boundry.obj", "w")
        pos = possion_sample.to_numpy()
        for i in range(possion_sample_count[0]):
            print("v", pos[i, 0], pos[i, 1], pos[i, 2], file = fo)
        fo.close()
        trial_times = 2*trial_times