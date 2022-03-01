import taichi as ti
import math
import numpy as np
import taichi as ti

from kernels.CubicKernel import CubicKernel

MAX_VERTEX = 3000000

@ti.data_oriented
class MCGrid:
    def __init__(self, particleR, maxInGrid, maxNeighbour, particle_data):
        
        self.fps             = 20.0
        self.frame           = 0

        self.maxInGrid       = maxInGrid
        self.maxNeighbour    = maxNeighbour

        self.particle_data   = particle_data

        self.gridR           = particleR*0.9
        self.invGridR        = 1.0 / self.gridR
        
        self.searchR         = self.gridR  * 4.0
        self.grid_num        = 0
        self.isolevel        = 0.5

        self.kernel_c        = CubicKernel(self.searchR)

        self.gridCount       = ti.field(dtype=ti.i32)
        self.grid            = ti.field(dtype=ti.i32)


        self.surface_value   = ti.field( dtype=ti.f32)


        self.debug_value     = ti.field(dtype=ti.f32)
        self.vertlist        = ti.Vector.field(3, dtype=ti.f32)
        self.triangle        = ti.Vector.field(3, dtype=ti.f32)
        self.vertex_count    = ti.field(dtype=ti.i32, shape=(1))

        self.edgetable       = ti.field(dtype=ti.i32, shape=(256))
        self.tritable        = ti.field(dtype=ti.i32, shape=(256, 16))
        self.edgetablenp     = np.ones(shape=(256), dtype=np.int32)
        self.tritablenp      = np.ones(shape=(256,16), dtype=np.int32)

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.maxboundarynp   = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp   = np.ones(shape=(1,3), dtype=np.float32)
        self.blockSize       = ti.Vector.field(3, dtype=ti.i32, shape=(1))
        self.blocknp         = np.ones(shape=(1,3), dtype=np.int32)


    
    def setup_grid_gpu(self, maxboundarynp, minboundarynp):
        
        for i in range(3):
            self.maxboundarynp[0, i]    = maxboundarynp[0, i]
            self.minboundarynp[0, i]    = minboundarynp[0, i]

        for i in range(3):
            self.blocknp[0, i]    = int(    (self.maxboundarynp[0, i] - self.minboundarynp[0, i]) / self.gridR + 1  )
        self.grid_num = int(self.blocknp[0, 0]*self.blocknp[0, 1]*self.blocknp[0, 2])

        ti.root.dense(ti.i,   self.grid_num).place(self.gridCount)
        ti.root.dense(ti.ij, (self.grid_num, self.maxInGrid)).place(self.grid)
        
        
        #for surface restruction
        ti.root.dense(ti.i, self.grid_num ).place(self.surface_value)


        ti.root.dense(ti.ij, (self.grid_num, 12)).place(self.vertlist)
        ti.root.dense(ti.i,  MAX_VERTEX).place(self.triangle)
        ti.root.dense(ti.i,  self.grid_num,).place(self.debug_value)

    
    def setup_grid_cpu(self, maxboundarynp, minboundarynp):

        line_index = 0
        for line in open("MCdata.txt", "r"):

            if line_index < 32:
                values = line.split(',', 8)
                for i in range(len(values)-1):
                    self.edgetablenp[line_index*8+i]      = int(values[i],16)

            else:
                values = line.split(',', 16)
                for i in range(len(values)-1):
                    self.tritablenp[line_index-32, i] = int(values[i])
            line_index += 1

        self.max_boundary.from_numpy(self.maxboundarynp)
        self.min_boundary.from_numpy(self.minboundarynp)
        self.blockSize.from_numpy(self.blocknp)
        self.edgetable.from_numpy(self.edgetablenp)
        self.tritable.from_numpy(self.tritablenp)
        print("MC grid szie:", self.grid_num, "MC grid R:", self.gridR)



    
    def export_vertex(self):
        debug_value = self.debug_value.to_numpy()
        iso_value   = self.surface_value.to_numpy()

        filename = "out/" + str(self.frame)  + ".obj"
        fo = open(filename, "w")
        yz_dim  = self.blocknp[0, 1] * self.blocknp[0, 2]

        for i in range(self.grid_num):
            if iso_value[i] > 0.0:
                x = float(i // yz_dim)                        * self.gridR + self.minboundarynp[0, 0]
                y = float((i % yz_dim) // self.blocknp[0, 2]) * self.gridR + self.minboundarynp[0, 1]
                z = float(i  % self.blocknp[0, 2])            * self.gridR + self.minboundarynp[0, 2]
                print ("v %f %f %f %f %f %f" %  (x,y,z, iso_value[i], debug_value[i]/20.0, 1.0), file = fo)
        fo.close()

    
    def export_mesh(self):
        vertex_count = self.vertex_count.to_numpy()
        tri_vertex   = self.triangle.to_numpy()
        tri_count    = vertex_count[0] // 3
        filename = "out/mc_" + str(self.frame)  + ".obj"
        fo = open(filename, "w")

        for i in range(vertex_count[0]):
            print ("v %f %f %f" %  (tri_vertex[i, 0], tri_vertex[i, 1], tri_vertex[i, 2]), file = fo)

        for i in range(tri_count):
            print ("f %d %d %d" %  (3*i+1, 3*i+2, 3*i+3), file = fo)

        fo.close()

    
    def export_surface(self, time):
        time_i = int(time * self.fps )
        if int(time_i) == self.frame:

            self.update_grid()
            
            self.cal_surface_point()

            #self.particle_data.cal_anistropic_kernel()
            #self.cal_surface_point_anistropic()

            self.marching_cube()

            self.export_mesh()
            #self.export_vertex()
            
            self.frame+=1
            


    @ti.kernel
    def update_grid(self):
        for i,j in self.grid:
            self.grid[i,j] = -1
            self.gridCount[i]=0
            self.vertex_count[0] = 0

        #insert pos
        for i in self.particle_data.pos:
            indexV         = ti.cast((self.particle_data.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
            
            if self.check_in_box(indexV) == 1 :
                index     = indexV.x * self.blockSize[0].y*self.blockSize[0].z + indexV.y * self.blockSize[0].z + indexV.z
                
                old = ti.atomic_add(self.gridCount[index] , 1)
                if old > self.maxInGrid-1:
                    print("mc exceed grid ", old)
                    self.gridCount[index] = self.maxInGrid
                else:
                    self.grid[index, old] = i



    @ti.kernel
    def cal_surface_point(self):
        for i in self.surface_value:
            self.surface_value[i]  = 0.0
            indexCur       = self.get_cell_indexV(i)
            posi               = self.get_posV(indexCur)  

            valid_num = 0
            for m in range(-4,5):
                for n in range(-4,5):
                    for q in range(-4,5):
                        indexNeiV = indexCur +ti.Vector([m, n, q])
                        if self.check_in_box(indexNeiV)==1:  
                            indexNei = self.get_cell_index(indexNeiV)

                            k = 0
                            while k < self.gridCount[indexNei]:
                                
                                j     = self.grid[indexNei, k]
                                posj  = self.particle_data.pos[j] 
                                r     = posi - posj
                                W     = self.kernel_c.Cubic_W(r)

                                if (j < self.particle_data.liquid_count) and (W > 0.0):
                                    if (self.particle_data.rho[j] > self.particle_data.liqiudMass * self.kernel_c.Cubic_W_norm(0.0)):
                                        self.surface_value[i] += self.particle_data.liqiudMass / self.particle_data.rho[j] * W
                                        valid_num += 1
                                k += 1
        
        

    
    @ti.kernel
    def cal_surface_point_anistropic(self):
        
        for i in self.surface_value:
            self.surface_value[i]  = 0.0
            indexCur       = self.get_cell_indexV(i)
            posi           = self.get_posV(indexCur) 

            for m in range(-4,5):
                for n in range(-4,5):
                    for q in range(-4,5):
                        indexNeiV = indexCur +ti.Vector([m, n, q])
                        if (self.check_in_box(indexNeiV)==1):  
                            indexNei = self.get_cell_index(indexNeiV)

                            k = 0
                            while k < self.gridCount[indexNei]:
                                
                                j     = self.grid[indexNei, k]
                                posj  = 0.05*self.particle_data.pos[j]  + 0.95 * self.particle_data.pos_avr[j]
                                r     = posi - posj
                                
                                
                                G     = self.particle_data.G[j] 
                                Gr    = G @ r * 2.0
                                Gr    = Gr
                                W     = self.kernel_c.Cubic_W(Gr) 
                            

                                if (j < self.particle_data.liquid_count) and(W>0.0):
                                    if (self.particle_data.rho[j] >  self.particle_data.liqiudMass * self.kernel_c.Cubic_W_norm(0.0)):
                                        self.surface_value[i]        += self.particle_data.liqiudMass / self.particle_data.rho[j] * W 
                                k += 1





    @ti.kernel
    def marching_cube(self):



        for i in self.gridCount:

            indexCur  = self.get_cell_indexV(i)

            if self.check_in_box(indexCur+ti.Vector([1, 1, 1]))==1: 
                index0 = i
                index1 = self.get_cell_index(indexCur + ti.Vector([1, 0, 0]))
                index2 = self.get_cell_index(indexCur + ti.Vector([1, 1, 0]))
                index3 = self.get_cell_index(indexCur + ti.Vector([0, 1, 0]))
                index4 = self.get_cell_index(indexCur + ti.Vector([0, 0, 1]))
                index5 = self.get_cell_index(indexCur + ti.Vector([1, 0, 1]))
                index6 = self.get_cell_index(indexCur + ti.Vector([1, 1, 1]))
                index7 = self.get_cell_index(indexCur + ti.Vector([0, 1, 1]))

                cubeindex = 0
                if self.surface_value[index0] < self.isolevel:
                    cubeindex |= 1
                if self.surface_value[index1] < self.isolevel: 
                    cubeindex |= 2
                if self.surface_value[index2] < self.isolevel: 
                    cubeindex |= 4
                if self.surface_value[index3] < self.isolevel: 
                    cubeindex |= 8
                if self.surface_value[index4] < self.isolevel: 
                    cubeindex |= 16
                if self.surface_value[index5] < self.isolevel: 
                    cubeindex |= 32
                if self.surface_value[index6] < self.isolevel: 
                    cubeindex |= 64
                if self.surface_value[index7] < self.isolevel: 
                    cubeindex |= 128

                for j in range(12):
                    self.vertlist[i,j]  =  self.get_pos(index0)

                if self.edgetable[cubeindex] != 0:
                    if self.edgetable[cubeindex] & 1:
                        self.vertlist[i,0]  = self.vertex_interp(self.get_pos(index0), self.get_pos(index1), self.surface_value[index0], self.surface_value[index1])
                    if self.edgetable[cubeindex] & 2:    
                        self.vertlist[i,1]  = self.vertex_interp(self.get_pos(index1), self.get_pos(index2), self.surface_value[index1], self.surface_value[index2])
                    if self.edgetable[cubeindex] & 4:    
                        self.vertlist[i,2]  = self.vertex_interp(self.get_pos(index2), self.get_pos(index3), self.surface_value[index2], self.surface_value[index3])
                    if self.edgetable[cubeindex] & 8:    
                        self.vertlist[i,3]  = self.vertex_interp(self.get_pos(index3), self.get_pos(index0), self.surface_value[index3], self.surface_value[index0])
                    if self.edgetable[cubeindex] & 16:    
                        self.vertlist[i,4]  = self.vertex_interp(self.get_pos(index4), self.get_pos(index5), self.surface_value[index4], self.surface_value[index5])
                    if self.edgetable[cubeindex] & 32:    
                        self.vertlist[i,5]  = self.vertex_interp(self.get_pos(index5), self.get_pos(index6), self.surface_value[index5], self.surface_value[index6])
                    if self.edgetable[cubeindex] & 64:    
                        self.vertlist[i,6]  = self.vertex_interp(self.get_pos(index6), self.get_pos(index7), self.surface_value[index6], self.surface_value[index7])
                    if self.edgetable[cubeindex] & 128:    
                        self.vertlist[i,7]  = self.vertex_interp(self.get_pos(index7), self.get_pos(index4), self.surface_value[index7], self.surface_value[index4])
                    if self.edgetable[cubeindex] & 256:    
                        self.vertlist[i,8]  = self.vertex_interp(self.get_pos(index0), self.get_pos(index4), self.surface_value[index0], self.surface_value[index4])
                    if self.edgetable[cubeindex] & 512:    
                        self.vertlist[i,9]  = self.vertex_interp(self.get_pos(index1), self.get_pos(index5), self.surface_value[index1], self.surface_value[index5])
                    if self.edgetable[cubeindex] & 1024:    
                        self.vertlist[i,10] = self.vertex_interp(self.get_pos(index2), self.get_pos(index6), self.surface_value[index2], self.surface_value[index6])
                    if self.edgetable[cubeindex] & 2048:    
                        self.vertlist[i,11] = self.vertex_interp(self.get_pos(index3), self.get_pos(index7), self.surface_value[index3], self.surface_value[index7])
                
                
                k = 0
                while (self.tritable[cubeindex, k] != -1):
                    old = ti.atomic_add(self.vertex_count[0], 3)
                    if old < MAX_VERTEX:
                        self.triangle[old+0] = self.vertlist[i, self.tritable[cubeindex, k  ]]
                        self.triangle[old+1] = self.vertlist[i, self.tritable[cubeindex, k+1]]
                        self.triangle[old+2] = self.vertlist[i, self.tritable[cubeindex, k+2]]
                    else:
                        print("exceed max tri", old)
                    k += 3

    @ti.func
    def check_in_box(self, index):
        ret = 1
        if (index.x < 0) or (index.x >= self.blockSize[0].x) or \
           (index.y < 0) or (index.y >= self.blockSize[0].y) or \
           (index.z < 0) or (index.z >= self.blockSize[0].z):
            ret = 0
        return ret
    


    @ti.func
    def get_cell_indexV(self, index):
        yz_dim = self.blockSize[0].y * self.blockSize[0].z
        x      =  index // yz_dim
        y      = (index %  yz_dim) //self.blockSize[0].z
        z      = index  %  self.blockSize[0].z
        return ti.Vector([x, y, z])
    
    @ti.func
    def get_cell_index(self, index):
        return index.x * self.blockSize[0].y*self.blockSize[0].z + index.y * self.blockSize[0].z + index.z

    @ti.func
    def get_posV(self, indexV):
        return self.min_boundary[0] +  ti.cast(indexV, ti.f32) * self.gridR 

    @ti.func
    def get_pos(self, index):
        return self.get_posV(self.get_cell_indexV(index))


    @ti.func
    def weight_func(self, xi, xj):
        ret = 0.0
        r   = xi - xj
        dis = r.norm() 
        if dis < self.searchR*2.0:
            ret = 1.0 - pow( r.norm() / (self.searchR*2.0), 3.0)
        return ret



    @ti.func
    def vertex_interp(self, p1, p2, valp1, valp2):
        if  self.check_pos(p2, p1) == 1:
            temp = p1
            p1 = p2
            p2 = temp  

            tmp1 = valp1
            valp1 = valp2
            valp2 = tmp1

        p = p1
        if abs(valp1 - valp2) > 0.00001:
            p = p1 + (p2 - p1) / (valp2 - valp1)*(self.isolevel - valp1)

        return p

    @ti.func
    def check_pos(self, p2, p1):
        ret = 1
        if p2.x < p1.x:
            ret = 1
        elif p2.x > p1.x:
            ret = 0

        if p2.y < p1.y:
            ret = 1
        elif p2.y > p1.y:
            ret = 0

        if p2.z < p1.z:
            ret = 1
        elif p2.z > p1.z:
            ret = 0

        return ret



