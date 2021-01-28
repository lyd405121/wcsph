import taichi as ti
import math
import numpy as np
import taichi as ti

#from ParticleData import ParticleData

@ti.data_oriented
class HashGrid:
    def __init__(self, gridR, maxInGrid, maxNeighbour, particle_data):

        self.maxInGrid       = maxInGrid
        self.maxNeighbour    = maxNeighbour

        self.particle_data   = particle_data
        self.invGridR        = 1.0 / gridR
        self.gridR           = gridR
        self.searchR         = gridR * 2.0


        self.gridCount       = ti.field(dtype=ti.i32)
        self.grid            = ti.field(dtype=ti.i32)
        self.neighborCount   = ti.field(dtype=ti.i32)
        self.maxCurNeighbour = ti.field(dtype=ti.i32)
        self.neighbor        = ti.field(dtype=ti.i32)

        self.blockSize       = ti.Vector.field(3, dtype=ti.i32, shape=(1))
        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))

   

    @ti.pyfunc
    def setup_grid_gpu(self):
        ti.root.dense(ti.i, self.particle_data.count).place(self.gridCount)
        ti.root.dense(ti.ij, (self.particle_data.count, self.maxInGrid)).place(self.grid)
        ti.root.dense(ti.i,  self.particle_data.count).place(self.maxCurNeighbour)

        ti.root.dense(ti.i, self.particle_data.liquid_count ).place(self.neighborCount)
        ti.root.dense(ti.ij, (self.particle_data.liquid_count , self.maxNeighbour)).place(self.neighbor)


    @ti.pyfunc
    def setup_grid_cpu(self, maxboundarynp, minboundarynp):
        
        blocknp   = np.ones(shape=(1,3), dtype=np.int32)
        for i in range(3):
            blocknp[0, i]    = int(    (maxboundarynp[0, i] - minboundarynp[0, i]) / self.gridR + 1  )

        self.max_boundary.from_numpy(maxboundarynp)
        self.min_boundary.from_numpy(minboundarynp)
        self.blockSize.from_numpy(blocknp)

        print("serach grid szie:", int(blocknp[0, 0]*blocknp[0, 1]*blocknp[0, 2]))

    @ti.kernel
    def update_grid(self):
        for i,j in self.grid:
            self.grid[i,j] = -1
            self.gridCount[i]=0
        
        for i,j in self.neighbor:
            self.neighbor[i,j]    = -1
            self.neighborCount[i] = 0

        #insert pos
        for i in self.particle_data.pos:
            indexV         = ti.cast((self.particle_data.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
            if self.check_in_box(indexV) == 1:
                hash_index     = self.get_cell_hash(indexV)
                old = ti.atomic_add(self.gridCount[hash_index] , 1)
                if old > self.maxInGrid-1:
                    print("exceed grid", old)
                    self.gridCount[hash_index] = self.maxInGrid
                else:
                    self.grid[hash_index, old] = i
        
        #find neighbour
        for i in self.neighborCount:
            indexV         = ti.cast((self.particle_data.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
            if self.check_in_box(indexV) == 1:
                for m in range(-2,3):
                    for n in range(-2,3):
                        for q in range(-2,3):
                            self.insert_neighbor(i, ti.Vector([m, n, q]) + indexV)


    @ti.func
    def insert_neighbor(self, i, index_neigh):
        if index_neigh.x >= 0 and index_neigh.x < self.blockSize[0].x and \
        index_neigh.y >= 0 and index_neigh.y < self.blockSize[0].y  and \
        index_neigh.z >= 0 and index_neigh.z < self.blockSize[0].z :

            hash_index = self.get_cell_hash(index_neigh)
            k=0
            while k < self.gridCount[hash_index]:
                j = self.grid[hash_index, k]
                if j >= 0 and (i != j):
                    r = self.particle_data.pos[i] - self.particle_data.pos[j]
                    old = ti.atomic_add(self.neighborCount[i] , 1)
                    if old > self.maxNeighbour-1:
                        old = old
                        print("exceed neighbor", old)
                    else:
                        self.neighbor[i, old] = j
                k += 1

    @ti.func
    def get_cell_hash(self, a):
        p1 = 73856093 * a.x
        p2 = 19349663 * a.y
        p3 = 83492791 * a.z

        return ((p1^p2^p3) % self.particle_data.count  +  self.particle_data.count ) % self.particle_data.count  
    

    @ti.func
    def check_in_box(self, index):
        ret = 1
        if (index.x < 0) or (index.x >= self.blockSize[0].x) or \
           (index.y < 0) or (index.y >= self.blockSize[0].y) or \
           (index.z < 0) or (index.z >= self.blockSize[0].z):
            ret = 0
        return ret


    @ti.kernel
    def process_neighbour(self, index: ti.i32):
        for i in self.maxCurNeighbour:
            if i % index == 0:
                offcet = int(index / 2)

                indexi = self.gridCount[i]
                indexj = self.gridCount[i+offcet]

                #indexi = self.neighborCount[i]
                #indexj = self.neighborCount[i+offcet]

                if indexi > indexj:
                    self.maxCurNeighbour[i] = indexi
                else:
                    self.maxCurNeighbour[i] = indexj


    @ti.pyfunc
    def get_max_neighbour(self):
        size = 2
        while size < self.particle_data.liquid_count:
            self.process_neighbour(size)
            size = size*2

        print(self.maxCurNeighbour.to_numpy()[0])

