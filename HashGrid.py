import taichi as ti
import math
import numpy as np
import taichi as ti


maxNeighbour = 100

@ti.data_oriented
class HashGrid:
    def __init__(self, n, num_l, maxb, minb, gridR):
        self.count        = n
        self.liquid_count = num_l
        self.invGridR = 1.0 / gridR
        self.searchR = gridR * 2.0

        self.gridCount = ti.field(dtype=ti.i32)
        self.grid       = ti.field(dtype=ti.i32)
        self.neighborCount = ti.field(dtype=ti.i32)
        self.neighbor = ti.field(dtype=ti.i32)
        self.pos         = ti.Vector.field(3, dtype=ti.f32)

        self.blockSize  = ti.Vector.field(3, dtype=ti.i32, shape=(1))
        self.min_boundary   = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary   = ti.Vector.field(3, dtype=ti.f32, shape=(1))


        ti.root.dense(ti.i, self.count).place(self.gridCount)
        ti.root.dense(ti.ij, (self.count, maxNeighbour)).place(self.grid)

        ti.root.dense(ti.i, self.liquid_count ).place(self.neighborCount)
        ti.root.dense(ti.ij, (self.liquid_count , maxNeighbour)).place(self.neighbor)

        ti.root.dense(ti.i, self.count ).place(self.pos)

        self.blocknp   = np.ones(shape=(1,3), dtype=np.int32)
        for i in range(3):
            self.blocknp[0, i]    = int((maxb[0, i] - minb[0, i]) / gridR + 1)

        self.gridSize     = int(self.blocknp[0, 0]*self.blocknp[0, 1]*self.blocknp[0, 2])
        self.max_boundary.from_numpy(maxb)
        self.min_boundary.from_numpy(minb)
        self.blockSize.from_numpy(self.blocknp)



    
    @ti.kernel
    def update_grid(self):
        for i,j in self.grid:
            self.grid[i,j] = -1
            self.gridCount[i]=0
        
        for i,j in self.neighbor:
            self.neighbor[i,j]    = -1
            self.neighborCount[i] = 0

        #insert pos
        for i in self.pos:
            indexV         = ti.cast((self.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
            if self.check_in_box(indexV) == 1:
                hash_index     = self.get_cell_hash(indexV)
                old = ti.atomic_add(self.gridCount[hash_index] , 1)
                if old > maxNeighbour-1:
                    print("exceed grid", old)
                    self.gridCount[hash_index] = maxNeighbour
                else:
                    self.grid[hash_index, old] = i
        
        #find neighbour
        for i in self.neighborCount:
            indexV         = ti.cast((self.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
            if self.check_in_box(indexV) == 1:
                for m in range(-1,2):
                    for n in range(-1,2):
                        for q in range(-1,2):
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
                    r = self.pos[i] - self.pos[j]
                    r_mod = r.norm()
                    if r_mod < self.searchR:
                        old = ti.atomic_add(self.neighborCount[i] , 1)
                        if old > maxNeighbour-1:
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

        return ((p1^p2^p3) % self.count  +  self.count ) % self.count  
    

    @ti.func
    def check_in_box(self, index):
        ret = 1
        if (index.x < 0) or (index.x >= self.blockSize[0].x) or \
           (index.y < 0) or (index.y >= self.blockSize[0].y) or \
           (index.z < 0) or (index.z >= self.blockSize[0].z):
            ret = 0
        return ret
