import taichi as ti
import math
import numpy as np
import taichi as ti
from HashGrid import HashGrid
from MarchingCubeGrid import MCGrid
from kernels.CubicKernel import CubicKernel


@ti.data_oriented
class ParticleData:
    def __init__(self, particleR):
        self.count              = 0
        self.liquid_count       = 0
        self.solid_count        = 0


        self.rho_L0             = 1000.0
        self.rho_S0             = self.rho_L0
        self.VL0                = particleR * particleR * particleR * 0.8 * 8.0
        self.VS0                = self.VL0 
        self.liqiudMass         = self.VL0 * self.rho_L0




        self.hash_grid          = HashGrid(particleR*2.0, 64, 2048, self)
        #self.mc_grid            = MCGrid(particleR*4.0, 64, 2048, self)
        self.mc_grid            = MCGrid(particleR, 4, 512, self)

        self.kernel_c           = CubicKernel(self.hash_grid.searchR)

        self.pos                = ti.Vector.field(3, dtype=ti.f32)
        self.normal             = ti.Vector.field(3, dtype=ti.f32)


     
        self.color              = ti.field( dtype=ti.f32)
        self.color_grad         = ti.Vector.field(3, dtype=ti.f32)
        
        self.pos_avr            = ti.Vector.field(3, dtype=ti.f32)
        self.G                  = ti.Matrix.field(3,3, dtype=ti.f32)



        self.vel_guess          = ti.Vector.field(3, dtype=ti.f32)
        self.vel                = ti.Vector.field(3, dtype=ti.f32)
        self.omega              = ti.Vector.field(3, dtype=ti.f32)
        self.vel_max            = ti.field(          dtype=ti.f32)
        self.d_vel              = ti.Vector.field(3, dtype=ti.f32)
        self.d_omega            = ti.Vector.field(3, dtype=ti.f32)


    
        self.pressure           = ti.field( dtype=ti.f32)
        self.rho                = ti.field( dtype=ti.f32)
        self.adv_rho            = ti.field( dtype=ti.f32)


        #viscorcity cg sovler
        self.gravity            = ti.Vector([0.0, -9.81, 0.0])
        self.dim_coff           = 10.0
        self.viscosity          = 10.0
        self.viscosity_b        = 10.0
        self.viscosity_err      = 0.05
         
        self.avg_density_err    = ti.field( dtype=ti.f32, shape=(1))
        self.cg_delta           = ti.field( dtype=ti.f32, shape=(1))
        self.cg_delta_old       = ti.field( dtype=ti.f32, shape=(1))
        self.cg_delta_zero      = ti.field( dtype=ti.f32, shape=(1))
         
        self.cg_Minv            = ti.Matrix.field(3, 3, dtype=ti.f32)
        self.cg_r               = ti.Vector.field(3,    dtype=ti.f32)
        self.cg_dir             = ti.Vector.field(3,    dtype=ti.f32)
        self.cg_Ad              = ti.Vector.field(3,    dtype=ti.f32)
        self.cg_s               = ti.Vector.field(3,    dtype=ti.f32)
        

        #tension
        self.tension_coff       = 0.0
        self.tension_coff_b     = 0.0
 
 
        #vorcity_coff 
        self.viscosity_omega    = 0.1
        self.vorticity_coff     = 0.01
        self.vorticity_init     = 0.5


        self.point_list = []
        
        self.maxboundarynp = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp = np.ones(shape=(1,3), dtype=np.float32)
        for j in range(3):
            self.maxboundarynp[0, j] = -10000.0
            self.minboundarynp[0, j] = 10000.0


    @ti.pyfunc
    def add_liquid_point(self, point):
        '''
        filename = "liqiud.obj"
        if len(self.point_list)  == 0:
            fo = open(filename, "w")
        else:
            fo = open(filename, "a+")
        print ("v %f %f %f" %  (point[0], point[1], point[2]), file = fo)
        fo.close()
        '''
        self.point_list.append(point)
        for j in range(3):
            self.maxboundarynp[0, j] = max(self.maxboundarynp[0, j], point[j])
            self.minboundarynp[0, j] = min(self.minboundarynp[0, j], point[j])
        
        
        self.count        += 1
        self.liquid_count += 1

    @ti.pyfunc
    def add_solid_point(self, point):
        self.point_list.append(point)
        for j in range(3):
            self.maxboundarynp[0, j] = max(self.maxboundarynp[0, j], point[j])
            self.minboundarynp[0, j] = min(self.minboundarynp[0, j], point[j])
        self.count        += 1
        self.solid_count += 1


    @ti.pyfunc
    def add_obj(self, filename):
        for line in open(filename, "r"):
            if line.startswith('#'): 
                continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                self.add_solid_point(v)


    @ti.pyfunc
    def setup_data_gpu(self):

        ti.root.dense(ti.i, self.count ).place(self.pos)

        ti.root.dense(ti.i, self.liquid_count ).place(self.normal)
        ti.root.dense(ti.i, self.liquid_count ).place(self.color)
        ti.root.dense(ti.i, self.liquid_count ).place(self.color_grad)

        ti.root.dense(ti.i, self.liquid_count ).place(self.pos_avr)
        ti.root.dense(ti.i, self.liquid_count ).place(self.G)



        ti.root.dense(ti.i, self.liquid_count ).place(self.vel_guess)
        ti.root.dense(ti.i, self.liquid_count ).place(self.vel      )
        ti.root.dense(ti.i, self.liquid_count ).place(self.omega    )
        ti.root.dense(ti.i, self.liquid_count ).place(self.vel_max  )
        ti.root.dense(ti.i, self.liquid_count ).place(self.d_vel    )
        ti.root.dense(ti.i, self.liquid_count ).place(self.d_omega  )


    
        ti.root.dense(ti.i, self.liquid_count ).place(self.pressure)
        ti.root.dense(ti.i, self.liquid_count ).place(self.rho     )
        ti.root.dense(ti.i, self.liquid_count ).place(self.adv_rho )


        #viscorcity cg sovler
        ti.root.dense(ti.i, self.liquid_count ).place(self.cg_Minv)
        ti.root.dense(ti.i, self.liquid_count ).place(self.cg_r   )
        ti.root.dense(ti.i, self.liquid_count ).place(self.cg_dir )
        ti.root.dense(ti.i, self.liquid_count ).place(self.cg_Ad  )
        ti.root.dense(ti.i, self.liquid_count ).place(self.cg_s   )
        
        self.hash_grid.setup_grid_gpu()
        self.mc_grid.setup_grid_gpu(self.maxboundarynp + self.mc_grid.searchR, self.minboundarynp- self.mc_grid.searchR)

    @ti.pyfunc
    def setup_data_cpu(self):

        self.pos.from_numpy(np.array(self.point_list, dtype = np.float32))
        self.hash_grid.setup_grid_cpu(self.maxboundarynp, self.minboundarynp)
        self.mc_grid.setup_grid_cpu(self.maxboundarynp + self.mc_grid.searchR, self.minboundarynp- self.mc_grid.searchR)
        print( "liqiud particle num:", self.liquid_count, "solid particle num:", self.solid_count)
    
    @ti.kernel
    def compute_color_map(self):
        for i in self.color:
            self.color[i]  = self.liqiudMass / self.rho[i] *  self.kernel_c.Cubic_W_norm(0.0)
            cur_neighbor     =  self.hash_grid.neighborCount[i]
            k=0
            while k < cur_neighbor:
                j = self.hash_grid.neighbor[i, k]
                r = self.pos[i] - self.pos[j]
                Wr = self.kernel_c.Cubic_W(r)

                if j < self.liquid_count:
                    self.color[i]     += self.liqiudMass / self.rho[j] * Wr  
                else:
                    self.color[i]     += self.VS0 * Wr  
                k += 1

        for i in self.color_grad:
            self.color_grad[i] = ti.Vector([0.0, 0.0, 0.0])
            cur_neighbor     =  self.hash_grid.neighborCount[i]
            k=0
            while k < cur_neighbor:
                j = self.hash_grid.neighbor[i, k]
                r = self.pos[i] - self.pos[j]

                if j < self.liquid_count:
                    self.color_grad[i]     +=  self.liqiudMass / self.rho[j] * self.color[j] * self.kernel_c.CubicGradW(r)
                k += 1
            self.color_grad[i] *= 1.0 / self.color[i]




    @ti.kernel
    def cal_anistropic_kernel(self):
        
        for i in self.pos_avr:
            sum_wij = 0.0
            sum_xj  = ti.Vector([0.0, 0.0, 0.0])
            self.pos_avr[i] = self.pos[i]
            

            cur_neighbor     =  self.hash_grid.neighborCount[i]
            k=0


            while k < cur_neighbor:
                j   = self.hash_grid.neighbor[i, k]
                if j < self.liquid_count:
                    wij = self.weight_func(self.pos[i] , self.pos[j]  )   
                    sum_wij += wij
                    sum_xj  += wij*self.pos[j]
                k+=1
            
            if sum_wij > 0.0:
                self.pos_avr[i] = sum_xj / sum_wij
         
        
        for i in self.G:
            kr = 4.0
            ks = 1400.0
            kn = 0.5
            ne = 25.0

            sum_wij = 0.0
            self.G[i]  = kn*ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            
            
            cur_neighbor     =  self.hash_grid.neighborCount[i]
            if cur_neighbor>ne:
                sum_ci  = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

                k=0
                while k < cur_neighbor:
                    j   = self.hash_grid.neighbor[i, k]
                    if j < self.liquid_count:
                        wij = self.weight_func(self.pos[i] , self.pos[j]  )  
                        r   = self.pos[j] - self.pos_avr[i]
                        sum_wij += wij
                        sum_ci  += wij * (r.outer_product(r))
                    k+=1

                C = sum_ci / sum_wij
                R, sigma, RT  = ti.svd(C, ti.f32)
                
                if sigma[0, 0] > 0.0:
                    sigmas        = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                    sigmas[0, 0]  = ks * sigma[0, 0]
                    sigmas[1, 1]  = ks * ti.max(sigma[1, 1], sigma[0, 0] / kr)
                    sigmas[2, 2]  = ks * ti.max(sigma[2, 2], sigma[0, 0] / kr)
                    inv_sigmas    = sigmas.inverse()
                    self.G[i]     = R @ inv_sigmas @ (RT.transpose())

                    '''
                    max_value     = ti.max( ti.max(inv_sigmas[2, 2], inv_sigmas[0, 0]),inv_sigmas[1, 1])
                    self.G[i]     = R @ inv_sigmas @ (RT.transpose()) / max_value
                    '''

            #self.color_grad[i] = ti.Vector([self.G[i][0,0], self.G[i][1,1], self.G[i][2,2]])
            #if abs(self.pos[i].x) < 0.05  and abs(self.pos[i].z) < 0.05  and ( (self.pos[i].y > 1.1) or (self.pos[i].y < 0.225)):
            #if abs(self.pos[i].z) > 0.45:
            #    print(1, self.pos[i], self.G[i] )
                    

    @ti.func
    def weight_func(self, xi, xj):
        ret = 0.0
        r   = xi - xj
        dis = r.norm() 
        if dis < self.mc_grid.searchR*2.0:
            ret = 1.0 - pow( dis / (self.mc_grid.searchR*2.0), 3.0)
        return ret


    @ti.pyfunc
    def export_kernel(self):

        pos   = self.pos.to_numpy()
        color = self.color_grad.to_numpy()

        filename = "out/test.obj"
        fo = open(filename, "w")
        for i in range(self.liquid_count):
            print ("v %f %f %f %f %f %f %f %f %f" %  (pos[i,0],pos[i,1],pos[i,2], color[i,0]*512.0,color[i,1]*512.0,color[i,2]*512.0,color[i,0],color[i,1],color[i,2]), file = fo)
        fo.close()



           


