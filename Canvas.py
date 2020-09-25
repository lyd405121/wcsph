import taichi as ti
import math
import numpy as np
import taichi as ti

@ti.data_oriented
class Canvas:
    def __init__(self, sizex, sizey):

        self.view = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1))
        self.proj = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1))

        self.screenRes = ti.Vector([sizex, sizey])
        self.img = ti.Vector.field(3, dtype=ti.f32, shape=[sizex, sizey])
        self.depth = ti.field(dtype=ti.f32, shape=[sizex, sizey])

        self.eye        = np.array([0.0, 0.0, 1.0])
        self.target     = np.array([0.0, 0.0, 0.0])
        self.up         = np.array([0.0, 1.0, 0.0])

        self.ratio      = sizex / sizey
        self.yaw        = 0.0
        self.pitch      = 0.3
        self.roll       = 0.0
        self.scale      = 1.0

        self.fov        = 1.0
        self.near       = 1.0
        self.far        = 1000.0
        
        self.ortho = 0

    @ti.kernel
    def clear_canvas(self):
        for i, j in self.img:
            self.img[i, j]=ti.Vector([0, 0, 0])
            self.depth[i, j] = 1.0

    @ti.pyfunc
    def update_cam(self):
        
        self.pitch = min(self.pitch, 1.57)
        self.pitch = max(self.pitch, -1.57)
        self.eye[0] = self.target[0] + self.scale * math.cos(self.pitch) * math.sin(self.yaw)
        self.eye[1] = self.target[1] + self.scale * math.sin(self.pitch)
        self.eye[2] = self.target[2] + self.scale * math.cos(self.pitch) * math.cos(self.yaw)
        self.up[0]  = -math.sin(self.pitch) * math.sin(self.yaw)
        self.up[1]  = math.cos(self.pitch)
        self.up[2]  = -math.sin(self.pitch) * math.cos(self.yaw)

        zaxis = self.eye - self.target
        zaxis = zaxis / np.linalg.norm(zaxis) 
        xaxis = np.cross(self.up, zaxis)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        yScale = 1.0    / ti.tan(self.fov/2.0)
        xScale = yScale / self.ratio


        view_np = self.view.to_numpy()
        view_np[0] = ti.np.array([ [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, self.eye)], \
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis,self.eye)], \
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis,self.eye)], [0.0, 0.0, 0.0, 1.0] ])
        self.view.from_numpy(view_np)


        proj_np = self.proj.to_numpy()
        
        if self.ortho == 0 :
            proj_np[0] =  ti.np.array([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, self.far/(self.near-self.far), self.near*self.far/(self.near-self.far)], [0.0, 0.0, -1.0, 0.0] ])
        else:
            proj_np[0] =  ti.np.array([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, 1.0/(self.near-self.far), self.near/(self.near-self.far)], [0.0, 0.0, 0.0, 1.0] ])
        self.proj.from_numpy(proj_np)


    @ti.pyfunc
    def set_view_point(self, yaw, pitch, roll,scale):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.scale = scale
        self.update_cam()

    @ti.pyfunc
    def set_fov(self, fov):
        self.fov = fov
        self.update_cam()

    @ti.pyfunc
    def set_target(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        self.update_cam()

    @ti.func
    def transform(self, v):
        screenP  = self.proj[0] @ self.view[0] @ ti.Vector([v.x, v.y, v.z, 1.0])
        screenP /= screenP.w
        return ti.Vector([(screenP.x+1.0)*0.5*self.screenRes.x, (screenP.y+1.0)*0.5*self.screenRes.y, screenP.z])

    @ti.func
    def fill_pixel(self, v, z, c):
        if (v.x >= 0) and  (v.x <self.screenRes.x) and (v.y >=0 ) and  (v.y < self.screenRes.y):
            if self.depth[v] > z:
                self.img[v] = c
                self.depth[v] = z

    @ti.func
    def draw_sphere(self, v, c):

        v  = self.transform(v)
        xc = ti.cast(v.x, ti.i32)
        yc = ti.cast(v.y, ti.i32)

        r=3
        x=0
        y = r
        d = 3 - 2 * r

        while x<=y:
            self.fill_pixel(ti.Vector([ xc + x, yc + y]), v.z, c)
            self.fill_pixel(ti.Vector([ xc - x, yc + y]), v.z, c)
            self.fill_pixel(ti.Vector([ xc + x, yc - y]), v.z, c)
            self.fill_pixel(ti.Vector([ xc - x, yc - y]), v.z, c)
            self.fill_pixel(ti.Vector([ xc + y, yc + x]), v.z, c)
            self.fill_pixel(ti.Vector([ xc - y, yc + x]), v.z, c)
            self.fill_pixel(ti.Vector([ xc + y, yc - x]), v.z, c)
            self.fill_pixel(ti.Vector([ xc - y, yc - x]), v.z, c)


            if d<0:
                d = d + 4 * x + 6
            else:
                d = d + 4 * (x - y) + 10
                y = y-1
            x +=1


    @ti.func
    def draw_point(self, v,c):
        v = self.transform(v)
        Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])
        self.fill_pixel(Centre, v.z, c)

    @ti.func
    def draw_solid_sphere(self, v, c):
        v = self.transform(v)
        r = 4
        Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])

        for i in range(-r, r+1):
            for j in range(-r, r+1):
                dis = i*i + j*j
                if (dis < r*r):
                    self.fill_pixel(Centre+ti.Vector([i,j]), v.z, c)


    @ti.func
    def draw_point(self, v, c):
        v = self.transform(v)
        Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])
        self.fill_pixel(Centre, v.z, c)