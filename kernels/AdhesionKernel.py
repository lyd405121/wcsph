import sys
import os
import time
import math
import numpy as np
import taichi as ti



@ti.data_oriented
class AdhesionKernel:
    def __init__(self, searchR):
        self.searchR        = searchR
        self.m_k   = 0.007  / math.pow(searchR, 3.25)





    @ti.func
    def Cubic_W_norm(self, r):
        res = 0.0
        radius2 = self.searchR*self.searchR
        r2 = r*r

        if r2 <= radius2:
            if r > 0.5 * self.searchR:
                res = self.m_k*pow(-4.0*r2 / self.searchR + 6.0*r - 2.0*self.searchR, 0.25)
        return res

    @ti.func
    def Cubic_W(self, r):
        return self.Cubic_W_norm(r.norm())

    


