import sys
import os
import time
import math
import numpy as np
import taichi as ti



@ti.data_oriented
class CohesionKernel:
    def __init__(self, searchR):
        self.searchR        = searchR
        self.m_k   = 32.0  / (math.pi * math.pow(searchR, 9.0))
        self.m_c   = math.pow(searchR, 6.0) / 64.0

    @ti.func
    def Cubic_W_norm(self, r):
        res = 0.0
        radius2 = self.searchR*self.searchR
        r2 = r*r

        if r2 <= radius2:
            r3 = r2*r
            if r > 0.5 * self.searchR:
                res = self.m_k*pow(self.searchR-r, 3.0) * r3
            else:
                res = self.m_k*2.0 * pow(self.searchR-r, 3.0) * r3 - self.m_c
        return res

    @ti.func
    def Cubic_W(self, r):
        return self.Cubic_W_norm(r.norm())

    


