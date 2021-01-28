import sys
import os
import time
import math
import numpy as np
import taichi as ti



@ti.data_oriented
class CubicKernel:
    def __init__(self, searchR):
        self.searchR        = searchR
        self.h3             = 1.0 / (searchR*searchR*searchR)
        self.m_k            = 8.0  / (math.pi)
        self.m_l            = 48.0 / (math.pi)



    @ti.func
    def CubicGradW(self, r):
        res = ti.Vector([0.0, 0.0, 0.0])
        rl = r.norm()
        q = rl / self.searchR
        if ((rl > 1.0e-5) and (q <= 1.0)):
        	gradq = r / ( rl*self.searchR)
        	if (q <= 0.5):
        		res = self.m_l*self.h3*q*(3.0*q - 2.0)*gradq
        	else:
        		factor = 1.0 - q
        		res = -self.m_l*self.h3*(factor*factor)*gradq
        return res


    @ti.func
    def Cubic_W_norm(self, v):
        return self.Cubic_W_P(v / self.searchR) * self.m_k*self.h3

    @ti.func
    def Cubic_W(self, v):
        return self.Cubic_W_norm(v.norm())

    @ti.func
    def Cubic_W_P(self, q):
        res = 0.0
        if q <= 1.0:
            if (q <= 0.5):
                qq = q*q
                qqq = qq*q
                res = 6.0*qqq - 6.0*qq+1.0
            else:
                factor = 1.0 - q
                res = 2.0*factor*factor*factor
        return res


    


