"""
Optimize the airfoil shape directly using genetic algorithm, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s):
    Viswanath, A., J. Forrester, A. I., Keane, A. J. (2011). Dimension Reduction for Aerodynamic Design Optimization.
    AIAA Journal, 49(6), 1256-1266.
    Grey, Z. J., Constantine, P. G. (2018). Active subspaces of airfoil shape parameterizations.
    AIAA Journal, 56(5), 2003-2017.
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from parsec.synthesis import synthesize
from utils import mean_err
from simulation import evaluate,compute_coeff




airfoil=np.loadtxt('naca0012.dat')
CL,CD=compute_coeff(airfoil, 10000, 0.0, 3, 100)
print (CL,CD)




