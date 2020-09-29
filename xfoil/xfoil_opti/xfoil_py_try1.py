"""
An example to show how to combine NURBS airfoil_generator and XFOIL
"""

from airfoil_generators.nurbs import NURBS 
from xfoil.xfoil import oper_visc_cl 

import matplotlib.pyplot as plt 
import numpy as np 
import os 
