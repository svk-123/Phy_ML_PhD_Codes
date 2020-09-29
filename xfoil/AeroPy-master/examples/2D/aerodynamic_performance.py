from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import aeropy.xfoil_module as xf
from aeropy.aero_module import Reynolds
from aeropy.geometry.airfoil import CST, create_x

af=np.loadtxt('2032c')
airfoil='2032c'

ab=xf.find_coefficients('af',alpha=[0,2,4,6],Reynolds=10000,iteration=100,\
                        NACA=False, delete=False, PANE=False, GDES=False)

#ab=xf.file_name(airfoil, alfas=3, output='Cp')