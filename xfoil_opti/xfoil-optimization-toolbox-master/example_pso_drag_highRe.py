"""
Test of Particle Swarm Optimization algorithm in combination with Xfoil and
the PARSEC airfoil parametrization. Trying to find high Re low drag airfoil.
"""

from __future__ import division, print_function
from os import remove
import numpy as np
from copy import copy
from string import ascii_uppercase
from random import choice
import matplotlib.pyplot as plt
from optimization_algorithms.pso import Particle
from airfoil_generators import parsec
from xfoil import xfoil

Re = 1E6
constraints = np.array((
#rle        x_pre/suc    d2ydx2_pre/suc  th_pre/suc
(.015,.05), (.3,.75),     (-2,.2),          (0,40)
))
# Good parameters at:
# http://hvass-labs.org/people/magnus/publications/pedersen10good-pso.pdf
iterations, S, omega, theta_g, theta_p = 12, 12, -0.2, 2.8, 0

def construct_airfoil(*pts):
    k = {}
    k['rle'] = pts[0]
    k['x_pre'] = pts[1]
    # Thickness 21%
    k['y_pre'] = -.105
    k['d2ydx2_pre'] = -pts[2]
    # Trailing edge angle
    k['th_pre'] = pts[3]
    # Suction part
    k['x_suc'] = k['x_pre']
    k['y_suc'] = -k['y_pre']
    k['d2ydx2_suc'] = -k['d2ydx2_pre']
    k['th_suc'] = -k['th_pre']
    # Trailing edge x and y position
    k['xte'] = 1
    k['yte'] = 0
    return parsec.PARSEC(k)

def score_airfoil(airfoil):    
    # Make unique filename
    randstr = ''.join(choice(ascii_uppercase) for i in range(20))
    filename = "parsec_{}.dat".format(randstr)
    # Save coordinates
    with open(filename, 'w') as af:
        af.write(airfoil.get_coords_plain())
    # Let Xfoil do its magic
    polar = xfoil.oper_visc_alpha(filename, 0, Re,
                                  iterlim=80, show_seconds=0)
    try:
        remove(filename)
    except WindowsError:
        print("\n\n\n\nWindows was not capable of removing the file.\n\n\n\n")

    try:
        score = polar[0][0][2]
        print("Score: ", score)
        # If it's not NaN
        if np.isfinite(score):
            print("Return score")
            return score
        else:
            print("Return None")
            return None
    except IndexError:
        print("Return None (IndexError)")
        return None

# Show plot and make redrawing possible
fig, (cur_afplt, lastpbest_afplt, gbest_afplt, score_plt) = plt.subplots(4,1)
# Enable auto-clearing
cur_afplt.hold(False)
lastpbest_afplt.hold(False)
gbest_afplt.hold(False)
plt.tight_layout()
# Interactive mode
plt.ion()
#plt.pause(.0001)

# Initialize globals
global_bestscore   = None
global_bestpos     = None
global_bestairfoil = None

# Constructing a particle automatically initializes position and speed
particles = [Particle(constraints) for i in xrange(0, S)]

scores_y = []

for n in xrange(iterations+1):
    print("\nIteration {}".format(n))
    for i_par, particle in enumerate(particles):
        # Keep scoring until converged
        score = None
        while not score:
            # Update particle's velocity and position, if global best
            if global_bestscore:
                print("Update particle")
                particle.update(global_bestpos, omega, theta_p, theta_g)
            # None if not converged
            airfoil = construct_airfoil(*particle.pts)
            score = score_airfoil(airfoil)
            plotstyle = "{}-".format(choice("rgb"))
            airfoil.plot(cur_afplt, score="Cd {}".format(score), style=plotstyle,
                         title="Current, particle n{}p{}".format(n, i_par))
            #plt.pause(.0001)
            if not score and (not global_bestscore or n==0):
                print("Not converged, no global best, or first round. Randomizing particle.")
                particle.randomize()
            elif not score:
                print("Not converged, there is a global best. Randomizing.")
                particle.randomize()

        if not particle.bestscore or score < particle.bestscore:
            particle.new_best(score)
            txt = 'particle best'
            airfoil.plot(lastpbest_afplt, score="Cd {}".format(score), style=plotstyle,
            title="Particle best, particle n{}p{}".format(n, i_par))
            #plt.pause(.0001)
            print("Found particle best, score {}".format(score))
        if not global_bestscore or score < global_bestscore:
            global_bestscore = score
            # Copy to avoid globaL_bestpos becoming reference to array
            global_bestpos = copy(particle.pts)
            txt = 'global best'
            airfoil.plot(gbest_afplt, score="Cd {}".format(score), style=plotstyle,
              title="Global best, particle n{}p{}".format(n, i_par))
            #plt.pause(.0001)
            print("Found global best, score {}".format(score))
            global_bestairfoil = airfoil
        
    scores_y.append(global_bestscore)
    score_plt.plot(scores_y, 'r-')
    score_plt.set_title("Global best per round")
    plt.pause(.0001)


print("Best airfoil found for Re={}, ".format(Re),
      "score = ", global_bestscore,
      ", pos = ", global_bestpos.__repr__(),
      ", airfoil points:\n{}".format(airfoil.get_coords_plain()))

plt.show()
