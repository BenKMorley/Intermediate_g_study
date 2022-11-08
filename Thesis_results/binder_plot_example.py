import sys
import os
from math import floor
from multiprocessing import Pool


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *
from Local.Plot_Binder import plot_Binder

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

model_type = 1
const_Bbar = True
show_plot = False
scale_with_fit = True
w = 0.5
crossings_file = f"width/width_{w:.1f}.h5"
N = 5
g = 0.6
L = 16

fig, ax = plt.subplots()

ax = plot_Binder(N, [g], [L], reweight=False, crossings_file=crossings_file, ax=ax,
                 legend=False)

plt.title(rf'$N = {N}$, $g = {g}$, $L = {L}$')
plt.xlabel(r'$m^2$')
plt.ylabel('$B$', rotation=0)
plt.savefig('Thesis_results/binder_plot_example.pdf')
plt.show()
