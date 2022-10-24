import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

# Parameters
N = 2
model = "B"
model_type = 1
Bbar = 0.52

a = analysis(N, model, model_type=1)

a.fit_all_gLmin_all_Bbar()

fig, ax = plt.subplots()
ax = a.BMM_plot_gLmin(0.52, show=False, params=['nu'], ax_in=ax)

plt.savefig('Thesis_results/gL_min_BMA_plot.pdf')
plt.show()
