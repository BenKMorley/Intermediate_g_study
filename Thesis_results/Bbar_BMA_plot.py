import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

plt.rcParams.update({"text.usetex": True})

# Parameters
N = 2
model = "B"
model_type = 1

a = analysis(N, model, model_type=model_type)
a.fit_all_gLmin_all_Bbar()
fig, ax = a.BMM_plot_overall(params=['nu'], return_ax=True)

fig.savefig('Thesis_results/BMA_plot_Bbar.pdf')
plt.show()
