import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *


# log(g) and log(L) scaling graphs
N = 3
model_type = 3
model = "GB"
run_params = []
parallel = True

a = analysis(N, model_name=model, model_type=model_type, no_samples=50)
a.fit_all_gLmin_all_Bbar()

fig, ax = plt.subplots()

a.BMM_overall()
a.BMM_plot_overall_gL(combine=True, params=['beta1', 'beta2'], colors=['green', 'purple'],
                      ax_in=ax)
plt.title(rf'N = {N}, model=${model[-1]}_3$')
plt.show()
