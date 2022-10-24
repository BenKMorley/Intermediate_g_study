
import sys
import os
from multiprocessing import Pool

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

plt.rcParams.update({"text.usetex": True})

models = ['A', 'B', 'C', 'D']
Ns = [2, 3, 4, 5]
model_type = 1
char_limit = 100
run_params = []
parallel = True
params_to_use = ['alpha', 'beta', 'nu']
shifts = {'A': 0, 'B': 0.1, 'C': 0.2, 'D': 0.3}
colors_ = {'A': 'k', 'B': 'r', 'C': 'b', 'D': 'g'}

for N in Ns:
    for model in models:
        run_params.append((N, model))


def run(run_params):
    N, model = run_params
    a = analysis(N, model, model_type=model_type)
    a.fit_all_gLmin_all_Bbar()


if parallel:
    p = Pool(os.cpu_count())
    p.map(run, run_params)

else:
    for params in run_params:
        run(params)


for param in params_to_use:
    fig, ax = plt.subplots()

    for model in models:
        for N in Ns:
            a = analysis(N, model, model_type=model_type)
            a.fit_all_gLmin_all_Bbar()
            a.BMM_overall(perform_checks=True)
            string = ''
            ignore_comma = True

            if model == 'D' and param == 'alpha':
                param = 'alpha1'

            i = a.param_names.index(param)

            mean = a.mean[i]
            std = numpy.sqrt(a.var[i])

            if N == 2:
                ax.errorbar(N + shifts[model], mean, yerr=std, color=colors_[model],
                            label=f'model = {model}', marker='_')

            else:
                ax.errorbar(N + shifts[model], mean, yerr=std, color=colors_[model], marker='_')

    if param == 'alpha1':
        param = 'alpha'

    plt.title(rf'${param_names_latex[param]}$')
    plt.xlabel('N')
    plt.legend()
    plt.savefig(f'Thesis_results/Overall_estimates_{param}.pdf')
    plt.show()
