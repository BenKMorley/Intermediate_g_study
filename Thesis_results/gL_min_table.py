from ast import arg
from ensurepip import bootstrap
import sys
import os
from multiprocessing import Pool

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

N_s = [2, 3, 4, 5]
models = ['A', 'B', 'C', 'D']
model_types = [1, 2]
run_params = []
parallel = True

for N in N_s:
    for model in models:
        for model_type in model_types:
            run_params.append((N, model, model_type))


def run(run_params):
    N, model, model_type = run_params
    a = analysis(N, model, model_type=model_type)
    a.fit_all_gLmin_all_Bbar(run_bootstrap=False)


if parallel:
    p = Pool(os.cpu_count())
    p.map(run, run_params)

else:
    for params in run_params:
        run(params)

print("\\begin{tabular}{|c|c|c|c|c|c|}")
print("\\hline\n$N$ & model & model type & $gL_{\\textrm{min}}$ & dof & AIC \\\\ \n \hline")

for N in N_s:
    for model in models:
        for model_type in model_types:
            a = analysis(N, model, model_type=model_type)
            a.fit_all_gLmin_all_Bbar(run_bootstrap=False)
            p_values_ = numpy.max(a.p_values, axis=0)
            max_dof = max(a.dofs)

            if sum(p_values_ > 0.05) > 0:
                arg_min = numpy.argmax(p_values_ > 0.05)
                B_arg_min = numpy.argmax(a.p_values[:, arg_min])

                gL_min = a.gL_mins[arg_min]
                dof = int(a.dofs[arg_min])

                # Caculate the AIC
                fitted = dof + len(a.param_names)
                ignored = max_dof - dof

                AIC = a.chisqs[B_arg_min, arg_min] + ignored * 2 + 2 * len(a.param_names)

                AIC = f'{AIC:.1f}'

            else:
                arg_min = -1
                gL_min = '-'
                dof = '-'
                AIC = '-'

            if model_type == 1:
                type_string = "$\\log(g)$"
            else:
                type_string = "$\\log(L)$"

            print(f'${N}$ & {model} & {type_string} & ${gL_min}$ & ${dof}$ & ${AIC}$\\\\')

print("\\hline")
print("\\end{tabular}")
