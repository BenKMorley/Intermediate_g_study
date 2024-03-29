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
from Core.MISC import nice_string_print


models = ['A', 'B', 'C', 'D']
Ns = [2, 3, 4, 5]
model_type = 1
run_params = []
parallel = False
g = 0.1  # Show the g = 0.1 data

for N in Ns:
    for model in models:
        run_params.append((N, model))


def run(params):
    N, model = params
    a = analysis(N, model, model_type=model_type)
    a.fit_all_gLmin_all_Bbar()


if parallel:
    p = Pool(os.cpu_count())
    p.map(run, run_params)

else:
    for params in run_params:
        run(params)


print('Keeping all fits')
print('_' * 100)

print(rf'N & model & $m_c^2(g=0.1)$\\')

for N in Ns:
    for model in models:
        a = analysis(N, model, model_type=model_type)
        a.fit_all_gLmin_all_Bbar()
        a.BMM_overall()
        print(rf'{N} & {model} & ', end='')

        string_piece = nice_string_print(a.mass_mean[g], numpy.sqrt(a.mass_var[g]))

        print(rf'${string_piece}$ \\')

print('', end='\n\n\n')
print('Removing Insignificant Fits')
print('_' * 100)

print(rf'N & model & $m_c^2(g=0.1)$\\')

for N in Ns:
    for model in models:
        a = analysis(N, model, model_type=model_type)
        a.fit_all_gLmin_all_Bbar()
        a.BMM_overall(perform_checks=True)
        print(rf'{N} & {model} & ', end='')

        string_piece = nice_string_print(a.mass_mean[g], numpy.sqrt(a.mass_var[g]))

        print(rf'$m_c^2(g=0.1)$: ${string_piece}$ \\')
