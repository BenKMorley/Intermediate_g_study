import sys
import os
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
char_limit = 100
run_params = []
parallel = True

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


print('Keeping all fits')
print('_' * 100)
for N in Ns:
    for model in models:
        a = analysis(N, model, model_type=model_type)
        a.fit_all_gLmin_all_Bbar()
        a.BMM_overall()
        print(rf'{N} & {model} & ', end='')
        string = ''
        ignore_comma = True

        for i, param in enumerate(a.param_names):
            mean = a.mean[i]
            std = numpy.sqrt(a.var[i])

            if not ignore_comma:
                string += ', '

            string_piece = nice_string_print(mean, std)

            string += rf'${param_names_latex[param]}$: ${string_piece}$'

            if len(string) > char_limit:
                print(string + '\\\\')
                string = '& &'
                ignore_comma = True

            else:
                ignore_comma = False

        if len(string) > 3:
            print(string + '\\\\')


print('', end='\n\n\n')
print('Removing Insignificant Fits')
print('_' * 100)

for N in Ns:
    for model in models:
        a = analysis(N, model, model_type=model_type)
        a.fit_all_gLmin_all_Bbar()
        a.BMM_overall(perform_checks=True)
        print(rf'{N} & {model} & ', end='')
        string = ''
        ignore_comma = True

        for i, param in enumerate(a.param_names):
            mean = a.mean[i]
            std = numpy.sqrt(a.var[i])

            if not ignore_comma:
                string += ', '

            string_piece = nice_string_print(mean, std)

            string += rf'${param_names_latex[param]}$: ${string_piece}$'

            if len(string) > char_limit:
                print(string + '\\\\')
                string = '& &'
                ignore_comma = True

            else:
                ignore_comma = False

        if len(string) > 3:
            print(string + '\\\\')
