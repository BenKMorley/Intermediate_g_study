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
        run(run_params)


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

            digit = floor(numpy.log10(abs(mean)))

            # We want the standard deviation rounded to 2 s.f.
            digit = floor(numpy.log10(abs(std)))
            std = numpy.round(std, -digit + 1)

            # Recaculate the digit for the edge case of a std that rounds up to 100
            digit = floor(numpy.log10(abs(std + sys.float_info.epsilon)))

            # Now round the mean to the same level as the standard deviation
            mean = numpy.round(mean, -digit + 1)

            # For the purpose of nice printing we want the std to print as a two digit number
            # unless it should contain a decimal point

            if mean == 0:
                digit2 = 0

            else:
                digit2 = floor(numpy.log10(abs(mean)))

            # We need to find the trailing zeros on the mean
            int_mean = mean * 10 ** (-digit + 1)
            factor = 10
            trailing_zeros = 0

            if digit != 0 and digit2 != 0:
                while(int_mean // factor != 0):
                    if int_mean % factor == 0:
                        factor *= 10
                        trailing_zeros += 1

                    else:
                        break

            if not ignore_comma:
                string += ', '

            if mean == 0:
                string += rf'${param_names_latex[param]}$: $0({std})$'

            elif digit == 0:
                string += rf'${param_names_latex[param]}$: ${mean}{"0" * trailing_zeros}({std:.1f})$'

            elif digit >= 1:
                string += rf'${param_names_latex[param]}$: ${int(mean)}({int(std)})$'

            else:
                std *= 10 ** (-digit + 1)
                string += rf'${param_names_latex[param]}$: ${mean}{"0" * trailing_zeros}({std:.0f})$'

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

            digit = floor(numpy.log10(abs(mean)))

            # We want the standard deviation rounded to 2 s.f.
            digit = floor(numpy.log10(abs(std)))
            std = numpy.round(std, -digit + 1)

            # Recaculate the digit for the edge case of a std that rounds up to 100
            digit = floor(numpy.log10(abs(std + sys.float_info.epsilon)))

            # Now round the mean to the same level as the standard deviation
            mean = numpy.round(mean, -digit + 1)

            # For the purpose of nice printing we want the std to print as a two digit number
            # unless it should contain a decimal point

            if mean == 0:
                digit2 = 0

            else:
                digit2 = floor(numpy.log10(abs(mean)))

            # We need to find the trailing zeros on the mean
            int_mean = mean * 10 ** (-digit + 1)
            factor = 10
            trailing_zeros = 0

            if digit != 0 and digit2 != 0:
                while(int_mean // factor != 0):
                    if int_mean % factor == 0:
                        factor *= 10
                        trailing_zeros += 1

                    else:
                        break

            if not ignore_comma:
                string += ', '

            if mean == 0:
                string += rf'${param_names_latex[param]}$: $0({std})$'

            elif digit == 0:
                string += rf'${param_names_latex[param]}$: ${mean}{"0" * trailing_zeros}({std:.1f})$'

            elif digit >= 1:
                string += rf'${param_names_latex[param]}$: ${int(mean)}({int(std)})$'

            else:
                std *= 10 ** (-digit + 1)
                string += rf'${param_names_latex[param]}$: ${mean}{"0" * trailing_zeros}({std:.0f})$'

            if len(string) > char_limit:
                print(string + '\\\\')
                string = '& &'
                ignore_comma = True

            else:
                ignore_comma = False

        if len(string) > 3:
            print(string + '\\\\')
