from scipy.optimize import minimize_scalar
import numpy


Ns = numpy.array([2, 3, 4, 5])


def min_func_maker(x):
    def f(chi):
        expected = chi * Ns
        difference = x - expected

        return sum(difference ** 2)

    return f


results = {}
results['B1'] = [5, 7.2, 21.6, 27.2]
results['B2'] = [14.4, 17.6, 28.8, 35.2]
results['C1'] = [2.4, 2.8, 13.6, 11.2]
results['C2'] = [12, 12.8, 24.8, 21.6]

for key in results:
    func = min_func_maker(results[key])

    res = minimize_scalar(func)

    print(f'{key}: {res.x}')
