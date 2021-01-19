import h5py
import re
import pdb
import numpy
import matplotlib
import matplotlib.pyplot as plt

from publication_results import get_statistical_errors_central_fit
from model_definitions import K1, mPT_1loop

matplotlib.use('tkagg')


def plot_Binder(N, g_s, L_s):
    with h5py.File("MCMC_test.h5", "r") as f:
        for g in g_s:
            for L in L_s:
                params = get_statistical_errors_central_fit(2)['params_central']
                alpha = params[0]
                beta = params[-2]
                nu = params[-1]

                m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))

                data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                masses = []
                Binders = []

                for string in data.keys():
                    m = float(re.findall(r'-\d+\.\d+', string)[0])
                    # pdb.set_trace()
                    masses.append(m)

                    mass_data = data[f'msq={m:.8f}']

                    M2 = mass_data[0]
                    M4 = mass_data[1]
                    Binder = 1 - (N / 3) * numpy.mean(M4) / (numpy.mean(M2) ** 2)

                    Binders.append(Binder)

                masses = numpy.array(masses)

                plt.scatter(((m_crit - masses) / m_crit) * (g * L) ** (1 / nu), Binders)
                plt.xlabel(r'$\frac{m_c^2 - m^2}{m_c^2} x^\frac{1}{\nu}$')
                plt.ylabel(r'$B(N, g, L)$')
                plt.show()


g_s = [1]
L_s = [16]
N = 2

plot_Binder(N, g_s, L_s)
