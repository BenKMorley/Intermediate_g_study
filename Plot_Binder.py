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
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\frac{m^2 - m_c^2}{g^2} x^\frac{1}{\nu}$')
    ax.set_ylabel(r'$B(N, g, L)$')
    markers = {16: 'v', 32: '<'}

    with h5py.File("MCMC_test.h5", "r") as f:
        for g in g_s:
            for L in L_s:
                params = get_statistical_errors_central_fit(N)['params_central']
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

                ax.scatter(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}')

    plt.legend()
    plt.show()


g_s = [1, 2, 4, 8, 16, 32]
L_s = [16]
N = 4

plot_Binder(N, g_s, L_s)