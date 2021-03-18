import h5py
import re
import pdb
import numpy
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from publication_results import get_statistical_errors_central_fit
from model_definitions import K1, mPT_1loop
from Binderanalysis import Critical_analysis

# matplotlib.use('Qt5Agg')


def fig3_color(gL, min_gL=0.79, max_gL=76.81, func=numpy.log):
    top = numpy.array([0.5, 0, 0.7])
    bottom = numpy.array([1, 1, 0])

    gL_low = func(min_gL)
    gL_high = func(max_gL)

    gL_relative = (func(gL) - gL_low) / (gL_high - gL_low)

    try:
        assert (0 <= gL_relative) and (1 >= gL_relative), "Please use a gL in the range [gL_min, gL_max]"

    except:
        pdb.set_trace()

    color_value = tuple((gL_relative * top + (1 - gL_relative) * bottom))

    return color_value


def plot_Binder(N, g_s, L_s, data_file="MCMC_test.h5", minus_sign_override=False, legend=True, ax=None, min_gL=3.1, max_gL=76.81, reweight=True):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\frac{m^2 - m_c^2}{g^2} x^\frac{1}{\nu}$')
        # ax.set_xlabel(r'$\frac{m_c^2 - m^2}{m_c^2} x^\frac{1}{\nu}$')
        ax.set_ylabel(r'$B(N, g, L)$')

    markers = {8: 'd', 16: 'v', 32: '<', 48: '^', 64: 's', 96: 'o', 128:'d'}

    with h5py.File(data_file, "r") as f:
        # params = get_statistical_errors_central_fit(N)['params_central']
        # alpha = params[0]
        # beta = params[-2]
        # nu = params[-1]
        # nu = 0.65
        # beta = 1.21
        # pdb.set_trace()

        for g in g_s:
            # m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))
            for L in L_s:
                data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                masses = []
                Binders = []

                for string in data.keys():
                    try:
                        m = float(re.findall(r'-\d+\.\d+', string)[0])

                    except:
                        if minus_sign_override:
                            m = -float(re.findall(r'\d+\.\d+', string)[0])
                        else:
                            print("Mass not found")

                    masses.append(m)

                    if minus_sign_override: 
                        mass_data = data[f'msq={-m:.8f}']

                    else:
                        try:
                            mass_data = data[f'msq={m:.8f}']
                        except:
                            pdb.set_trace()

                    M2 = mass_data[0]
                    M4 = mass_data[1]
                    Binder = 1 - (N / 3) * numpy.mean(M4) / (numpy.mean(M2) ** 2)

                    pdb.set_trace()

                    Binders.append(Binder)

                masses = numpy.array(masses)

                ax.scatter(masses, Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors='none')
                # ax.scatter(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors='none')

                if reweight:
                  # Use Andreas's Critical Analysis Class to do the reweighting
                  min_m, max_m = min(masses), max(masses)

                  System = Critical_analysis(N, g, L)
                  System.h5load_data()
                  System.compute_tauint_and_bootstrap_indices()
                  System.Bbar = 0.5
                  L0bs = System.refresh_L0_bsindices()
                  L1bs = []

                  for j in range(len(System.actualm0sqlist)):
                      N_ = System.phi2[str(j)].shape[0]
                      L1bs.append(numpy.arange(int(numpy.floor(N_ / System.Nbin_tauint[j]))))

                  mass_range = numpy.linspace(min_m, max_m, 100)
                  results = []

                  # System.plot_tr_phi2_distributions()
                  # plt.show()

                  for i, m in tqdm(enumerate(mass_range)):
                      Binder_bit = System.reweight_Binder(m, L1bs, L0bs)

                      results.append(Binder_bit)

                  results = numpy.array(results)

                  plt.plot(((mass_range - m_crit) / g ** 2) * (g * L) ** (1 / nu), results + System.Bbar)
                  plt.plot(mass_range, results + System.Bbar)


    if legend:
        plt.legend()

    return ax
