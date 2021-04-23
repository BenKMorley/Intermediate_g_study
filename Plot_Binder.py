import h5py
import re
import pdb
import numpy
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_definitions import mPT_1loop

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


def plot_Binder(N, g_s, L_s, data_file="MCMC_test.h5", form="g2", minus_sign_override=False, legend=True, ax=None, min_gL=3.1, max_gL=76.81, reweight=True, params=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\frac{m^2 - m_c^2}{g^2} x^\frac{1}{\nu}$')
        # ax.set_xlabel(r'$\frac{m_c^2 - m^2}{m_c^2} x^\frac{1}{\nu}$')
        ax.set_ylabel(r'$B(N, g, L)$')

    markers = {8: 'd', 16: 'v', 32: '<', 48: '^', 64: 's', 96: 'o', 128:'d'}

    with h5py.File(data_file, "r") as f:
        if params is None:
            params = get_statistical_errors_central_fit(N)['params_central']

        alpha = params[0]
        beta = params[-2]
        nu = params[-1]
        # nu = 0.65
        # beta = 1.21
        # pdb.set_trace()

        for g in g_s:
            m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))

            for L in L_s:
                if (g * L) > 12.7:
                    data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                    masses = []
                    Binders = []
                    Binder_sigmas = []

                    System = Critical_analysis(N, g, L)
                    System.MCMCdatafile = data_file
                    System.datadir = ''

                    System.h5load_data()

                    bootstrap_success = True
                    try:
                        System.compute_tauint_and_bootstrap_indices()
                    except Exception:
                        bootstrap_success = False

                    for i, string in enumerate(data.keys()):
                        # I'm going to assume the tauint is in the same order as
                        # the masses in data.keys since they're both from the same
                        # h5file accessed in the same way
                        if bootstrap_success:
                            bin_size = System.Nbin_tauint[i]

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

                        # Sort the data into bins
                        if bootstrap_success:
                            if len(M4) % bin_size != 0:
                                M4_binned = M4[:-(len(M4) % bin_size)]
                                M2_binned = M2[:-(len(M2) % bin_size)]
                            else:
                                M4_binned = M4
                                M2_binned = M2

                            no_bins = len(M4_binned) // bin_size

                            M4_binned = M4_binned.reshape((no_bins, bin_size))
                            M2_binned = M2_binned.reshape((no_bins, bin_size))

                            # Do a bootstrapping procedure
                            no_samples = 500
                            boot_indices = numpy.random.randint(no_bins, size=(no_samples, no_bins))
                            Binder_boot = numpy.zeros(no_samples)

                            for i in range(no_samples):
                                M4_sample = M4_binned[boot_indices[i]]
                                M2_sample = M2_binned[boot_indices[i]]
                                
                                try:
                                    Binder_boot[i] = 1 - (N / 3) * numpy.mean(M4_sample) / (numpy.mean(M2_sample) ** 2)
                                except:
                                    pdb.set_trace()

                            Binder_sigmas.append(numpy.std(Binder_boot))

                        Binders.append(Binder)

                    masses = numpy.array(masses)

                    if form == "g2":
                        if bootstrap_success:
                            ax.errorbar(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if form == "m2":
                        if bootstrap_success:
                            ax.errorbar(((m_crit - masses) / (m_crit * g)) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((m_crit - masses) / (m_crit * g)) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if form == "m4":
                        if bootstrap_success:
                            ax.errorbar(((m_crit - masses) / m_crit ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((m_crit - masses) / m_crit ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if form == "square_num_g2":
                        if bootstrap_success:
                            ax.errorbar(((m_crit ** 2 - masses ** 2) / g ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((m_crit ** 2 - masses ** 2) / g ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if form == "square_num_m4":
                        if bootstrap_success:
                            ax.errorbar(((m_crit ** 2 - masses ** 2) / m_crit ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((m_crit ** 2 - masses ** 2) / m_crit ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if form == "square_num_m2":
                        if bootstrap_success:
                            ax.errorbar(((m_crit ** 2 - masses ** 2) / m_crit) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((m_crit ** 2 - masses ** 2) / m_crit) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if form == "mPT":
                        if bootstrap_success:
                            ax.errorbar(((m_crit - masses) / mPT_1loop(g, N) ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((m_crit - masses) / mPT_1loop(g, N) ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

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

                        ax.plot(((mass_range - m_crit) / g ** 2) * (g * L) ** (1 / nu), results + System.Bbar)
                        ax.plot(mass_range, results + System.Bbar)

    if legend:
        plt.legend()

    return ax, [alpha, beta, nu, m_crit]
