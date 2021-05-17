import h5py
import re
import pdb
import numpy
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')


from Core.model_definitions import mPT_1loop, K1
from Core.publication_results import get_statistical_errors_central_fit
from Core.Binderanalysis import Critical_analysis

# matplotlib.use('Qt5Agg')


def fig3_color(gL, min_gL=0.79, max_gL=76.81, func=numpy.log):
    top = numpy.array([0.5, 0, 0.7])
    bottom = numpy.array([1, 1, 0])

    gL_low = func(min_gL)
    gL_high = func(max_gL)

    gL_relative = (func(gL) - gL_low) / (gL_high - gL_low)

    try:
        assert (0 <= gL_relative) and (1 >= gL_relative), "Please use a gL in the range [gL_min, gL_max]"

    except Exception:
        pdb.set_trace()

    color_value = tuple((gL_relative * top + (1 - gL_relative) * bottom))

    return color_value


def plot_Binder(N, g_s, L_s, data_file="../h5data/MCMC_data_full.h5", minus_sign_override=False, legend=True, ax=None, min_gL=3.1, max_gL=76.81, reweight=True, params=None, GL_lim=12.7):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\frac{m^2 - m_c^2}{g^2} x^\frac{1}{\nu}$')
        # ax.set_xlabel(r'$\frac{m_c^2 - m^2}{m_c^2} x^\frac{1}{\nu}$')
        ax.set_ylabel(r'$B(N, g, L)$')

    markers = {8: 'd', 16: 'v', 32: '<', 48: '^', 64: 's', 96: 'o', 128: 'd'}

    with h5py.File(data_file, "r") as f:
        critical_found = True

        if params is None:
            try:
                params = get_statistical_errors_central_fit(N)['params_central']

                alpha = params[0]
                beta = params[-2]
                nu = params[-1]

            except Exception:
                critical_found = False

        for g in g_s:
            try:
                m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))

            except Exception:
                critical_found = False

            for L in L_s:
                if (g * L) > GL_lim:
                    data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                    masses = []
                    Binders = []
                    Binder_sigmas = []

                    System = Critical_analysis(N, g, L)
                    System.MCMCdatafile = data_file
                    System.datadir = ''

                    System.h5load_data()

                    bootstrap_success = True

                    # try:
                    System.compute_tauint_and_bootstrap_indices()

                    # except Exception:
                    #     bootstrap_success = False

                    for i, string in enumerate(data.keys()):
                        # I'm going to assume the tauint is in the same order as
                        # the masses in data.keys since they're both from the same
                        # h5file accessed in the same way
                        if bootstrap_success:
                            bin_size = System.Nbin_tauint[i]

                        # pdb.set_trace()

                        try:
                            m = float(re.findall(r'-\d+\.\d+', string)[0])

                        except Exception:
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

                            except Exception:
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

                                except Exception:
                                    pdb.set_trace()

                            Binder_sigmas.append(numpy.std(Binder_boot))

                        Binders.append(Binder)

                    masses = numpy.array(masses)
                    critical_found = False

                    if critical_found:
                        if bootstrap_success:
                            ax.errorbar(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    else:
                        if bootstrap_success:
                            ax.errorbar(masses, Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')
                        else:
                            ax.scatter(masses, Binders, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), facecolors=None)

                    if reweight:
                        # Use Andreas's Critical Analysis Class to do the reweighting
                        min_m, max_m = min(masses), max(masses)

                        System.Bbar = 0.5
                        L0bs = System.refresh_L0_bsindices()
                        L1bs = []
                        trphi2 = []
                        print("mass     lower sigma     upper sigma")

                        for j in range(len(System.actualm0sqlist)):
                            trphi2.append(System.phi2[str(j)])
                            lower_extreme_sigma = (numpy.mean(trphi2[j]) - min(trphi2[j])) / numpy.std(trphi2[j])
                            upper_extreme_sigma = (max(trphi2[j]) - numpy.mean(trphi2[j])) / numpy.std(trphi2[j])
                            print(f"{System.actualm0sqlist[j]:.3f}        ", end="")
                            print(f"{lower_extreme_sigma:.2f}       ", end="")
                            print(f"{upper_extreme_sigma:.2f}")

                            N_ = trphi2[j].shape[0]
                            L1bs.append(numpy.arange(int(numpy.floor(N_ / System.Nbin_tauint[j]))))

                        no_reweight_samples = 100
                        # min_m = -0.044454545454545455
                        # min_m = -0.044454545454545455

                        mass_range = numpy.linspace(min_m, max_m, no_reweight_samples)
                        results = numpy.zeros(no_reweight_samples)
                        sigmas = numpy.zeros(no_reweight_samples)

                        # System.plot_tr_phi2_distributions()
                        # plt.show()

                        for i, m in tqdm(enumerate(mass_range)):
                            # try:
                            Binder_bit, sigma = System.reweight_Binder(m, L1bs, L0bs, sigma=True)

                            # except Exception:
                            #     continue

                            print(Binder_bit, sigma)

                            if abs(Binder_bit) < 10 ** -10:
                                x = 2
                                print("hello?")
                                x = 3

                            results[i] = Binder_bit
                            sigmas[i] = sigma

                        if critical_found:
                            ax.fill_between(((mass_range - m_crit) / g ** 2) * (g * L) ** (1 / nu), results + System.Bbar - sigmas, results + System.Bbar + sigmas)

                        else:
                            ax.fill_between(mass_range, results + System.Bbar - sigmas, results + System.Bbar + sigmas)

    if legend:
        plt.legend()

    return ax, critical_found
