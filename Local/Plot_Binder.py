import h5py
import re
import pdb
import numpy
import matplotlib.pyplot as plt
from scipy.optimize.nonlin import NoConvergence
from tqdm import tqdm
import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')


from Core.model_definitions import mPT_1loop, K1
from publication_results import get_statistical_errors_central_fit
from Core.Binderanalysis import Critical_analysis
from Core.parameters import *

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


def plot_Binder(N, g_s, L_s, data_file=None, data_dir=None, minus_sign_override=False,
                legend=True, ax=None, min_gL=3.1, max_gL=76.81, reweight=True, params=None, GL_lim=12.7,
                mlims=None, min_traj=100001, therm=10000, scale_with_fit=False, no_reweight_samples=100,
                crossings_file=None, plot_crossings=False):

    if data_file is None:
        data_file = param_dict[N]["MCMC_data_file"]

    if crossings_file is None:
        crossings_file = param_dict[N]["h5_data_file"]

    if data_dir is None:
        data_dir = h5data_dir

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\frac{m^2 - m_c^2}{g^2} x^\frac{1}{\nu}$')
        ax.set_ylabel(r'$B(N, g, L)$')

    markers = {8: 'd', 16: 'v', 32: '<', 48: '^', 64: 's', 96: 'o', 128: 'd'}

    f = h5py.File(f"{data_dir}{data_file}", "r")
    if scale_with_fit:
        if params is None:
            try:
                params = get_statistical_errors_central_fit(N)['params_central']

                alpha = params[0]
                beta = params[-2]
                nu = params[-1]

            except Exception:
                scale_with_fit = False

    for g in g_s:
        if scale_with_fit:
            m_crit = mPT_1loop(g, N) + g ** 2 * (alpha - beta * K1(g, N))

        for L in L_s:
            if (g * L) > GL_lim:
                data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                masses = []
                Binders = []
                Binder_sigmas = []

                System = Critical_analysis(N, g, L)
                System.MCMCdatafile = data_file
                System.datadir = data_dir
                System.min_traj = min_traj

                System.h5load_data()

                System.compute_tauint_and_bootstrap_indices()

                for i, m in enumerate(System.actualm0sqlist):
                    bin_size = max(4 * System.Nbin_tauint[i], 50)
                    mass_data = data[f'msq={m:.8f}']

                    if minus_sign_override:
                        masses.append(-m)

                    M2 = mass_data[0]
                    M4 = mass_data[1]

                    Binder = 1 - (N / 3) * numpy.mean(M4) / (numpy.mean(M2) ** 2)

                    # Sort the data into bins
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

                        Binder_boot[i] = 1 - (N / 3) * numpy.mean(M4_sample) / (numpy.mean(M2_sample) ** 2)

                    Binders.append(Binder)
                    Binder_sigmas.append(numpy.std(Binder_boot))

                masses = numpy.array(masses)

                if scale_with_fit:
                    ax.errorbar(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu), Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')

                else:
                    ax.errorbar(masses, Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}', color=fig3_color(g * L, min_gL=min_gL, max_gL=max_gL), ls='', fillstyle='none')

                if reweight:
                    # Use Andreas's Critical Analysis Class to do the reweighting
                    if mlims is None:
                        min_m, max_m = min(masses), max(masses)
                    else:
                        min_m, max_m = mlims

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

                    mass_range = numpy.linspace(min_m, max_m, no_reweight_samples)
                    results = numpy.zeros(no_reweight_samples)
                    sigmas = numpy.zeros(no_reweight_samples)

                    # System.plot_tr_phi2_distributions()
                    # plt.show()

                    for i, m in tqdm(enumerate(mass_range)):
                        Binder_bit, sigma = System.reweight_Binder(m, L1bs, L0bs, sigma=True)

                        print(Binder_bit, sigma)

                        results[i] = Binder_bit
                        sigmas[i] = sigma

                    if scale_with_fit:
                        ax.fill_between(((mass_range - m_crit) / g ** 2) * (g * L) ** (1 / nu), results + System.Bbar - sigmas, results + System.Bbar + sigmas)

                    else:
                        ax.fill_between(mass_range, results + System.Bbar - sigmas, results + System.Bbar + sigmas)

                    # Now plot the Binderanalysis fits if possible
                    if plot_crossings:
                        cross_file = h5py.File(crossings_file, 'r')
                        cross_data = cross_file[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                        Bbar_s = []
                        means = []
                        stds = []
                        for key in cross_data.keys():
                            Bbar = float(re.findall(r'Bbar=\d+.\d+', key)[0][5:])
                            mean = cross_data[key]['central'][()]
                            std = cross_data[key]['std'][()]

                            Bbar_s.append(Bbar)
                            means.append(mean)
                            stds.append(std)

                        if scale_with_fit:
                            ax.errorbar(((means - m_crit) / g ** 2) * (g * L) ** (1 / nu), Bbar_s, xerr=(stds / g ** 2) * (g * L) ** (1 / nu))

                        else:
                            ax.errorbar(means, Bbar_s, xerr=stds, ls='', color='k')

                        cross_file.close()

    f.close()

    if legend:
        plt.legend()

    ax.set_xlim(min(mass_range), max(mass_range))

    return ax, scale_with_fit
