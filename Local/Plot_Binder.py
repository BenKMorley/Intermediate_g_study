import h5py
import re
import pdb
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from tqdm import tqdm
import sys
import os
from scipy.stats import shapiro


# Import from the Core directory
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.Binderanalysis import Critical_analysis
from Core.parameters import *
from Local.analysis_class import *


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


def plot_Binder(N, g_s=None, L_s=None, data_file=None, data_dir=None, minus_sign_override=True,
                legend=True, ax=None, min_gL=3.1, reweight=True, plot_lims=None, min_traj=0,
                scale_with_fit=False, no_reweight_samples=100, crossings_file=None, model='A',
                plot_crossings=False, plot_histograms=False, width=0.5, remove_outliers=False):
    dont_plot_hist = False
    text_height = 0.003

    if data_file is None:
        data_file = param_dict[N]["MCMC_data_file"]

    if crossings_file is None:
        crossings_file = param_dict[N]["h5_data_file"]

    if data_dir is None:
        data_dir = f'{h5data_dir}/'

    if ax is None:
        fig, ax = plt.subplots()

        ax.set_xlabel(r'$m^2_r$')
        ax.set_ylabel(r'$B$')

    markers = {8: 'd', 16: 'v', 32: '<', 48: '^', 64: 's', 96: 'o', 128: 'd'}

    f = h5py.File(f"{data_dir}{data_file}", "r")

    a = analysis(N, model)
    a.fit_all_gLmin_all_Bbar()
    a.BMM_overall(perform_checks=True)

    if g_s is None:
        g_s = set(a.g_s)

    if L_s is None:
        L_s = set(a.L_s)

    for g in g_s:
        m_crit = a.mass_mean[g]
        nu = a.mean[-1]
        a1 = a.mean[1]
        a2 = a.mean[2]

        if model == "B" or model == "C":
            c1 = a.mean[3]
            c2 = a.mean[4]

        if model == "C":
            e1 = a.mean[5]
            e2 = a.mean[6]

        # Correction to scaling term
        def correction(B):
            omega = 0.8
            epsilon = 2

            if model == "B":
                return -(c1 * B + c2) * (g * L) ** -omega

            if model == "C":
                return -(c1 * B + c2) * (g * L) ** -omega - (e1 * B + e2) * (g * L) ** -epsilon

            # For model A there is no correction
            else:
                return 0

        for L in L_s:
            print(f'Running for g = {g}, L = {L}')

            if (g * L) <= min_gL:
                continue

            data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

            masses = []
            Binders = []
            Binder_sigmas = []

            System = Critical_analysis(N, g, L, width=width)
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

                else:
                    masses.append(m)

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

                    Binder_boot[i] = 1 - (N / 3) * numpy.mean(M4_sample) / \
                                                  (numpy.mean(M2_sample) ** 2)

                Binders.append(Binder)
                Binder_sigmas.append(numpy.std(Binder_boot))

            masses = numpy.array(masses)
            Binders = numpy.array(Binders)

            if scale_with_fit:
                correction_term = correction(Binders)

                ax.errorbar(((masses - m_crit) / g ** 2) * (g * L) ** (1 / nu) + correction_term,
                            Binders, Binder_sigmas, marker=markers[L], label=f'g={g}, L={L}',
                            color=fig3_color(g * L, min_gL=min_gL), ls='',
                            fillstyle='none')

            else:
                ax.errorbar(masses, Binders, Binder_sigmas, marker=markers[L],
                            label=f'g={g}, L={L}',
                            color=fig3_color(g * L, min_gL=min_gL), ls='',
                            fillstyle='none')

            # Find the largest deviation between masses and assume half of this is the reweighting
            # length
            masses = numpy.array(masses)
            max_gap = numpy.max(numpy.abs(masses[1:] - masses[:-1]))

            if plot_lims is None:
                min_m, max_m = min(masses) - max_gap, max(masses) + max_gap
            else:
                min_m, max_m = plot_lims

            if reweight:
                # Use Andreas's Critical Analysis Class to do the reweighting
                System.Bbar = 0.5
                L0bs = System.refresh_L0_bsindices()
                L1bs = []
                trphi2 = []

                for j in range(len(System.actualm0sqlist)):
                    trphi2.append(System.phi2[str(j)])
                    N_ = trphi2[j].shape[0]
                    L1bs.append(numpy.arange(int(numpy.floor(N_ / System.Nbin_tauint[j]))))

                mass_range = numpy.linspace(min_m, max_m, no_reweight_samples)
                results = numpy.zeros(no_reweight_samples)
                sigmas = numpy.zeros(no_reweight_samples)

                for i, m in enumerate(mass_range):
                    results[i], sigmas[i] = System.reweight_Binder(m, L1bs, L0bs, sigma=True)

                if scale_with_fit:
                    correction_term = correction(results + System.Bbar)

                    ax.fill_between(((mass_range - m_crit) / g ** 2) * (g * L) ** (1 / nu) +
                                        correction_term, results + System.Bbar - sigmas,
                                        results + System.Bbar + sigmas, alpha=0.2,
                                        color=fig3_color(g * L, min_gL=min_gL))

                else:
                    ax.fill_between(mass_range, results + System.Bbar - sigmas,
                                    results + System.Bbar + sigmas,
                                    color=fig3_color(g * L, min_gL=min_gL))

                # Now plot the Binderanalysis fits if possible
                if plot_crossings:
                    cross_file = h5py.File(f'{data_dir}{crossings_file}', 'r')
                    cross_data = cross_file[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                    Bbar_s = []
                    means = []
                    stds = []

                    for key in cross_data.keys():
                        Bbar = float(re.findall(r'Bbar=\d+.\d+', key)[0][5:])
                        mean = cross_data[key]['central'][()]
                        std = cross_data[key]['std'][()]

                        if mean > min_m and mean < max_m:
                            Bbar_s.append(Bbar)

                            if remove_outliers:
                                samples = cross_data[key]['bs_bins'][()]
                                samples = samples[numpy.abs(samples - mean) < 5 * std]
                                mean = numpy.mean(samples)
                                std = numpy.std(samples)

                            means.append(mean)
                            stds.append(std)

                            # ax.text(mean - 2 * std, Bbar - text_height * 2, f'sigma={std:.2e}', color='r',
                            #         alpha=0.5)
                            # ax.text(mean - 2 * std, Bbar - text_height * 3, f'mu={mean:.5e}', color='g',
                            #         alpha=0.5)

                    Bbar_s = numpy.array(Bbar_s)

                    if Bbar_s != []:
                        if scale_with_fit:
                            correction_term = correction(Bbar_s)

                            ax.errorbar(((means - m_crit) / g ** 2) * (g * L) ** (1 / nu)
                                        + correction_term, Bbar_s,
                                        xerr=(stds / g ** 2) * (g * L) ** (1 / nu), ls='', color='k')

                        else:
                            ax.errorbar(means, Bbar_s, xerr=stds, ls='', color='k')
                            delta = max(means) - min(means)
                            ax.set_xlim(min(means) - 0.1 * delta, max(means) + 0.1 * delta)

                            # Scale the plot appropriately
                            ax.set_ylim(min(Bbar_s) - 0.01, max(Bbar_s) + 0.01)

                    else:
                        print('No crossing points found')
                        dont_plot_hist = True

                if plot_histograms and not dont_plot_hist:
                    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    current_color_index = 0

                    color_dict = {}

                    cross_file = h5py.File(f'{data_dir}{crossings_file}', 'r')
                    cross_data = cross_file[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                    Bbar_s = []
                    means = []
                    stds = []

                    for key in cross_data.keys():
                        Bbar = float(re.findall(r'Bbar=\d+.\d+', key)[0][5:])
                        data = cross_data[key]['bs_bins'][()]
                        print(f"Found {len(data)} data samples")

                        mean = cross_data[key]['central'][()]
                        std = cross_data[key]['std'][()]

                        if mean < min_m or mean > max_m:
                            print(f'Skipping histogram at {mean}')
                            continue

                        if remove_outliers:
                            data = cross_data[key]['bs_bins'][()]
                            print(f'Max deviation = {numpy.max(numpy.abs(data - mean)) / std} sigma')
                            data = data[numpy.abs(data - mean) < 5 * std]
                            mean = numpy.mean(data)
                            std = numpy.std(data)

                        # Global histogram to fix the bins
                        num_bins = 20
                        heights, bins = numpy.histogram(data, num_bins)

                        # Extract the seperate contributions to this histogram
                        m = numpy.array([i for i in cross_data[key]])
                        m = m[m != 'central']
                        m = m[m != 'bs_bins']
                        m = m[m != 'std']

                        # Data that doesn't have contribution splitting
                        if len(m) == 0:
                            ax.bar(bins[:-1], heights / 10000, bottom=Bbar,
                                    width=numpy.diff(bins), alpha=0.3, color='k')

                        else:
                            for sub_key in m:
                                sub_data = cross_data[key][sub_key]

                                if remove_outliers:
                                    print(f'Max deviation = {numpy.max(numpy.abs(sub_data - cross_data[key]["central"][()])) / cross_data[key]["std"][()]} sigma')
                                    sub_data = sub_data[numpy.abs(sub_data - cross_data[key]['central'][()]) < 5 * cross_data[key]['std'][()]]

                                heights, sub_bins = numpy.histogram(sub_data, bins=bins)

                                if sub_key not in color_dict:
                                    color_dict[sub_key] = color_cycle[current_color_index % len(color_cycle)]
                                    current_color_index += 1

                                    # Add combination of reweighting contributions to the legend
                                    ax.bar(sub_bins[:-1], heights / 10000, bottom=Bbar,
                                        width=numpy.diff(bins), alpha=0.3,
                                        color=color_dict[sub_key])

                                else:
                                    ax.bar(sub_bins[:-1], heights / 10000, bottom=Bbar,
                                        width=numpy.diff(bins), alpha=0.3,
                                        color=color_dict[sub_key])

                        # Do the shapiro test on the data
                        try:
                            W, p = shapiro(data)
                            ax.text(mean - 2 * std, Bbar - text_height, f'p = {p:.4f}', color='k', alpha=0.5)

                        except Exception:
                            print("Could not perform the Shapiro test")

    f.close()

    if legend:
        plt.legend()

    if not scale_with_fit:
        ax.set_xlim(min_m, max_m)

    plt.title(rf"$N = {N}$, $\nu = {nu:.2f}$, model={model}")

    # Make a legend
    legend = []
    for L in L_s:
        legend.append(Line2D([0], [0], marker=markers[L], color='k', label=f'L = {L}',
                              markerfacecolor="None", linestyle=''))

    ax.legend(handles=legend, loc='lower left')

    return ax
