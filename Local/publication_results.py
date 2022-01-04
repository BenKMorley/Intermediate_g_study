###############################################################################
# Copyright (C) 2020
#
# Author: Ben Kitching-Morley bkm1n18@soton.ac.uk
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# See the full license in the file "LICENSE" in the top level distribution
# directory

# The code has been used for the analysis presented in
# "Nonperturbative infrared finiteness in super-renormalisable scalar quantum
# field theory" https://arxiv.org/abs/2009.14768
###################################################################################

import matplotlib.pyplot as plt
import pdb
import sys
import os
import numpy


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.frequentist_run import run_frequentist_analysis
from Core.model_definitions import *
# from Core.bayesian_functions import *
from Core.parameters import param_dict, seperator, h5data_dir
from Core.MISC import calc_gL_mins


def get_pvalues_central_fit(N):
    """
        This function will reproduce the p-value data against the gL_min cut
        for the central fit.

        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields

        OUTPUTS:
        --------
        pvalues: dict of arrays of floats. Each array is of the length of the
            number of gL_min cut values, and the corresponding p-value to each
            cut is recorded.

            > "pvalues1": Values for model 1 (Lambda_IR = g / (4 pi N))
            > "pvalues2": Values for model 1 (Lambda_IR = 1 / L)
    """
    N_s = [N]
    Bbar_s = param_dict[N]["Central_Bbar"]
    g_s = param_dict[N]["g_s"]
    L_s = param_dict[N]["L_s"]
    x0 = param_dict[N]["x0"]
    model_1 = param_dict[N]["model_1"]
    model_2 = param_dict[N]["model_2"]
    h5_data_file = f"{h5data_dir}/{param_dict[N]['h5_data_file']}"

    gL_mins = calc_gL_mins(g_s, L_s)
    gL_mins = gL_mins[gL_mins <= param_dict[N]["gL_max"]]

    pvalues_1 = numpy.zeros(len(gL_mins))
    pvalues_2 = numpy.zeros(len(gL_mins))

    for i, gL_min in enumerate(gL_mins):
        pvalues_1[i], params1, dof = \
            run_frequentist_analysis(h5_data_file, model_1, N_s, g_s, L_s, Bbar_s, gL_min, x0,
                                     run_bootstrap=False)

        pvalues_2[i], params2, dof = \
            run_frequentist_analysis(h5_data_file, model_2, N_s, g_s, L_s, Bbar_s, gL_min, x0,
                                     run_bootstrap=False, print_info=False)

    pvalues = {}
    pvalues["pvalues1"] = pvalues_1
    pvalues["pvalues2"] = pvalues_2

    print(seperator)

    return pvalues


def get_statistical_errors_central_fit(N, model_name="model_1"):
    """
        This function gets the statistical error bands (and central fit values)
        for the model parameters, and the value of the critical mass at g=0.1
        quoted in the publication.

        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields
        model_name: string, either "model_1" or "model_2". Determines whether to
            look at the central fit for either Lambda_IR = g / 4 pi N (model_1)
            or Lambda_IR = 1 / L (model_2)

        OUTPUTS :
        ---------
        results: dictionary containing:
            > "params": list of floats, parameter estimates of the central fit
            > "params_std": list of floats, statistical error on these
                estimates
            > "param_names": list of strings, the names of the parameters in
                the same order as in "params" and "params_std"

            if N == 2 and model_name == "model_1":
                > "m_c": float, Estimate of the critical mass at g = 0.1
                > "m_c_error": float, Statistical error on this estimate

            if N == 4 and model_name == "model_1":
                > "m_c1": float, Estimate of the critical mass using alpha 1 at
                    g = 0.1
                > "m_c_error1": float, Statistical error on this estimate using
                    alpha1
                > "m_c2": float, Estimate of the critical mass using alpha 2 at
                    g = 0.1
                > "m_c_error2": float, Statistical error on this estimate using
                    alpha2
    """
    N_s = [N]
    Bbar_s = param_dict[N]["Central_Bbar"]
    g_s = param_dict[N]["g_s"]
    L_s = param_dict[N]["L_s"]
    x0 = param_dict[N]["x0"]
    model = param_dict[N][model_name]
    gL_min = param_dict[N]["gL_central"][model_name]
    param_names = param_dict[N]["param_names"]
    h5_data_file = f"{h5data_dir}/{param_dict[N]['h5_data_file']}"

    # Run once with the full dataset (no resampling)
    pvalue, params_central, dof =\
        run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s, gL_min, x0,
                                 run_bootstrap=False)

    # Run with all the bootstrap samples
    pvalue, params, dof =\
        run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s, gL_min, x0,
                                 print_info=False)

    sigmas = numpy.zeros(len(params_central))
    no_samples = params.shape[0]

    for i, param in enumerate(param_names):
        # Find the standard deviation with respect to the central values
        sigmas[i] = numpy.sqrt(numpy.sum((params[:, i] - params_central[i]) ** 2) / no_samples)

        print(f"{param} = {params_central[i]} +- {sigmas[i]}")

    results = {}
    results["params_central"] = params_central
    results["params_std"] = sigmas

    # Calculate the value of the non-perterbative critical mass for g = 0.1 and
    # it's statistical error
    if model_name == "model_1":
        g = 0.1
        m_c = mPT_1loop(g, N) + g ** 2 * (params_central[0] - params_central[-2] * K1(g, N))

        if model == model1_1a:
            print("Critical Mass using alpha1:")

        print(f"m_c = {m_c}")

        alphas = params[..., 0]
        betas = params[..., -2]

        m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - betas * K1(g, N))
        m_c_error = numpy.sqrt(numpy.sum((m_cs - m_c) ** 2) / no_samples)

        print(f"m_c_error = {m_c_error}")
        results["m_c"] = m_c
        results["m_c_error"] = m_c_error

    # If there are two values of alpha also calculate the critical mass with alpha2
    if model == model1_2a:
        alphas2 = params[..., 1]

        m_c2 = mPT_1loop(g, N) + g ** 2 * (params_central[1] - params_central[-2] * K1(g, N))

        print("Critical Mass using alpha2:")
        print(f"m_c2 = {m_c2}")

        m_c2s = mPT_1loop(g, N) + g ** 2 * (alphas2 - betas * K1(g, N))
        m_c2_error = numpy.sqrt(numpy.sum((m_c2s - m_c2) ** 2) / no_samples)

        print(f"m_c2_error = {m_c2_error}")
        results["m_c1"] = m_c
        results["m_c_error1"] = m_c_error
        results["m_c2"] = m_c2
        results["m_c_error2"] = m_c2_error

    results["param_names"] = param_names

    return results


def get_systematic_errors(N, model_name="model_1", min_dof=15):
    """
        This function calculates the systematic error bands (and central fit
        values) for the model parameters, and the value of the critical mass at
        g = 0.1 quoted in the publication.

        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields
        model_name: string, either "model_1" or "model_2". Determines whether to
            look at either Lambda_IR = g / 4 pi N (model_1) or
            Lambda_IR = 1 / L (model_2)
        min_dof: int, the minimum number of degrees of freedom for a fit to be considered
            acceptable

        OUTPUTS :
        ---------
        results: dictionary containing:
            > "params": list of floats, parameter estimates of the central fit
            > "params_std": list of floats, systematic error on these estimates
            > "param_names": list of strings, the names of the parameters in
                the same order as in "params" and "params_std"

            if N == 2 and model_name == "model_1":
                > "m_c": float, Estimate of the critical mass
                > "m_c_error": float, Systematic error on this estimate

            if N == 4 and model_name == "model_2":
                > "m_c1": float, Estimate of the critical mass using alpha 1
                > "m_c_error1": float, Systematic error on this estimate using
                    alpha1
                > "m_c2": float, Estimate of the critical mass using alpha 2
                > "m_c_error2": float, Systematic error on this estimate using
                    alpha2
                > "m_c": float, Overall estimate of critical mass (same as
                    alpha 1 result)
                > "m_c_error": float, Overall systematic error when accounting
                    for both alpha values
    """
    N_s = [N]
    Bbar_s = param_dict[N]["Bbar_list"]
    g_s = param_dict[N]["g_s"]
    L_s = param_dict[N]["L_s"]
    x0 = param_dict[N]["x0"]
    model = param_dict[N][model_name]
    h5_data_file = f"{h5data_dir}/{param_dict[N]['h5_data_file']}"
    param_names = param_dict[N]["param_names"]
    n_params = len(param_names)

    gL_mins = calc_gL_mins(g_s, L_s)
    gL_mins = gL_mins[gL_mins <= param_dict[N]["gL_max"]]

    # Make a list of all Bbar pairs
    Bbar_list = []
    for i in range(len(Bbar_s)):
        for j in range(i + 1, len(Bbar_s)):
            Bbar_list.append([Bbar_s[i], Bbar_s[j]])

    pvalues = numpy.zeros((len(Bbar_list), len(gL_mins)))
    params = numpy.zeros((len(Bbar_list), len(gL_mins), n_params))
    dofs = numpy.zeros((len(Bbar_list), len(gL_mins)))

    for i, Bbar_s in enumerate(Bbar_list):
        Bbar_1, Bbar_2 = Bbar_s
        print(f"Running fits with Bbar_1 = {Bbar_1}, Bbar_2 = {Bbar_2}")

        for j, gL_min in enumerate(gL_mins):
            pvalues[i, j], params[i, j], dofs[i, j] = \
                run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s, gL_min, x0,
                                         run_bootstrap=False)

    # Extract the index of the smallest gL_min fit that has an acceptable p-value
    r = len(gL_mins)
    best = r - 1

    for i, gL_min in enumerate(gL_mins):
        if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
            best = r - 1 - i

    best_Bbar_index = numpy.argmax(pvalues[:, best])
    best_Bbar = Bbar_list[best_Bbar_index]

    print("##################################################################")
    print("BEST RESULT")
    print(f"Bbar_s = {best_Bbar}")
    print(f"gL_min = {gL_mins[best]}")
    print(f"pvalue : {pvalues[best_Bbar_index, best]}")
    print(f"dof : {dofs[best_Bbar_index, best]}")
    print("##################################################################")

    params_central = params[best_Bbar_index, best]

    # Find the parameter variation over acceptable fits
    acceptable = numpy.logical_and(
                    numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                    dofs >= min_dof)

    if numpy.sum(acceptable) == 0:
        print("##############################################################")
        print("No acceptable fits found!")
        return None

    # Find the most extreme values of the parameter estimates that are deemed acceptable
    sys_sigmas = numpy.zeros(n_params)

    for i, param in enumerate(param_names):
        param_small = params[..., i]
        minimum = numpy.min(param_small[acceptable])
        maximum = numpy.max(param_small[acceptable])

        # Define the systematic error bar by the largest deviation from the
        # central fit by an acceptable fit
        sys_sigmas[i] = max(maximum - params[best_Bbar_index, best, i],
                            params[best_Bbar_index, best, i] - minimum)

        print(f"{param} = {params_central[i]} +- {sys_sigmas[i]}")

    # Find the systematic variation in the critical mass
    if model.__name__ == "model1_1a" or model.__name__ == "model1_2a":
        g = 0.1
        m_c = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 0] -
                                          params[best_Bbar_index, best, -2] * K1(g, N))

        if model.__name__ == "model1_2a":
            print("Critical Mass using alpha1:")

        print(f"m_c = {m_c}")

        alphas = params[..., 0]
        betas = params[..., -2]

        # Only include parameter estimates from those fits that are acceptable
        alphas = alphas[acceptable]
        betas = betas[acceptable]

        m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - betas * K1(g, N))

        minimum_m = numpy.min(m_cs)
        maximum_m = numpy.max(m_cs)

        m_c_error = max(m_c - minimum_m, maximum_m - m_c)
        print(f"m_c_error = {m_c_error}")

    results = {}
    results["params_central"] = params_central
    results["params_std"] = sys_sigmas
    results["m_c"] = m_c
    results["m_c_error"] = m_c_error

    if model == model1_2a:
        alphas2 = params[..., 1]

        # Only include parameter estimates from those fits that are acceptable
        alphas2 = alphas2[acceptable]

        # Calculate using alpha2
        m_c2 = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 1] -
                params[best_Bbar_index, best, -2] * K1(g, N))

        print("Critical Mass using alpha2:")
        print(f"m_c2 = {m_c2}")

        m_c2s = mPT_1loop(g, N) + g ** 2 * (alphas2 - betas * K1(g, N))

        minimum_m2 = numpy.min(m_c2s)
        maximum_m2 = numpy.max(m_c2s)

        m_c2_error = max(m_c2 - minimum_m2, maximum_m2 - m_c2)
        print(f"m_c2_error = {m_c2_error}")

        # Get the overall systematic error accounting for both alphas
        m_c_error_overall = max(max(m_c - minimum_m, maximum_m - m_c),
                                max(m_c - minimum_m2, maximum_m2 - m_c))

        print("Overall result when accounting for both alphas:")
        print(f"m_c = {m_c} +- {m_c_error_overall}")

        results["m_c1"] = m_c
        results["m_c_error1"] = m_c_error
        results["m_c2"] = m_c2
        results["m_c_error2"] = m_c2_error
        results["m_c_error"] = m_c_error_overall

    results["param_names"] = param_names

    return results


def get_Bayes_factors(N, points=5000, gL_max=numpy.inf):
    """
        This function produces the Bayes Factors shown in the publication.
        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields
        points: int, number of points to use in the MULTINEST algorithm. The
            higher this is the more accurate the algorithm will be, but at the
            price of computational cost. To produce the plot of the Bayes
            factor against gL_min 5000 points were used. For the posterior
            plots 1000 points were used.
        OUTPUTS :
        ---------
        Bayes_factors: The log10 of the Bayes factor of the
            Lambda_IR = g / (4 pi N) model over the Lambda_IR = 1 / L model.
            This is an array of lenght equal to the number of gL_min cuts
            considered, with each element containin the log Bayes factor of the
            corresponding gL_min cut.
    """
    N_s_in = [N]
    Bbar_s_in = param_dict[N]["Central_Bbar"]
    g_s_in = param_dict[N]["g_s"]
    L_s_in = param_dict[N]["L_s"]
    model_1 = param_dict[N]["model_1"]
    model_2 = param_dict[N]["model_2"]
    h5_data_file = f"{h5data_dir}/{param_dict[N]['h5_data_file']}"
    param_names = param_dict[N]["param_names"]
    n_params = len(param_names)
    prior_range = param_dict[N]["prior_range"]

    gL_mins = calc_gL_mins(g_s_in, L_s_in)
    gL_mins = gL_mins[gL_mins <= param_dict[N]["gL_max"]]

    # Where the output samples will be saved
    directory = "Data/MULTINEST_samples/"

    # Use this to label different runs if you edit something
    tag = ""

    # Prior Name: To differentiate results which use different priors
    prior_name = "A"

    # For reproducability
    seed = 3475642

    Bayes_factors = numpy.zeros(len(gL_mins))

    for i, gL_min in enumerate(gL_mins):
        samples, g_s, L_s, Bbar_s, m_s = \
            load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, gL_min, gL_max)

        analysis1, best_fit1 = \
            run_pymultinest(prior_range, model_1, gL_min, gL_max, n_params, directory, N, g_s,
                            Bbar_s, L_s, samples, m_s, param_names, n_live_points=points,
                            sampling_efficiency=0.3, clean_files=True, tag=tag,
                            prior_name=prior_name, keep_GLmax=False, return_analysis_small=True,
                            seed=seed)

        analysis2, best_fit2 = \
            run_pymultinest(prior_range, model_2, gL_min, gL_max, n_params, directory, N, g_s,
                            Bbar_s, L_s, samples, m_s, param_names, n_live_points=points,
                            sampling_efficiency=0.3, clean_files=True, tag=tag,
                            prior_name=prior_name, keep_GLmax=False, return_analysis_small=True,
                            seed=seed)

        # This is the log of the Bayes factor equal to the difference in the
        # log-evidence's between the two models
        Bayes_factors[i] = analysis1[0] - analysis2[0]

    # Change log bases to log10 to match the plot in the publication
    Bayes_factors = Bayes_factors / numpy.log(10)

    return Bayes_factors


def Plot_fit(N):
    def colors(g):
        if g == 0.1:
            return 'g'
        if g == 0.2:
            return 'r'
        if g == 0.3:
            return 'k'
        if g == 0.5:
            return 'y'
        if g == 0.6:
            return 'b'

    if N == 3:
        model = model_1_1a
        Bbar_s = numpy.array([0.43, 0.44])

        g_s_in = [0.1, 0.2, 0.3, 0.6]
        L_s_in = [16, 32, 48, 64, 96]
        N_s_in = [N]
        Bbar_s_in = [0.43, 0.44]
        gL_min = 12.8
        gL_max = 76.8

    results = get_statistical_errors_central_fit(N)
    central_values = results["params_central"]

    samples, g_s, L_s, Bbar_s, m_s = load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, gL_min, gL_max)

    gL_space = numpy.linspace(min(g_s * L_s), max(g_s * L_s), 1000)

    for g in set(g_s):
        for i, Bbar in enumerate(set(Bbar_s)):
            sub_ind = numpy.argwhere(numpy.logical_and(g_s == g, Bbar_s == Bbar))

            if N == 2 or N == 3:
                if i == 0:
                    plt.plot(gL_space, model(N, g, gL_space / g, Bbar, *central_values) / g, label=f'g = {g}', color=colors(g))

                if i == 1:
                    plt.plot(gL_space, model(N, g, gL_space / g, Bbar, *central_values) / g, color=colors(g))

            if N == 4:
                if i == 0:
                    plt.plot(gL_space, model_Bbar_list(list(set(Bbar_s)), N, g, gL_space / g, Bbar, *central_values) / g, label=f'g = {g}', color=colors(g))

                if i == 1:
                    plt.plot(gL_space, model_Bbar_list(list(set(Bbar_s)), N, g, gL_space / g, Bbar, *central_values) / g, color=colors(g))

            plt.scatter(g * L_s[sub_ind], m_s[sub_ind] / g, color=colors(g))

    plt.xlabel("x")
    plt.ylabel("m[B = Bbar] / g")
    plt.legend()
    plt.show()
