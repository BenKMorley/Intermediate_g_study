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
###############################################################################

from model_definitions import *
from scipy.optimize import minimize, least_squares
from scipy.linalg import logm
from tqdm import tqdm
import warnings


def run_frequentist_analysis(input_h5_file, model, N_s, g_s, L_s, Bbar_s_in,
                             GL_min, GL_max, param_names, x0, method="lm",
                             no_samples=500, run_bootstrap=True,
                             print_info=True):
    """
        Central function to run the frequentist analysis. This function is
        used repeatedly in publication_results.py

        INPUTS:
        -------
        input_h5_file: string, file name of input Binder crossing mass values
        model: Fit anzatz function
        N_s: List of ints of N (SU(N) rank) values to be studied. N values
            should be ints
        g_s: List of ag (coupling constant in lattice units) values to be
            studied, ag values should be floats
        L_s: List of L / a (Lattice Size) values to be studied, L values should
            be ints
        Bbar_s_in: List of Bbar crossing values to be studied, Bbar values
            should be floats
        GL_min: Float, minimum value of the product (ag) * (L / a) to be
            included in the fit
        GL_max: Float, maximum value of the product (ag) * (L / a) to be
            included in the fit
        param_names: list of strings, with each entry being the name of a
            parameter of the fit anzatz
        x0: list of floats, starting parameter guesses for the minimizer
        method: string, name of the minimization routine to be used
        no_samples: int, number of bootstrap samples to use
        run_bootstrap: bool, if False the function won't run a bootstrap
        print_info: bool, if True extra infomation about the fit results is
            returned to std::out

        OUTPUTS:
        --------
        Depends on whether a bootstrap is run. This occurs only if
        run_bootstrap is True and the p-value of the central fit is greater
        than 0.05.

        If a bootstrap is run:
        > p: float, p-value of the central fit
        > param_estimates: list of floats, parameter values at MAP estimate
        > dof: int, number of degrees of freedom in the fit
        > sigmas: list of floats, standard deviation of the parameter estimates
            under the bootstrap

        If a bootstrap isn't run:
        > p: float, p-value of the central fit
        > param_estimates: list of floats, parameter values at MAP estimate
        > dof: int, number of degrees of freedom in the fit
    """

    samples, g_s, L_s, Bbar_s, m_s = load_h5_data(input_h5_file, N_s, g_s, L_s,
                                                  Bbar_s_in, GL_min, GL_max)
    N = N_s[0]

    cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

    # Using scipy.optimize.least_squares
    if method == "least_squares":
        res = least_squares(res_function, x0, args=(cov_inv, model))

    if method == "lm":
        res = least_squares(res_function, x0, args=(cov_inv, model),
                            method="lm")

    # Using scipy.optimize.minimize
    if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS", "COBYLA"]:
        res = minimize(lambda x, y, z: numpy.sum(res_function(x, y, z) ** 2),
                       x0, args=(cov_inv, model), method=method)

    chisq = chisq_calc(res.x, cov_inv, model, res_function)
    n_params = len(res.x)
    dof = g_s.shape[0] - n_params
    p = chisq_pvalue(dof, chisq)
    param_central = res.x

    if print_info:
        print("##############################################################")
        print(f"Config: N = {N}, Bbar_s = [{Bbar_s_in[0]}, {Bbar_s_in[1]}],"
              f" gL_min = {GL_min}, gL_max = {GL_max}")
        print(f"chisq = {chisq}")
        print(f"chisq/dof = {chisq / dof}")
        print(f"pvalue = {p}")
        print(f"dof = {dof}")

    # If the pvalue is acceptable, run a bootstrap to get a statistical error
    if run_bootstrap:
        param_estimates = numpy.zeros((no_samples, n_params))

        for i in tqdm(range(no_samples)):
            m_s = samples[:, i]

            res_function = make_res_function(N_s[0], m_s, g_s, L_s, Bbar_s)

            if method == "least_squares":
                res = least_squares(res_function, x0, args=(cov_inv, model))

            if method == "lm":
                res = least_squares(res_function, x0, args=(cov_inv, model),
                                    method=method)

            # Using scipy.optimize.minimize
            if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS",
                          "COBYLA"]:
                def dummy_func(x, y, z):
                    return numpy.sum(res_function(x, y, z) ** 2)
                res = minimize(dummy_func, x0, args=(cov_inv, model),
                               method=method)

            param_estimates[i] = numpy.array(res.x)

        return p, param_estimates, dof

    return p, res.x, dof
