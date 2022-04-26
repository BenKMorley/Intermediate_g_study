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

import enum
from time import sleep
import matplotlib.pyplot as plt
import pdb
import sys
import os
import re
import numpy
import logging
from tqdm import tqdm
from scipy.optimize import minimize, least_squares

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.model_definitions import *
# from Core.bayesian_functions import *
from Core.parameters import param_dict, seperator, h5data_dir


logging.basicConfig(filename='Local/py_log/scaling_analysis_log.txt', level="INFO")

num_Bbars = 2


def run_minimizer(N, m_s, g_s, L_s, Bbar_s, cov_inv, x0, model, method, res_function):
    if method == "least_squares":
        res = least_squares(res_function, x0, args=(cov_inv, model))

    if method == "lm":
        res = least_squares(res_function, x0, args=(cov_inv, model), method=method)

    # Using scipy.optimize.minimize
    if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS", "COBYLA"]:
        def dummy_func(x, y, z):
            return numpy.sum(res_function(x, y, z) ** 2)

        res = minimize(dummy_func, x0, args=(cov_inv, model), method=method)

    return res


class scaling_analysis(object):
    def __init__(self, N, filename, model=model1_1a, gL_abs_min=0.7, min_dof=15):
        self.filename = filename
        self.N = N
        self.minimization_method = "lm"
        self.x0 = param_dict[self.N]['x0'][model.__name__]
        self.model = model
        self.min_dof = min_dof
        self.no_samples = 500
        self.param_names = list(model.__code__.co_varnames[:model.__code__.co_argcount][4:])
        self.n_params = len(self.param_names)

        self.load_data()

        self.gL_mins = numpy.sort(list(set(self.g_s * self.L_s)))
        self.gL_mins = self.gL_mins[self.gL_mins >= gL_abs_min]

    def load_data(self):
        self.g_s = []
        self.L_s = []
        self.B_s = []
        self.m_s = []
        self.samples = []

        with h5py.File(self.filename, 'r') as f:
            # We only want to analyse a single N
            assert f'N={self.N}' in f.keys(), "The data file needs to contain data for this N"

            data = f[f'N={self.N}']
            data_points = 0

            for g_key in data.keys():
                if len(re.findall(r'g=\d+\.\d+', g_key)) != 0:
                    g = float(re.findall(r'\d+\.\d+', g_key)[0])
                    g_data = data[g_key]

                else:
                    print(f'Invalid g key: {g_key}')
                    continue

                for L_key in g_data.keys():
                    if len(re.findall(r'L=\d+', L_key)) != 0:
                        L = int(re.findall(r'\d+', L_key)[0])
                        L_data = g_data[L_key]

                    else:
                        print(f'Invalid L key: {L_key}')
                        continue

                    for B_key in L_data.keys():
                        if len(re.findall(r'Bbar=\d+.\d+', B_key)) != 0:
                            B = float(re.findall(r'\d+.\d+', B_key)[0])
                            B_data = L_data[B_key]
                            data_points += 1

                        else:
                            print(f'Invalid L key: {B_key}')
                            continue

                        # Flag any samples that are zeros to avoid a singular matrix
                        if numpy.sum(numpy.abs(B_data['bs_bins'])) == 0:
                            print(f"Samples full of zeros for g = {g}, L = {L}, Bbar = {B}")
                            continue

                        if B in param_dict[self.N]["Bbar_list"]:
                            self.g_s.append(g)
                            self.L_s.append(L)
                            self.B_s.append(B)
                            self.m_s.append(B_data['central'][()])
                            self.samples.append(B_data['bs_bins'][()])

        self.g_s = numpy.array(self.g_s)
        self.L_s = numpy.array(self.L_s)
        self.B_s = numpy.array(self.B_s)
        self.m_s = numpy.array(self.m_s)
        self.samples = numpy.array(self.samples)

        # Remove nan values
        keep = numpy.logical_not(numpy.isnan(self.samples))[:, 0]
        self.samples = self.samples[keep]
        self.g_s = self.g_s[keep]
        self.L_s = self.L_s[keep]
        self.B_s = self.B_s[keep]
        self.m_s = self.m_s[keep]

        # Make lists of the unique value of the parameters
        self.g_list = list(numpy.sort(list(set(list(self.g_s)))))
        self.L_list = list(numpy.sort(list(set(list(self.L_s)))))
        self.B_list = list(numpy.sort(list(set(list(self.B_s)))))
        self.m_list = list(numpy.sort(list(set(list(self.m_s)))))

        print(f'Found {len(self.g_s)} data points')

    def get_systematic_errors(self):
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
        # Make a list of all Bbar pairs
        Bbar_list = []
        Bbar_s = list(set(self.B_s))
        for i in range(len(Bbar_s)):
            for j in range(i + 1, len(Bbar_s)):
                Bbar_list.append([Bbar_s[i], Bbar_s[j]])

        pvalues = numpy.zeros((len(Bbar_list), len(self.gL_mins)))
        params = numpy.zeros((len(Bbar_list), len(self.gL_mins), self.n_params))
        dofs = numpy.zeros((len(Bbar_list), len(self.gL_mins)))

        for i, Bs in enumerate(Bbar_list):
            Bbar_1, Bbar_2 = Bs
            print(f"Running fits with Bbar_1 = {Bbar_1}, Bbar_2 = {Bbar_2}")
            print_string = True

            for j, gL_min in enumerate(self.gL_mins):
                # Select the relevent subset of the data
                keep = numpy.logical_or(self.B_s == Bbar_1, self.B_s == Bbar_2)
                keep = numpy.logical_and(keep, self.g_s * self.L_s > gL_min - 10 ** -10)

                # Check if there are enough datapoints to do the fit
                if sum(keep) - self.n_params < 2:
                    continue

                samples = self.samples[keep]
                g_s = self.g_s[keep]
                L_s = self.L_s[keep]
                Bbar_s = self.B_s[keep]
                m_s = self.m_s[keep]

                cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
                cov_1_2 = numpy.linalg.cholesky(cov_matrix)
                cov_inv = numpy.linalg.inv(cov_1_2)

                res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

                res = run_minimizer(self.N, m_s, g_s, L_s, Bbar_s, cov_inv, self.x0, self.model,
                                    self.minimization_method, res_function)

                chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
                n_params = len(res.x)
                dof = g_s.shape[0] - n_params
                p = chisq_pvalue(dof, chisq)
                dofs[i, j] = dof
                pvalues[i, j] = p
                params[i, j] = res.x

                string = f"{seperator}\n"
                string += f"Config: N = {self.N}, Bbars = {Bs}, gL_min = {gL_min:.2f}\n"
                string += f"chisq = {chisq}\n"
                string += f"chisq/dof = {chisq / dof}\n"
                string += f"pvalue = {p}\n"
                string += f"dof = {dof}\n"
                string += f"params = {res.x}\n"

                if p > 0.05 and print_string:
                    print(string)
                    print_string = False

        # Extract the index of the smallest gL_min fit that has an acceptable p-value
        r = len(self.gL_mins)
        best = r - 1

        for i, gL_min in enumerate(self.gL_mins):
            if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
                best = r - 1 - i

        best_Bbar_index = numpy.argmax(pvalues[:, best])
        best_Bbar = Bbar_list[best_Bbar_index]
        self.gL_min_central = self.gL_mins[best]
        self.Bbars_central = best_Bbar
        self.p = pvalues[best_Bbar_index, best]

        print("##################################################################")
        print("BEST RESULT")
        print(f"Bbar_s = {best_Bbar}")
        print(f"gL_min = {self.gL_mins[best]:.2f}")
        print(f"pvalue : {pvalues[best_Bbar_index, best]}")
        print(f"dof : {dofs[best_Bbar_index, best]}")
        print("##################################################################")

        params_central = params[best_Bbar_index, best]

        # Find the parameter variation over acceptable fits
        acceptable = numpy.logical_and(numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                                       dofs >= self.min_dof)

        # if numpy.sum(acceptable) == 0:
        #     print("##############################################################")
        #     print("No acceptable fits found!")
        #     return None

        # Find the most extreme values of the parameter estimates that are deemed acceptable
        sys_sigmas = numpy.zeros(n_params)

        self.param_histogram = numpy.zeros((self.n_params, numpy.sum(acceptable)))

        for i, param in enumerate(self.param_names):
            if numpy.sum(acceptable) == 0:
                sys_sigmas[i] = 0
                continue

            param_small = params[..., i]
            minimum = numpy.min(param_small[acceptable])
            maximum = numpy.max(param_small[acceptable])

            self.param_histogram[i] = param_small[acceptable]

            # Define the systematic error bar by the largest deviation from the central fit by an
            # acceptable fit
            sys_sigmas[i] = max(maximum - params[best_Bbar_index, best, i],
                                params[best_Bbar_index, best, i] - minimum)

            print(f"{param} = {params_central[i]} +- {sys_sigmas[i]}")

        # Find the systematic variation in the critical mass
        if self.model.__name__ == "model1_1a" or self.model.__name__ == "model1_2a":
            g = 0.1
            m_c = mPT_1loop(g, self.N) + g ** 2 * (params[best_Bbar_index, best, 0] -
                                            params[best_Bbar_index, best, -2] * K1(g, self.N))

            if self.model.__name__ == "model1_2a":
                print("Critical Mass using alpha1:")

            print(f"m_c = {m_c}")

            alphas = params[..., 0]
            betas = params[..., -2]

            # Only include parameter estimates from those fits that are acceptable
            alphas = alphas[acceptable]
            betas = betas[acceptable]

            m_cs = mPT_1loop(g, self.N) + g ** 2 * (alphas - betas * K1(g, self.N))

            minimum_m = numpy.min(m_cs)
            maximum_m = numpy.max(m_cs)

            m_c_error = max(m_c - minimum_m, maximum_m - m_c)
            print(f"m_c_error = {m_c_error}")

            self.m_c = m_c
            self.m_c_error = m_c_error

        self.params_central = params_central
        self.sigmas_systematic = sys_sigmas

        if self.model == model1_2a:
            alphas2 = params[..., 1]

            # Only include parameter estimates from those fits that are acceptable
            alphas2 = alphas2[acceptable]

            # Calculate using alpha2
            m_c2 = mPT_1loop(g, self.N) + g ** 2 * (params[best_Bbar_index, best, 1] -
                    params[best_Bbar_index, best, -2] * K1(g, self.N))

            print("Critical Mass using alpha2:")
            print(f"m_c2 = {m_c2}")

            m_c2s = mPT_1loop(g, self.N) + g ** 2 * (alphas2 - betas * K1(g, self.N))

            minimum_m2 = numpy.min(m_c2s)
            maximum_m2 = numpy.max(m_c2s)

            m_c2_error = max(m_c2 - minimum_m2, maximum_m2 - m_c2)
            print(f"m_c2_error = {m_c2_error}")

            # Get the overall systematic error accounting for both alphas
            m_c_error_overall = max(max(m_c - minimum_m, maximum_m - m_c),
                                    max(m_c - minimum_m2, maximum_m2 - m_c))

            print("Overall result when accounting for both alphas:")
            print(f"m_c = {m_c} +- {m_c_error_overall}")

            self.m_c1 = m_c
            self.m_c_error1 = m_c_error
            self.m_c2 = m_c2
            self.m_c_error2 = m_c2_error
            self.m_c_error = m_c_error_overall

    def get_statistical_errors(self):
        # Select the relevent subset of the data
        Bbar_1, Bbar_2 = self.Bbars_central
        keep = numpy.logical_or(self.B_s == Bbar_1, self.B_s == Bbar_2)
        keep = numpy.logical_and(keep, self.g_s * self.L_s > self.gL_min_central - 10 ** -10)

        samples = self.samples[keep]
        g_s = self.g_s[keep]
        L_s = self.L_s[keep]
        Bbar_s = self.B_s[keep]
        m_s = self.m_s[keep]

        cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
        cov_1_2 = numpy.linalg.cholesky(cov_matrix)
        cov_inv = numpy.linalg.inv(cov_1_2)

        # Run with all the bootstrap samples
        param_estimates = numpy.zeros((self.no_samples, self.n_params))

        for i in range(self.no_samples):
            m_s = samples[:, i]

            res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

            res = run_minimizer(self.N, m_s, g_s, L_s, Bbar_s, cov_inv, self.x0, self.model,
                                self.minimization_method, res_function)

            param_estimates[i] = numpy.array(res.x)

        sigmas = numpy.zeros(self.n_params)

        for i, param in enumerate(self.param_names):
            # Find the standard deviation with respect to the central values
            sigmas[i] = numpy.sqrt(numpy.sum((param_estimates[:, i] - self.params_central[i]) ** 2)
                                 / self.no_samples)

            print(f"{param} = {self.params_central[i]} +- {sigmas[i]}")

        self.sigmas_statistiacal = sigmas

        # Calculate the value of the non-perterbative critical mass for g = 0.1 and
        # it's statistical error using alpha1 (the effect of changing alpha is captured in the
        # systematic error)
        g = 0.1
        m_c = mPT_1loop(g, self.N) + g ** 2 * (self.params_central[0] - self.params_central[-2]
                                               * K1(g, self.N))
        print(f"m_c = {m_c}")

        alphas = param_estimates[..., 0]
        betas = param_estimates[..., -2]

        m_cs = mPT_1loop(g, self.N) + g ** 2 * (alphas - betas * K1(g, self.N))
        m_c_error = numpy.sqrt(numpy.sum((m_cs - m_c) ** 2) / self.no_samples)

        print(f"m_c_error = {m_c_error}")
        self.m_c_error_statistical = m_c_error

    def fit_all_data(self):
        pvalues = numpy.zeros((len(self.gL_mins)))
        params = numpy.zeros((len(self.gL_mins), self.n_params))
        dofs = numpy.zeros((len(self.gL_mins)))

        for j, gL_min in enumerate(self.gL_mins):
            print(f'gL = {gL_min}')
            # Select the relevent subset of the data
            keep = self.g_s * self.L_s > gL_min - 10 ** -10

            # Check if there are enough datapoints to do the fit
            if sum(keep) - self.n_params < 2:
                continue

            samples = self.samples[keep]
            g_s = self.g_s[keep]
            L_s = self.L_s[keep]
            Bbar_s = self.B_s[keep]
            m_s = self.m_s[keep]

            cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
            cov_1_2 = numpy.linalg.cholesky(cov_matrix)
            cov_inv = numpy.linalg.inv(cov_1_2)

            res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

            res = run_minimizer(self.N, m_s, g_s, L_s, Bbar_s, cov_inv, self.x0, self.model,
                                self.minimization_method, res_function)

            chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
            n_params = len(res.x)
            dof = g_s.shape[0] - n_params
            p = chisq_pvalue(dof, chisq)
            dofs[j] = dof
            pvalues[j] = p
            params[j] = res.x

            string = f"{seperator}\n"
            string += f"Config: N = {self.N}, gL_min = {gL_min:.2f}\n"
            string += f"chisq = {chisq}\n"
            string += f"chisq/dof = {chisq / dof}\n"
            string += f"pvalue = {p}\n"
            string += f"dof = {dof}\n"
            string += f"params = {res.x}\n"

            print(string)

            # if p > 0.05 and print_string:
            #     print(string)
            #     print_string = False

    def get_systematic_errors_1B(self):
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
        Bbar_list = list(numpy.sort(list(set(self.B_s))))
        pvalues = numpy.zeros((len(Bbar_list), len(self.gL_mins)))
        params = numpy.zeros((len(Bbar_list), len(self.gL_mins), self.n_params))
        dofs = numpy.zeros((len(Bbar_list), len(self.gL_mins)))

        Bbar_fits = []
        best_fits = []

        for i, Bbar in enumerate(Bbar_list):
            print_string = True
            string = ""

            for j, gL_min in enumerate(self.gL_mins):
                # Select the relevent subset of the data
                keep = numpy.logical_and(self.B_s == Bbar, self.g_s * self.L_s > gL_min - 10 ** -10)

                # Check if there are enough datapoints to do the fit
                if sum(keep) - self.n_params < 2:
                    continue

                samples = self.samples[keep]
                g_s = self.g_s[keep]
                L_s = self.L_s[keep]
                Bbar_s = self.B_s[keep]
                m_s = self.m_s[keep]

                cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
                cov_1_2 = numpy.linalg.cholesky(cov_matrix)
                cov_inv = numpy.linalg.inv(cov_1_2)

                res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

                res = run_minimizer(self.N, m_s, g_s, L_s, Bbar_s, cov_inv, self.x0, self.model,
                                    self.minimization_method, res_function)

                chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
                n_params = len(res.x)
                dof = g_s.shape[0] - n_params
                p = chisq_pvalue(dof, chisq)
                dofs[i, j] = dof
                pvalues[i, j] = p
                params[i, j] = res.x

                string = f"{seperator}\n"
                string += f"Config: N = {self.N}, Bbar = {Bbar}, gL_min = {gL_min:.2f}\n"
                string += f"chisq = {chisq}\n"
                string += f"chisq/dof = {chisq / dof}\n"
                string += f"pvalue = {p}\n"
                string += f"dof = {dof}\n"
                string += f"params = {res.x}\n"

                if p > 0.05 and print_string:
                    print(string)
                    print_string = False
                    best_fits.append(res.x)
                    Bbar_fits.append(Bbar)

        # Extract the index of the smallest gL_min fit that has an acceptable p-value
        r = len(self.gL_mins)
        best = r - 1

        for i, gL_min in enumerate(self.gL_mins):
            if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
                best = r - 1 - i

        best_Bbar_index = numpy.argmax(pvalues[:, best])
        best_Bbar = Bbar_list[best_Bbar_index]
        self.gL_min_central = self.gL_mins[best]
        self.Bbars_central = best_Bbar
        self.p = pvalues[best_Bbar_index, best]

        print("##################################################################")
        print("BEST RESULT")
        print(f"Bbar_s = {best_Bbar}")
        print(f"gL_min = {self.gL_mins[best]:.2f}")
        print(f"pvalue : {pvalues[best_Bbar_index, best]}")
        print(f"dof : {dofs[best_Bbar_index, best]}")
        print("##################################################################")

        params_central = params[best_Bbar_index, best]

        # Find the parameter variation over acceptable fits
        acceptable = numpy.logical_and(numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                                       dofs >= self.min_dof)

        # if numpy.sum(acceptable) == 0:
        #     print("##############################################################")
        #     print("No acceptable fits found!")
        #     return None

        # Find the most extreme values of the parameter estimates that are deemed acceptable
        sys_sigmas = numpy.zeros(n_params)

        self.param_histogram = numpy.zeros((self.n_params, numpy.sum(acceptable)))

        for i, param in enumerate(self.param_names):
            if numpy.sum(acceptable) == 0:
                sys_sigmas[i] = 0
                continue

            param_small = params[..., i]
            minimum = numpy.min(param_small[acceptable])
            maximum = numpy.max(param_small[acceptable])

            self.param_histogram[i] = param_small[acceptable]

            # Define the systematic error bar by the largest deviation from the central fit by an
            # acceptable fit
            sys_sigmas[i] = max(maximum - params[best_Bbar_index, best, i],
                                params[best_Bbar_index, best, i] - minimum)

            print(f"{param} = {params_central[i]} +- {sys_sigmas[i]}")

        # Find the systematic variation in the critical mass
        if self.model.__name__ == "model1_1a" or self.model.__name__ == "model1_2a":
            g = 0.1
            m_c = mPT_1loop(g, self.N) + g ** 2 * (params[best_Bbar_index, best, 0] -
                                            params[best_Bbar_index, best, -2] * K1(g, self.N))

            if self.model.__name__ == "model1_2a":
                print("Critical Mass using alpha1:")

            print(f"m_c = {m_c}")

            alphas = params[..., 0]
            betas = params[..., -2]

            # Only include parameter estimates from those fits that are acceptable
            alphas = alphas[acceptable]
            betas = betas[acceptable]

            m_cs = mPT_1loop(g, self.N) + g ** 2 * (alphas - betas * K1(g, self.N))

            minimum_m = numpy.min(m_cs)
            maximum_m = numpy.max(m_cs)

            m_c_error = max(m_c - minimum_m, maximum_m - m_c)
            print(f"m_c_error = {m_c_error}")

            self.m_c = m_c
            self.m_c_error = m_c_error

        self.params_central = params_central
        self.sigmas_systematic = sys_sigmas

        if self.model == model1_2a:
            alphas2 = params[..., 1]

            # Only include parameter estimates from those fits that are acceptable
            alphas2 = alphas2[acceptable]

            # Calculate using alpha2
            m_c2 = mPT_1loop(g, self.N) + g ** 2 * (params[best_Bbar_index, best, 1] -
                    params[best_Bbar_index, best, -2] * K1(g, self.N))

            print("Critical Mass using alpha2:")
            print(f"m_c2 = {m_c2}")

            m_c2s = mPT_1loop(g, self.N) + g ** 2 * (alphas2 - betas * K1(g, self.N))

            minimum_m2 = numpy.min(m_c2s)
            maximum_m2 = numpy.max(m_c2s)

            m_c2_error = max(m_c2 - minimum_m2, maximum_m2 - m_c2)
            print(f"m_c2_error = {m_c2_error}")

            # Get the overall systematic error accounting for both alphas
            m_c_error_overall = max(max(m_c - minimum_m, maximum_m - m_c),
                                    max(m_c - minimum_m2, maximum_m2 - m_c))

            print("Overall result when accounting for both alphas:")
            print(f"m_c = {m_c} +- {m_c_error_overall}")

            self.m_c1 = m_c
            self.m_c_error1 = m_c_error
            self.m_c2 = m_c2
            self.m_c_error2 = m_c2_error
            self.m_c_error = m_c_error_overall

        return Bbar_fits, best_fits

    def all_Bbar_one_g(self):
        pvalues = numpy.zeros((len(set(self.g_s)), len(self.gL_mins)))
        params = numpy.zeros((len(set(self.g_s)), len(self.gL_mins), self.n_params))
        dofs = numpy.zeros((len(set(self.g_s)), len(self.gL_mins)))

        L_s_list = numpy.sort(list(set(self.L_s)))

        for i, g in enumerate(numpy.sort(list(set(self.g_s)))):
            print(f"Running fits with g = {g}")

            for j, L_min in enumerate(L_s_list):
                # Select the relevent subset of the data
                keep = numpy.logical_and(self.g_s == g, self.L_s > L_min - 10 ** -10)
                Bbar_s_list = numpy.sort(list(set(self.B_s)))[0: num_Bbars]
                keep = numpy.logical_and(keep, numpy.isin(self.B_s, Bbar_s_list))

                # Check if there are enough datapoints to do the fit
                if sum(keep) - self.n_params < 2:
                    continue

                samples = self.samples[keep]
                g_s = self.g_s[keep]
                L_s = self.L_s[keep]
                Bbar_s = self.B_s[keep]
                m_s = self.m_s[keep]

                cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
                cov_1_2 = numpy.linalg.cholesky(cov_matrix)
                cov_inv = numpy.linalg.inv(cov_1_2)

                res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

                res = run_minimizer(self.N, m_s, g_s, L_s, Bbar_s, cov_inv, self.x0, self.model,
                                    self.minimization_method, res_function)

                chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
                n_params = len(res.x)
                dof = g_s.shape[0] - n_params
                p = chisq_pvalue(dof, chisq)
                # dofs[i, j] = dof
                # pvalues[i, j] = p
                # params[i, j] = res.x

                print(seperator)
                print(f"Bbars = {Bbar_s_list}")
                print(f"Config: N = {self.N}, g = {g}, L_min = {L_min}")
                print(f"chisq = {chisq}")
                print(f"chisq/dof = {chisq / dof}")
                print(f"pvalue = {p}")
                print(f"dof = {dof}")
                print(f"params = {res.x}")

        return None
        # Extract the index of the smallest gL_min fit that has an acceptable p-value
        r = len(self.gL_mins)
        best = r - 1

        for i, gL_min in enumerate(self.gL_mins):
            if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
                best = r - 1 - i

        best_Bbar_index = numpy.argmax(pvalues[:, best])
        best_Bbar = Bbar_list[best_Bbar_index]
        self.gL_min_central = self.gL_mins[best]
        self.Bbars_central = best_Bbar
        self.p = pvalues[best_Bbar_index, best]

        print("##################################################################")
        print("BEST RESULT")
        print(f"Bbar_s = {best_Bbar}")
        print(f"gL_min = {self.gL_mins[best]:.2f}")
        print(f"pvalue : {pvalues[best_Bbar_index, best]}")
        print(f"dof : {dofs[best_Bbar_index, best]}")
        print("##################################################################")

        params_central = params[best_Bbar_index, best]

        # Find the parameter variation over acceptable fits
        acceptable = numpy.logical_and(numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                                       dofs >= self.min_dof)

        # if numpy.sum(acceptable) == 0:
        #     print("##############################################################")
        #     print("No acceptable fits found!")
        #     return None

        # Find the most extreme values of the parameter estimates that are deemed acceptable
        sys_sigmas = numpy.zeros(n_params)

        self.param_histogram = numpy.zeros((self.n_params, numpy.sum(acceptable)))

        for i, param in enumerate(self.param_names):
            if numpy.sum(acceptable) == 0:
                sys_sigmas[i] = 0
                continue

            param_small = params[..., i]
            minimum = numpy.min(param_small[acceptable])
            maximum = numpy.max(param_small[acceptable])

            self.param_histogram[i] = param_small[acceptable]

            # Define the systematic error bar by the largest deviation from the central fit by an
            # acceptable fit
            sys_sigmas[i] = max(maximum - params[best_Bbar_index, best, i],
                                params[best_Bbar_index, best, i] - minimum)

            print(f"{param} = {params_central[i]} +- {sys_sigmas[i]}")

    def get_pvalues(self):
        Bbar_1, Bbar_2 = self.Bbars_central

        self.pvalues = []

        for i, gL_min in enumerate(self.gL_mins):
            keep = numpy.logical_or(self.B_s == Bbar_1, self.B_s == Bbar_2)
            keep = numpy.logical_and(keep, self.g_s * self.L_s > gL_min - 10 ** -10)

            # Check if there are enough datapoints to do the fit
            if sum(keep) - self.n_params < 2:
                continue

            samples = self.samples[keep]
            g_s = self.g_s[keep]
            L_s = self.L_s[keep]
            Bbar_s = self.B_s[keep]
            m_s = self.m_s[keep]

            cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
            cov_1_2 = numpy.linalg.cholesky(cov_matrix)
            cov_inv = numpy.linalg.inv(cov_1_2)

            res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

            res = run_minimizer(self.N, m_s, g_s, L_s, Bbar_s, cov_inv, self.x0, self.model,
                                self.minimization_method, res_function)

            chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
            n_params = len(res.x)
            dof = g_s.shape[0] - n_params
            p = chisq_pvalue(dof, chisq)

            self.pvalues.append(p)

    def plot_fit(self):
        colors = ['b', 'r', 'k', 'g', 'yellow', 'purple']
        # Bbar_1, Bbar_2 = self.Bbars_central

        # keep = numpy.logical_or(self.B_s == Bbar_1, self.B_s == Bbar_2)
        # keep = numpy.logical_and(keep, self.g_s * self.L_s > self.gL_min_central - 10 ** -10)
        keep = numpy.full_like(self.g_s, True, dtype=bool)

        samples = self.samples[keep]
        g_s = self.g_s[keep]
        L_s = self.L_s[keep]
        Bbar_s = self.B_s[keep]
        m_s = self.m_s[keep]

        gL_s = g_s * L_s
        # nu = self.params_central[-1]
        nu = 0.85
        x_scale = gL_s ** (-1 / nu)

        x_max = max(x_scale)
        x_min = min(x_scale)

        for i, g in enumerate(list(set(g_s))):
            plt.errorbar(x_scale[g_s == g], (m_s / g_s)[g_s == g],
                numpy.std(samples[g_s == g], axis=1), ls='', color=colors[i], marker='o',
                markerfacecolor='none')

            # m_c = mPT_1loop(g, self.N) + g ** 2 * (self.params_central[0] - self.params_central[-2]
            #                                     * K1(g, self.N))

            # plt.plot([x_min, x_max], [m_c / g, ] * 2, color=colors[i], ls='--')

            # for j, Bbar in enumerate(self.Bbars_central):
            #     L_min = x_max ** - nu / g
            #     L_max = x_min ** - nu / g
            #     L_s = numpy.linspace(L_min, L_max, 1000)
            #     x_scale_ = (L_s * g) ** (-1 / self.params_central[-1])

            #     m_cs = self.model(self.N, g, L_s, Bbar, *self.params_central)

            #     if j == 0:
            #         plt.plot(x_scale_, m_cs / g, label=f'ag = {g}', color=colors[i])

            #     else:
            #         plt.plot(x_scale_, m_cs / g, color=colors[i])

        plt.xlabel(r'$1 / (gL)^{1/\nu}$')
        plt.ylabel(r'$a\bar{m}^2(g, L) / g$')
        plt.legend()
        plt.show()

    def plot_gradient(self):
        B0 = self.B_list[0]

        for g in self.g_list:
            for L in self.L_list:
                # Do first the reference point
                B1 = self.B_list[0]
                B2 = self.B_list[1]
                high_index = numpy.logical_and(numpy.logical_and(self.g_s == g, self.L_s == L),
                                                self.B_s == B2)
                low_index = numpy.logical_and(numpy.logical_and(self.g_s == g, self.L_s == L),
                                                self.B_s == B1)

                dm_0 = self.samples[high_index] - self.samples[low_index]
                dB_0 = B2 - B1

                means = []
                stds = []

                for i in range(1, len(self.B_list[:-1])):
                    B1 = self.B_list[i]
                    B2 = self.B_list[i + 1]
                    high_index = numpy.logical_and(numpy.logical_and(self.g_s == g, self.L_s == L),
                                                   self.B_s == B2)

                    low_index = numpy.logical_and(numpy.logical_and(self.g_s == g, self.L_s == L),
                                                   self.B_s == B1)

                    dm = self.samples[high_index] - self.samples[low_index]
                    dB = B2 - B1

                    ratio = (dB / dm) / (dB_0 / dm_0)
                    stds.append(numpy.std(ratio))
                    means.append(numpy.mean(ratio))

                means = numpy.array(means)
                stds = numpy.array(stds)

                plt.fill_between(self.B_list[1: -1], means - stds, means + stds, label=f'L = {L}',
                                 alpha=0.2)

            plt.legend()
            plt.title(f'g = {g}')
            plt.show()



a = scaling_analysis(5, "h5data/width/width_0.5.h5", model=simple_1_1)
a.get_systematic_errors_1B()


def print_results(width, N, model):
    a = scaling_analysis(N, f"h5data/width/width_{width:.1f}.h5", model=model)

    a.get_systematic_errors()
    a.get_statistical_errors()

    print(f'{width:.1f}'.ljust(15), end='')

    for i in range(a.n_params):
        central = a.params_central[i]
        sys_sigma = a.sigmas_systematic[i]
        stat_sigma = a.sigmas_statistiacal[i]

        digit_stat = -int(numpy.floor(numpy.log10(stat_sigma)))
        digit_central = -int(numpy.floor(numpy.log10(numpy.abs(central))))

        if sys_sigma == 0:
            sys = '0'
            limit = digit_stat

        else:
            digit_sys = -int(numpy.floor(numpy.log10(sys_sigma)))
            limit = max(digit_stat, digit_sys)
            sys = str(round(sys_sigma, limit)).lstrip('0').lstrip('.').lstrip('0')

        stat = str(round(stat_sigma, limit)).lstrip('0').lstrip('.').lstrip('0')

        central = f'{round(central, limit):.20f}'[:limit + 2]

        print(f'{central}({stat})({sys})'.ljust(15), end='')

    print(f'{a.p:.2f}'.ljust(15), end='')
    print(f'{a.Bbars_central}'.ljust(15), end='')
    print(f'{a.gL_min_central}'.ljust(15))


def run(model, N, file=f"h5data/width/width_0.0.h5"):
    print(f'Running for model : {model.__name__}')

    a = scaling_analysis(N, file, model=model)
    print('width'.ljust(15), end='')

    for param in a.param_names:
        print(f'{param}'.ljust(15), end='')

    print('p'.ljust(15), end='')
    print('Bbar_s'.ljust(15), end='')
    print('gL_min'.ljust(15))

    for width in numpy.arange(0, 11) / 10:
        try:
            print_results(width, N, model)

        except Exception:
            continue


def param_histograms(model, N, file):
    a = scaling_analysis(N, file, model=model)
    a.get_systematic_errors()

    for i, param in enumerate(a.param_names):
        plt.hist(a.param_histogram[i])
        plt.title(param)
        plt.show()


def scan_omega_2(omega_value):
    def simple_2_1(N, g, L, Bbar, alpha, a, c, lambduh, nu):
        omega = omega_value

        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** (omega - 2 / nu)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling - lambduh * K1(g, N))

    def fixed_correction(N, g, L, Bbar, a, c, lambduh, nu):
        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** -omega_value

        return mPT_1loop(g, N) + g ** 2 * (Scaling - lambduh * K1(g, N))

    a = scaling_analysis(5, "h5data/width/width_0.5.h5", fixed_correction, min_dof=15)
    # a.plot_gradient()
    # a.plot_fit()
    # a.all_Bbar_one_g()
    # a.fit_all_data()
    # a.get_systematic_errors()
    Bbar_fits, best_fits = a.get_systematic_errors_1B()
    best_fits = numpy.array(best_fits)
    print(f'omega={omega_value}, std of nu: {numpy.std(best_fits[:, -1])}')

    return best_fits[:, -1]


def fit_omega(omega):
    def fixed_correction(N, g, L, Bbar, a, c, lambduh, nu):
        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** -omega_value

        return mPT_1loop(g, N) + g ** 2 * (Scaling - lambduh * K1(g, N))

nus = []
for omega in numpy.linspace(1.2, 2, 9):
    nu = scan_omega_2(omega)
    nus.append(nu)
    plt.plot(nu, label=f'o={omega}')

plt.legend()
plt.show()
nus = numpy.array(nus)


# param_histograms(model1_1a, 3, "h5data/width/width_0.4.h5")
sleep(1000)


def pvalue_plot(N, model1, model2):
    fig, ax = plt.subplots()
    widths = numpy.arange(5) / 5

    for width in widths:
        print(f'Running for width: {width:.1f}')

        a = scaling_analysis(N, f"h5data/width/width_{width:.1f}.h5", model=model1)

        a.get_systematic_errors()
        a.get_pvalues()

        pvalues_1 = a.pvalues

        ax.scatter(a.gL_mins[:len(pvalues_1)], pvalues_1, marker='o',
                   color=((1 - width) / max(1 - widths), 0, width / max(widths)),
                   label=f'log(g), w={width:.1f}', facecolors='none')

    for width in widths:
        print(f'Running for width: {width:.1f}')

        a = scaling_analysis(N, f"h5data/width/width_{width:.1f}.h5", model=model2)

        a.get_systematic_errors()
        a.get_pvalues()

        pvalues_2 = a.pvalues

        ax.scatter(a.gL_mins[:len(pvalues_2)], pvalues_2, marker='^',
                   color=((1 - width) / max(1 - widths), 0, width / max(widths)),
                   label=f'log(L), w={width:.1f}', facecolors='none')

    plt.legend()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(f'Local/graphs/pvalues_N{N}.pdf')


for N in [2, 3, 4, 5]:
    for model in [model1_1a, model1_2a]:
        print(f'N = {N}, model={model.__name__}')
        run(model, N, "h5data/width/width_0.4.h5")
        print('')

# pvalue_plot(2, model1_1a, model2_1a)
# pvalue_plot(4, model1_2a, model2_2a)
