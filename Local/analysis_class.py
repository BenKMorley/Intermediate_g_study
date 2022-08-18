#!/usr/bin/env python3
import json
from typing import Optional, List, Tuple
from multiprocessing import current_process
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
import sys
import os
import re
import numpy
from tqdm import tqdm
from scipy.optimize import minimize, least_squares
import pymultinest
from getdist import loadMCSamples, plots_edit

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.parameters import param_dict, seperator, prior_range_dict_maker, \
                            x0_dict, param_names_latex
from Core.bayesian_functions import likelihood_maker, prior_maker
from Core.model_definitions import *


def make_function(function_name, omega, eps, model_type, A=4 * numpy.pi):
    def modelA(N, g, L, Bbar, alpha, a1, a2, beta, nu):
        a = numpy.zeros_like(g)

        a_s = (a1, a2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            a[Bbar == Bbar_list[i]] = a_s[i]

        Scaling = a * (g * L) ** (-1 / nu)

        if model_type == 1:
            PT = -beta * K1(g, N)

        if model_type == 2:
            PT = -beta * K2(L, N)

        if model_type == 3:
            pref = ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)

            PT = -beta * (numpy.where((g * L) > A * N,
                            K1(g, N) - pref * numpy.log(A / (4 * numpy.pi)),
                            K2(L, N)))

            # Run a test to make sure this is okay
            L = 16
            assert abs(K1(A * N / L, N) - pref * numpy.log(A / (4 * numpy.pi)) - K2(L, N)) < 10 ** -10

        # PT without the g^2 term
        if model_type == 4:
            PT = -beta * K2(L, N) + alpha * (g * L)

            return mPT_1loop(g, N) + g ** 2 * (Scaling + PT)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelB(N, g, L, Bbar, alpha, a1, a2, c1, c2, beta, nu):
        a = numpy.zeros_like(g)
        c = numpy.zeros_like(g)

        a_s = (a1, a2)
        c_s = (c1, c2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            a[Bbar == Bbar_list[i]] = a_s[i]
            c[Bbar == Bbar_list[i]] = c_s[i]

        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** -(omega + 1 / nu)

        if model_type == 1:
            PT = -beta * K1(g, N)

        if model_type == 2:
            PT = -beta * K2(L, N)

        if model_type == 3:
            pref = ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)

            PT = -beta * (numpy.where((g * L) > A * N,
                            K1(g, N) - pref * numpy.log(A / (4 * numpy.pi)),
                            K2(L, N)))

            # Run a test to make sure this is okay
            L = 16
            assert abs(K1(A * N / L, N) - pref * numpy.log(A / (4 * numpy.pi)) - K2(L, N)) < 10 ** -10

        # PT without the g^2 term
        if model_type == 4:
            PT = -beta * K2(L, N) + alpha * (g * L)

            return mPT_1loop(g, N) + g ** 2 * (Scaling + PT)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelC(N, g, L, Bbar, alpha, a1, a2, c1, c2, e1, e2, beta, nu):
        a = numpy.zeros_like(g)
        c = numpy.zeros_like(g)
        e = numpy.zeros_like(g)

        a_s = (a1, a2)
        c_s = (c1, c2)
        e_s = (e1, e2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            a[Bbar == Bbar_list[i]] = a_s[i]
            c[Bbar == Bbar_list[i]] = c_s[i]
            e[Bbar == Bbar_list[i]] = e_s[i]

        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** -(omega + 1 / nu) +\
                e * (g * L) ** -(eps + 1 / nu)

        if model_type == 1:
            PT = -beta * K1(g, N)

        if model_type == 2:
            PT = -beta * K2(L, N)

        if model_type == 3:
            pref = ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)

            PT = -beta * (numpy.where((g * L) > A * N,
                            K1(g, N) - pref * numpy.log(A / (4 * numpy.pi)),
                            K2(L, N)))

            # Run a test to make sure this is okay
            L = 16
            assert abs(K1(A * N / L, N) - pref * numpy.log(A / (4 * numpy.pi)) - K2(L, N)) < 10 ** -10

        # PT without the g^2 term
        if model_type == 4:
            PT = -beta * K2(L, N) + alpha * (g * L)

            return mPT_1loop(g, N) + g ** 2 * (Scaling + PT)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelD(N, g, L, Bbar, alpha1, alpha2, a1, a2, beta, nu):
        a = numpy.zeros_like(g)
        alpha = numpy.zeros_like(g)

        a_s = (a1, a2)
        alpha_s = (alpha1, alpha2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            alpha[Bbar == Bbar_list[i]] = alpha_s[i]
            a[Bbar == Bbar_list[i]] = a_s[i]

        Scaling = a * (g * L) ** (-1 / nu)

        if model_type == 1:
            PT = -beta * K1(g, N)

        if model_type == 2:
            PT = -beta * K2(L, N)

        if model_type == 3:
            pref = ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)

            PT = -beta * (numpy.where((g * L) > A * N,
                            K1(g, N) - pref * numpy.log(A / (4 * numpy.pi)),
                            K2(L, N)))

            # Run a test to make sure this is okay
            L = 16
            assert abs(K1(A * N / L, N) - pref * numpy.log(A / (4 * numpy.pi)) - K2(L, N)) < 10 ** -10

        # PT without the g^2 term
        if model_type == 4:
            PT = -beta * K2(L, N) + alpha * (g * L)

            return mPT_1loop(g, N) + g ** 2 * (Scaling + PT)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelGA(N, g, L, Bbar, alpha, a1, a2, beta1, beta2, nu):
        a = numpy.zeros_like(g)

        a_s = (a1, a2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            a[Bbar == Bbar_list[i]] = a_s[i]

        Scaling = a * (g * L) ** (-1 / nu)

        if model_type == 1:
            PT = -beta1 * K1(g, N)

        if model_type == 2:
            PT = -beta2 * K2(L, N)

        # lambda1 ~ [1, 1.2], lambda2 ~ [1.3, 1.5]
        # log(g) ~ [1.3, 1.5], log(L) ~ [-0.3, -0.1]
        if model_type == 3:
            PT = - beta1 * K1(g, N) - beta2 * K2(L, N)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelGB(N, g, L, Bbar, alpha, a1, a2, c1, c2, beta1, beta2, nu):
        a = numpy.zeros_like(g)
        c = numpy.zeros_like(g)

        a_s = (a1, a2)
        c_s = (c1, c2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            a[Bbar == Bbar_list[i]] = a_s[i]
            c[Bbar == Bbar_list[i]] = c_s[i]

        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** -(omega + 1 / nu)

        if model_type == 1:
            PT = -beta1 * K1(g, N)

        if model_type == 2:
            PT = -beta2 * K2(L, N)

        if model_type == 3:
            PT = - beta1 * K1(g, N) - beta2 * K2(L, N)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelGC(N, g, L, Bbar, alpha, a1, a2, c1, c2, e1, e2, beta1, beta2, nu):
        a = numpy.zeros_like(g)
        c = numpy.zeros_like(g)
        e = numpy.zeros_like(g)

        a_s = (a1, a2)
        c_s = (c1, c2)
        e_s = (e1, e2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            a[Bbar == Bbar_list[i]] = a_s[i]
            c[Bbar == Bbar_list[i]] = c_s[i]
            e[Bbar == Bbar_list[i]] = e_s[i]

        Scaling = a * (g * L) ** (-1 / nu) + c * (g * L) ** -(omega + 1 / nu) +\
                e * (g * L) ** -(eps + 1 / nu)

        if model_type == 1:
            PT = -beta1 * K1(g, N)

        if model_type == 2:
            PT = -beta2 * K2(L, N)

        if model_type == 3:
            PT = - beta1 * K1(g, N) - beta2 * K2(L, N)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    def modelGD(N, g, L, Bbar, alpha1, alpha2, a1, a2, beta1, beta2, nu):
        a = numpy.zeros_like(g)
        alpha = numpy.zeros_like(g)

        a_s = (a1, a2)
        alpha_s = (alpha1, alpha2)

        Bbar_list = list(numpy.sort(list(set(list(Bbar)))))

        for i in range(len(Bbar_list)):
            alpha[Bbar == Bbar_list[i]] = alpha_s[i]
            a[Bbar == Bbar_list[i]] = a_s[i]

        Scaling = a * (g * L) ** (-1 / nu)

        if model_type == 1:
            PT = -beta1 * K1(g, N)

        if model_type == 2:
            PT = -beta2 * K2(L, N)

        if model_type == 3:
            PT = - beta1 * K1(g, N) - beta2 * K2(L, N)

        return mPT_1loop(g, N) + g ** 2 * (alpha + Scaling + PT)

    if function_name == "A":
        return modelA

    if function_name == "B":
        return modelB

    if function_name == "C":
        return modelC

    if function_name == "D":
        return modelD

    if function_name == "GA":
        return modelGA

    if function_name == "GB":
        return modelGB

    if function_name == "GC":
        return modelGC

    if function_name == "GD":
        return modelGD


def run_minimizer(cov_inv, x0, model, method, res_function):
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


def color_gs(g):
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


class analysis:
    def __init__(self, N, model_name, x0_dict=x0_dict, omega=0.8, eps=2, A=4 * numpy.pi,
                 model_type=1, width=0.5, filename=None, no_samples=200, gL_max=numpy.inf,
                 bayes_points=400, scale=10):
        self.width = width
        if filename is None:
            filename = f'h5data/width/width_{width:.1f}.h5'

        self.omega = omega
        self.eps = eps
        self.p_min = 0.05  # Minimum acceptable p-value
        self.min_dof = 2
        self.evidence = None
        self.model_name = model_name
        self.model_type = model_type
        self.model = make_function(model_name, omega, eps, model_type, A=A)
        self.filename = filename
        self.N = N
        self.x0_dict = x0_dict
        self.minimization_method = "lm"
        self.nu_max = 10

        arg_count = self.model.__code__.co_argcount
        self.param_names = self.model.__code__.co_varnames[4:arg_count]

        self.x0 = [x0_dict[param] for param in self.param_names]

        self.scale = scale
        prior_ranges = prior_range_dict_maker(self.scale)
        self.prior_ranges = [prior_ranges[param] for param in self.param_names]
        self.param_names_latex = [param_names_latex[param] for param in self.param_names]
        self.prior = prior_maker(self.prior_ranges)

        self.no_samples = no_samples
        self.n_params = len(self.param_names)
        self.load_data()
        self.Bbar_list = numpy.sort(numpy.array(list(set(self.B_s))))
        self.Bbar_pieces = self.Bbar_list[:-1]

        self.frozen = True

        # Use 2 d.o.f. minimum
        self.gL_mins, self.counts = numpy.unique(numpy.round_(self.g_s * self.L_s, 1),
                                            return_counts=True)
        self.counts = self.counts / len(self.Bbar_list)

        # If the data is complete the counts should be integers
        assert numpy.max(numpy.abs((self.counts - numpy.round(self.counts, 0)))) < 10 ** -10

        self.dofs = 2 * numpy.cumsum(self.counts[::-1])[::-1] - self.n_params

        self.gL_mins = self.gL_mins[self.dofs >= 2]
        self.counts = self.counts[self.dofs >= 2]
        self.dofs = self.dofs[self.dofs >= 2]
        self.dof_matrix = self.dofs.reshape((1, len(self.gL_mins))).repeat(len(self.Bbar_pieces),
                                            axis=0)

        # Set a random seed - we want this to be the same for all Bbar
        numpy.random.seed(584523)

        # Generate random numbers - want different for each gL_min
        self.bootstraps = numpy.random.randint(0, self.samples.shape[-1],
                                size=(len(self.gL_mins), self.no_samples, self.samples.shape[-1]))

        # BMM means and variances
        self.mean_pieces = {}
        self.var_pieces = {}
        self.mean_pieces_gL = {}
        self.var_pieces_gL = {}
        self.mean = None
        self.var = None

        # For the purpose of saving data
        self.metadata = f"N{N}_model{model_name}{model_type}_w{width}_" + \
                        f"o{omega:.1f}_e{eps:.1f}_A{A:.1f}"

        self.bayes_points = bayes_points

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

    def find_keep(self, Bbar, gL_min, gL_max=numpy.inf):
        # Find the set of Bbar we are using for this fit
        index = numpy.argwhere(self.Bbar_list == Bbar)
        Bbar_s_in = self.Bbar_list[index[0][0]: index[0][0] + 2]

        # Select the relevent subset of the data
        keep = self.g_s * self.L_s > gL_min - 10 ** -10
        keep = numpy.logical_and(keep, self.g_s * self.L_s < gL_max + 10 ** -10)
        keep = numpy.logical_and(keep, numpy.isin(self.B_s, Bbar_s_in))

        return keep

    def find_median_gLmin(self, gL_min):
        i = numpy.argwhere(self.gL_mins == gL_min)[0][0]

        size = sum(self.counts[i:])

        return self.gL_mins[i + numpy.argwhere(numpy.cumsum(self.counts[i:]) > size / 2)[0][0]]

    def find_cov_matrix(self, Bbar, gL_min, gL_max=numpy.inf):
        keep = self.find_keep(Bbar, gL_min, gL_max=gL_max)

        samples = self.samples[keep]
        g_s = self.g_s[keep]
        L_s = self.L_s[keep]
        m_s = self.m_s[keep]

        cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
        cov_1_2 = numpy.linalg.cholesky(cov_matrix)

        # Record this so we can use it repeatedly in the bootstrap
        self.cov_inv = numpy.linalg.inv(cov_1_2)

    def fit(self, Bbar: float, gL_min: float, gL_max: float = numpy.inf, plot: bool = False,
            m_s: Optional[numpy.ndarray] = None, find_cov: bool = True,
            print_info: bool = False, ) -> Tuple[List[float], float, int]:
        if find_cov:
            self.find_cov_matrix(Bbar, gL_min, gL_max=gL_max)

        keep = self.find_keep(Bbar, gL_min, gL_max=gL_max)

        samples = self.samples[keep]
        g_s = self.g_s[keep]
        L_s = self.L_s[keep]
        Bbar_s = self.B_s[keep]

        if m_s is None:
            m_s = self.m_s[keep]

        res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

        res = run_minimizer(self.cov_inv, self.x0, self.model, self.minimization_method,
                            res_function)

        self.model(self.N, g_s, L_s, Bbar_s, *res.x)

        chisq = chisq_calc(res.x, self.cov_inv, self.model, res_function)
        dof = g_s.shape[0] - self.n_params
        p = chisq_pvalue(dof, chisq)

        if print_info:
            string = f"{seperator}\n"
            string += f"N={self.N}, gL_min={gL_min:.2f}, Bbar={Bbar}, type={self.model_type}\n"
            string += f"chisq = {chisq}"
            string += f"chisq/dof = {chisq / dof}\n"
            string += f"pvalue = {p}\n"
            string += f"dof = {dof}\n"
            string += f"params = {res.x}\n"

            print(string)

        if plot:
            for g in list(set(g_s)):
                stds = numpy.sqrt(1 / numpy.sqrt(numpy.diag(self.cov_inv)[g_s == g]))
                p = plt.errorbar(1 / (g * L_s[g_s == g]), m_s[g_s == g] / g, stds, ls='')

                L_s_curve = numpy.linspace(min(L_s), max(L_s), 1000)
                results = self.model(self.N, numpy.ones_like(
                    L_s_curve) * g, L_s_curve, numpy.ones_like(L_s_curve) * Bbar, *res.x)
                plt.plot(1 / (g * L_s_curve), results /
                         g, color=p[0].get_color())

            plt.close()

        return res.x, chisq

    def run_bootstrap(self, Bbar: float, gL_min: float,
                      gL_max: float = numpy.inf) -> numpy.ndarray:
        self.find_cov_matrix(Bbar, gL_min, gL_max=gL_max)
        keep = self.find_keep(Bbar, gL_min, gL_max=gL_max)
        param_results = numpy.zeros((self.no_samples, len(self.param_names)))

        for i in tqdm(range(self.no_samples)):
            m_s = self.samples[keep, i]
            param_results[i] = self.fit(Bbar, gL_min, find_cov=False, m_s=m_s, gL_max=gL_max)[0]

        return param_results

    def plot_fits_all_Bbar(self) -> None:
        Bbar_list = self.Bbar_pieces
        means = numpy.zeros((len(Bbar_list), self.n_params))
        stds = numpy.zeros((len(Bbar_list), self.n_params))
        p_values = numpy.zeros((len(Bbar_list), len(self.gL_mins)))

        for i, Bbar in enumerate(Bbar_list):
            means[i], stds[i], p_values[i] = self.fit_all_gLmin(Bbar)

        # Only include contributions with at least one good fit
        means = means[numpy.max(p_values, axis=1) > 0.05]
        stds = stds[numpy.max(p_values, axis=1) > 0.05]
        Bbar_list = Bbar_list[numpy.max(p_values, axis=1) > 0.05]

        var_s = stds ** 2

        # Find the overall average by doing an inverse-variance weighting
        weights = 1 / var_s

        # Weight the data evenly (e.g. all Bbar's are treated equally)
        mean = numpy.average(means, axis=0)
        var = 1 / numpy.sum(weights, axis=0)

        for i, param in enumerate(self.param_names):
            fig = plt.figure(current_process().pid)
            ax = fig.add_subplot()

            ax.scatter(Bbar_list, means[:, i], color='k', marker='_')
            ax.errorbar(Bbar_list, means[:, i], stds[:, i], ls='', color='k')
            ax.fill_between([min(Bbar_list), max(Bbar_list)],
                            [mean[i] - numpy.sqrt(var[i]),
                             mean[i] - numpy.sqrt(var[i])],
                            [mean[i] + numpy.sqrt(var[i]), mean[i] + numpy.sqrt(var[i])], alpha=0.1, color='k')
            ax.set_title(param)

            fig.savefig(f'Local/graphs/fit_params{self.param_names}_N{self.N}_{param}.pdf')
            plt.close('all')

    def fit_all_gLmin_all_Bbar(self, run_bootstrap=True) -> None:
        filename_means = f"Local/data/means_{self.metadata}.npy"
        filename_chisqs = f"Local/data/chisqs_{self.metadata}.npy"

        try:
            self.means = numpy.load(filename_means)
            self.chisqs = numpy.load(filename_chisqs)

            assert self.means.shape == (len(self.Bbar_pieces), len(self.gL_mins), self.n_params)
            # raise(Exception)

        except Exception:
            self.means = numpy.zeros((len(self.Bbar_pieces), len(self.gL_mins), self.n_params))
            self.chisqs = numpy.zeros((len(self.Bbar_pieces), len(self.gL_mins)))

            for i, Bbar in enumerate(self.Bbar_pieces):
                for j, gL_min in enumerate(self.gL_mins):
                    self.means[i, j], self.chisqs[i, j] = self.fit(Bbar, gL_min, print_info=True)

            numpy.save(filename_means, self.means)
            numpy.save(filename_chisqs, self.chisqs)

        if run_bootstrap:
            filename = f"Local/data/full_results_{self.metadata}boot{self.no_samples}.npy"

            try:
                self.full_results = numpy.load(filename)
                assert self.full_results.shape == (len(self.Bbar_pieces), len(self.gL_mins),
                                                    self.no_samples, self.n_params)
                # raise(Exception)

            except Exception:
                self.full_results = numpy.zeros((len(self.Bbar_pieces), len(self.gL_mins),
                                                 self.no_samples, self.n_params))

                for i, Bbar in enumerate(self.Bbar_pieces):
                    for j, gL_min in enumerate(self.gL_mins):
                        self.full_results[i, j] = self.run_bootstrap(Bbar, gL_min)

                numpy.save(filename, self.full_results)

        self.p_values = chisq_pvalue(self.dof_matrix, self.chisqs)

    def get_frequentist_info(self):
        acceptable = self.p_values > self.p_min

        acceptable[:, self.dofs < self.min_dof] = False

        # Find the lowest gL_min fit with an acceptable p-value
        gL_ind = numpy.argmax(acceptable, axis=1)
        gL_ind = numpy.where(numpy.sum(acceptable, axis=1) == 0, numpy.inf, gL_ind)

        try:
            gL_ind = int(numpy.rint(numpy.min(gL_ind)))

        # If there are no appropriate fits
        except OverflowError:
            self.Bbar_best = None
            self.gL_min_best = None
            return None

        # In case of a tie breaker choose the highest p-value
        Bbar_ind = numpy.argmax(self.p_values[:, gL_ind])

        # Find the Bbar with the lowest acceptable gL_min
        Bbar_ind = numpy.argmin(gL_ind)

        self.Bbar_best = self.Bbar_pieces[Bbar_ind]
        self.gL_min_best = self.gL_mins[gL_ind]

        # Get the best fit
        self.best_fit = self.means[Bbar_ind, gL_ind]

        # Find error bars given a minimum number of dofs
        means = self.means.reshape((len(self.Bbar_pieces) * len(self.gL_mins), self.n_params))
        acceptable = acceptable.flatten()

        means_ = means[acceptable]
        self.maxs = numpy.max(means_, axis=0)
        self.mins = numpy.min(means_, axis=0)

        acceptable_ind = numpy.argwhere(acceptable).flatten()
        max_ind = acceptable_ind[numpy.argmax(means_, axis=0)]
        Bbar_ind = max_ind // len(self.gL_mins)
        gL_min_ind = max_ind % len(self.gL_mins)

        # Now get the statistical error of the best fit
        self.stat_error = numpy.std(self.full_results[Bbar_ind, gL_ind], axis=0)

    def freq_plot_gL_mins(self, ax=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        for i, Bbar in enumerate(self.Bbar_list):
            plt.scatter(self.gL_mins, self.p_values[i], label=f'Bbar = {Bbar}', *args, **kwargs)

        plt.legend()

        return ax

    def BMM_specific(self, Bbar, perform_checks=False):
        i = numpy.argwhere(numpy.array(self.Bbar_list) == Bbar)[0][0]

        var_s = numpy.std(self.full_results[i], axis=1) ** 2

        # Find the probabilities of the models given this data
        exponents = -0.5 * self.chisqs[i] + self.dof_matrix[i]

        if max(exponents) < 0:
            exponents -= max(exponents)

        ps = numpy.exp(exponents)

        if perform_checks:
            # Set any ps to zero if they have larger errors than values
            for j in range(self.n_params):
                stds_check = numpy.sqrt(var_s[:, j])
                means_check = self.means[i, :, j]
                ratio = numpy.abs(stds_check / means_check)
                ps[ratio > 1] = 0

        if sum(ps) == 0:
            self.mean_pieces[Bbar] = numpy.zeros(len(self.param_names))
            self.var_pieces[Bbar] = numpy.zeros(len(self.param_names))

            return ps

        mean = numpy.average(self.means[i], weights=ps, axis=0)

        var = numpy.average(var_s, weights=ps, axis=0) +\
            numpy.average(self.means[i] ** 2, weights=ps, axis=0) - mean ** 2

        self.mean_pieces[Bbar] = mean
        self.var_pieces[Bbar] = var

        return ps

    def BMM_specific_gL_min(self, gL_min, perform_checks=False):
        i = numpy.argwhere(numpy.array(self.gL_mins) == gL_min)[0][0]

        var_s = numpy.std(self.full_results[:, i], axis=1) ** 2

        # Find the probabilities of the models given this data
        exponents = -0.5 * self.chisqs[:, i] + self.dof_matrix[:, i]

        if max(exponents) < 0:
            exponents -= max(exponents)

        ps = numpy.exp(exponents)

        if perform_checks:
            # Set any ps to zero if they have larger errors than values
            for j in range(self.n_params):
                stds_check = numpy.sqrt(var_s[:, j])
                means_check = self.means[:, i, j]
                ratio = numpy.abs(stds_check / means_check)
                ps[ratio > 1 / 2] = 0

            # Remove values which have a mean of nu greater than 2
            ps[self.means[:, i, -1] > self.nu_max] = 0

        if sum(ps) == 0:
            self.mean_pieces_gL[gL_min] = numpy.zeros(len(self.param_names))
            self.var_pieces_gL[gL_min] = numpy.zeros(len(self.param_names))

            return ps

        mean = numpy.average(self.means[:, i], weights=ps, axis=0)

        var = numpy.average(var_s, weights=ps, axis=0) +\
            numpy.average(self.means[:, i] ** 2, weights=ps, axis=0) - mean ** 2

        self.mean_pieces_gL[gL_min] = mean
        self.var_pieces_gL[gL_min] = var

        return ps

    def BMM_plot_gLmin(self, Bbar, params=None, ax_in=None, plot_dict={'nu': [0, 1],
                    'beta': [-2, 2], 'beta1': [-2, 2], 'beta2': [-2, 2]},
                    show_pvalue=True, colors=None, show=False, **kwargs):
        i = numpy.argwhere(numpy.array(self.Bbar_list) == Bbar)[0][0]

        p_values = chisq_pvalue(self.dof_matrix[i], self.chisqs[i])

        var_s = numpy.std(self.full_results[i], axis=1) ** 2

        ps = self.BMM_specific(Bbar, **kwargs)

        # Normalize the ps for the sake of plotting them later
        ps = ps / numpy.sum(ps)

        mean = self.mean_pieces[Bbar]

        var = self.var_pieces[Bbar]

        include = ps > 0

        if params is None:
            params = self.param_names

        color_idx = 0
        for k, param in enumerate(self.param_names):
            if param not in params:
                continue

            if colors is not None:
                color = colors[color_idx]
                color_idx += 1

            else:
                color = 'k'

            if ax_in is None:
                fig, ax = plt.subplots()

            else:
                ax = ax_in

            sc = ax.scatter(self.gL_mins[include], self.means[i, include, k], marker='_', color=color)
            color = sc.get_facecolors()[0]

            ax.errorbar(self.gL_mins[include], self.means[i, include, k], numpy.sqrt(var_s[include, k]), ls='',
                        color=color)

            ax.fill_between([min(self.gL_mins), max(self.gL_mins)],
                        [mean[k] - numpy.sqrt(var[k]), mean[k] - numpy.sqrt(var[k])],
                        [mean[k] + numpy.sqrt(var[k]), mean[k] + numpy.sqrt(var[k])],
                        alpha=0.1, color=color, label=f'{param}')

            ax.set_xlabel(r'$gL_{min}$')
            ax.set_ylabel(rf'${param_names_latex[param]}$')

            if param in plot_dict:
                ax.set_ylim(plot_dict[param])

            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('p-value, weights')

            if show_pvalue:
                ax2.scatter(self.gL_mins[include], p_values[include], label='p-values', color='r')

            ax2.scatter(self.gL_mins[include], ps[include], marker='x', color='g', label='weights')

            plt.legend()

            if ax_in is None:
                fig.savefig(f'Local/graphs/BMA_study/gLmin_plots/{self.metadata}boot{self.no_samples}_param{param}_Bbar{Bbar}.pdf')

                if show:
                    plt.show()

        if ax_in is None:
            plt.close('all')

    def BMM_overall(self, perform_checks=False):
        var_s = numpy.std(self.full_results, axis=2) ** 2
        self.P_s = numpy.zeros_like(self.chisqs)

        for i, Bbar in enumerate(self.Bbar_pieces):
            self.P_s[i] = self.BMM_specific(Bbar, perform_checks=perform_checks)

        self.mean = []
        self.var = []

        try:
            for i in range(len(self.param_names)):
                self.mean.append(numpy.average(self.means[..., i], weights=self.P_s, axis=(0, 1)))

                self.var.append(numpy.average(var_s[..., i], weights=self.P_s, axis=(0, 1)) +
                                numpy.average(self.means[..., i] ** 2, weights=self.P_s, axis=(0, 1))
                            - self.mean[i] ** 2)

        except ZeroDivisionError:
            self.mean = [0, ] * len(self.param_names)
            self.var = [0, ] * len(self.param_names)

        # Out of interest let's calculate the p-value of a constant fit across the Bbar's. This
        # is not a statistically sound thing to do but will give us an idea on whether the results
        # are coming out consistantly across Bbar.

        mean_pieces = numpy.array([self.mean_pieces[i][..., -1] for i in self.mean_pieces])
        std_pieces = numpy.sqrt(numpy.array([self.var_pieces[i][..., -1] for i in self.var_pieces]))

        std_pieces = std_pieces[std_pieces > 0]
        mean_pieces = mean_pieces[mean_pieces > 0]

        try:
            mean = numpy.average(mean_pieces, weights=1 / std_pieces ** 2)
            chisq = 0.5 * numpy.sum(((mean_pieces - mean) / std_pieces) ** 2)

            p = chisq_pvalue(len(mean_pieces) - 1, chisq)

        except ZeroDivisionError:
            p = 0

        return p

    def BMM_plot_overall(self, plot_dict={'nu': [0, 1]}, show=False, params=None):
        mean_pieces = numpy.zeros((len(self.Bbar_pieces), len(self.param_names)))
        var_pieces = numpy.zeros((len(self.Bbar_pieces), len(self.param_names)))

        for i, Bbar in enumerate(self.Bbar_pieces):
            if Bbar not in self.mean_pieces:
                self.BMM_specific(Bbar)

            mean_pieces[i] = self.mean_pieces[Bbar]
            var_pieces[i] = self.var_pieces[Bbar]

        if self.mean is None:
            self.BMM_overall()

        if params is None:
            params = self.param_names

        # The overall weight given to each Bbar
        P_s = self.P_s / numpy.sum(self.P_s)
        ps = numpy.sum(P_s, axis=1)

        for i, param in enumerate(self.param_names):
            if param not in params:
                continue

            fig = plt.figure()
            ax = fig.add_subplot()

            ax.scatter(self.Bbar_pieces, mean_pieces[:, i], color='k', marker='_')
            ax.errorbar(self.Bbar_pieces, mean_pieces[:, i], numpy.sqrt(var_pieces)[:, i], ls='',
                        color='k')
            ax.fill_between([min(self.Bbar_pieces), max(self.Bbar_pieces)],
                [self.mean[i] - numpy.sqrt(self.var[i]), self.mean[i] - numpy.sqrt(self.var[i])],
                [self.mean[i] + numpy.sqrt(self.var[i]), self.mean[i] + numpy.sqrt(self.var[i])],
                alpha=0.1, color='k')

            ax.set_xlabel(r'$\bar{B}$')
            ax.set_ylabel(rf'${param_names_latex[param]}$')

            if param in plot_dict:
                ax.set_ylim(plot_dict[param])

            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('weights')
            ax2.scatter(self.Bbar_pieces, ps, marker='x', color='g', label='weights')

            fig.savefig(f'Local/graphs/BMA_study/omega_plots/{self.metadata}boot{self.no_samples}_param{param}.pdf')

            if show:
                plt.show()

            else:
                plt.close('all')

    def BMM_plot_overall_gL(self, combine=False, params=None, plot_dict={'nu': [0, 1],
                            'beta': [-2, 2], 'beta1': [-2, 2], 'beta2': [-2, 2]}, colors=None):
        mean_pieces = numpy.zeros((len(self.gL_mins), len(self.param_names)))
        var_pieces = numpy.zeros((len(self.gL_mins), len(self.param_names)))

        for i, gL_min in enumerate(self.gL_mins):
            if gL_min not in self.mean_pieces_gL:
                self.BMM_specific_gL_min(gL_min)

            mean_pieces[i] = self.mean_pieces_gL[gL_min]
            var_pieces[i] = self.var_pieces_gL[gL_min]

        if self.mean is None:
            self.BMM_overall()

        color_idx = 0

        for i, param in enumerate(self.param_names):
            if params is not None:
                if param not in params:
                    continue

            if colors is not None:
                color = colors[color_idx]
                color_idx += 1

            else:
                color = 'k'

            plt.scatter(self.gL_mins, mean_pieces[:, i], color=color, marker='_')
            plt.errorbar(self.gL_mins, mean_pieces[:, i], numpy.sqrt(var_pieces)[:, i], ls='',
                        color=color, label=rf'${param_names_latex[param]}$')

            if param in plot_dict:
                plt.ylim(plot_dict[param])

            plt.legend()
            plt.title(f'N = {self.N}, model={self.model_name}')
            plt.xlabel(r'$gL_{min}$')
            plt.savefig(f'Local/graphs/BMA_study/omega_plots/gL_min_overall{self.metadata}boot{self.no_samples}_param{param}.pdf')

            if not combine:
                plt.close('all')

        if combine:
            plt.show()

    def BMM_run_all_plots(self, plot_dict={'nu': [0, 1]}):
        for Bbar in self.Bbar_pieces:
            self.BMM_plot_gLmin(Bbar, plot_dict=plot_dict)

        self.BMM_plot_overall(plot_dict=plot_dict)

    def BMM_plot_fit(self):
        for i, Bbar in enumerate(self.Bbar_pieces):
            for g in set(self.g_s):
                for Bbar_n in range(2):
                    keep = self.find_keep(Bbar, self.gL_mins[0])
                    keep = numpy.logical_and(keep, self.g_s == g)

                    plt.scatter(1 / (g * self.L_s[keep]), self.m_s[keep] / g, color=color_gs(g))

                    for j, gL_min in enumerate(self.gL_mins):
                        central_values = self.means[i, j]

                        Bbar = self.Bbar_list[i + Bbar_n]
                        keep = self.find_keep(Bbar, gL_min)
                        keep = numpy.logical_and(keep, self.g_s == g)
                        g_s = self.g_s[keep]
                        L_s = self.L_s[keep]

                        if sum(keep) > 1:
                            gL_space = numpy.linspace(min(g_s * L_s), max(g_s * L_s), 1000)

                            plt.plot(1 / gL_space, self.model(self.N, g, gL_space / g, Bbar,
                                     *central_values) / g, color=color_gs(g))

        plt.xlabel("1 / x")
        plt.ylabel("m[B = Bbar] / g")
        plt.title(f"N = {self.N}, model = {self.model_name}")
        plt.legend()
        plt.close()


class analysis_Bayes(analysis):
    def run_pymultinest(self, Bbar, gL_min, INS=False, keep_all=True, tag="", seed=-1,
                        sampling_efficiency=0.3, return_analysis_small=True, delete_files=True):
        """
            See documentation for pymultinest.run()
            INPUTS :
            --------
            gL_min: float, Minimum value of g * L that is used in the fit
            INS: Bool, if True multinest will use importance nested sampling method
            clean_files: Bool, if True then the large data files of MULTINEST samples will be kept
            sampling_efficiency: See MULTINEST documentation. float between 0 and 1.  Here, 0.3 is
                used as this is the reccomended value for Bayesian Evidence
            return_analysis_small: Bool, if True return only key values of the run, explicitly...
                [E, delta_E, sigma_1_range, sigma_2_range, median], where E is the log-evidence,
                delta_E is the error in the log-evidence as estimated by MULTINEST (not always
                accurate), sigma_1_range and sigma_2_range are the 1 and 2 sigma band estimates of
                parameter values from the posterior distribution respectively, median is the median
                parameter estimate, also from the posterior distribution.
            tag: string, change this to label the files associated with a run uniquely
            seed: int, starting seed for the random MULTINEST algorithm

            OUTPUTS:
            --------
            This depends on whether return_analysis_small is True or not. If it is
            then the following are returned:
            > analysis_small: list contatining
                - E: float, Bayesian Evidence
                - delta_E: float, Estimated statistical error in E
                - sigma_1_range: list of (2, ) lists of floats. 1 sigma confidence
                    bands of the parameter estimates
                - sigma_2_range: list of (2, ) lists of floats. 2 sigma confidence
                    bands of the parameter estimates
                - median: list of floats. Median estimated values of the fit
                    parameters
            > best_fit: list of floats containing the parameter estimates of the
                Maximum A Posteriori (MAP) point
            If return_analysis_small is False then the following are returned:
            > analysis: list contatining
                - E: float, Bayesian Evidence
                - delta_E: float, Estimated statistical error in E
                - sigma_1_range: list of (2, ) lists of floats. 1 sigma confidence
                    bands of the parameter estimates
                - sigma_2_range: list of (2, ) lists of floats. 2 sigma confidence
                    bands of the parameter estimates
                - median: list of floats. Median estimated values of the fit
                    parameters
                - posterior_data: Array contatining Bayesian Evidence values and
                    parameter values at all points visited by the MULTINEST=
                    algorithm
            > best_fit: list of floats containing the parameter estimates of the
                Maximum A Posteriori (MAP) point
        """
        basename = (f"Local/data/Bayes_{self.metadata}p{self.bayes_points}")

        # Remove the remaining saved files to conserve disk space
        file_endings = ["ev.dat", "live.points", ".paramnames", "params.json", "phys_live.points",
                        "post_equal_weights.dat", "post_separate.dat", ".ranges", "resume.dat",
                        "stats.dat", "summary.txt", ".txt"]

        for ending in file_endings:
            os.popen(f'rm {basename}{ending}')

        keep = self.find_keep(Bbar, gL_min)

        n_live_points = self.bayes_points

        g_s = self.g_s[keep]
        L_s = self.L_s[keep]
        m_s = self.m_s[keep]
        Bbar_s = self.B_s[keep]

        self.find_cov_matrix(Bbar, gL_min)

        # Obtain a function for getting the normalized residuals
        res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

        # Calculate log-likelihood and prior functions for use by MULTINEST
        likelihood_function = likelihood_maker(self.n_params, self.cov_inv, self.model,
                                               res_function)

        # The filenames are limited to 100 charachters in the MUTLTINEST code
        if len(basename) >= 100:
            print('Name too long for MULTINEST')
            exit()

        # Run the MULTINEST sampler
        Bbar_s_set = list(set(Bbar_s))
        print("################### Initiating MULTINEST ###########################")
        print(f"Config: N = {self.N}, Bbar = {Bbar}, gL_min = {gL_min}," +
              f"model = {self.model.__name__}")
        pymultinest.run(likelihood_function, self.prior, self.n_params,
                        outputfiles_basename=basename, resume=False, n_live_points=n_live_points,
                        sampling_efficiency=sampling_efficiency, evidence_tolerance=0.1,
                        importance_nested_sampling=INS, seed=seed)
        print("MULTINEST run complete")

        # Save the prior ranges
        f = open(basename + '.ranges', 'w')
        for i in range(len(self.param_names)):
            f.write(f"{self.param_names[i]} {self.prior_ranges[i][0]} {self.prior_ranges[i][1]}\n")

        f.close()

        # save parameter names
        f = open(basename + '.paramnames', 'w')
        for i in range(len(self.param_names)):
            f.write(f"{self.param_names[i]}\t{self.param_names_latex[i]}\n")

        f.close()

        # Also save as a json
        json.dump(self.param_names, open(f'{basename}params.json', 'w'))

        # Get information about the MULTINEST run
        analysis = pymultinest.Analyzer(outputfiles_basename=basename, n_params=self.n_params)
        stats = analysis.get_stats()

        # Extract the log-evidence and its error
        E, delta_E = stats['global evidence'], stats['global evidence error']

        # Extract parameter estimates from the posterior distributions
        sigma_1 = [analysis.get_stats()['marginals'][i]['1sigma'] for i in range(self.n_params)]
        sigma_2 = [analysis.get_stats()['marginals'][i]['2sigma'] for i in range(self.n_params)]
        median = [analysis.get_stats()['marginals'][i]['median'] for i in range(self.n_params)]

        # Extract the points sampled by MULTINEST
        posterior_data = analysis.get_equal_weighted_posterior()

        # Collate data for saving/returning
        analysis_data = [E, delta_E, sigma_1, sigma_2, posterior_data, median]

        if keep_all:
            pickle.dump(analysis_data, open(f"{basename}_analysis.pcl", "wb"))

        # Find the parameter estimates at the MAP
        best_fit = analysis.get_best_fit()

        # Make a cut down version for the purpose of quicker transfer
        analysis_small = [E, delta_E, sigma_1, sigma_2, median, best_fit]

        pickle.dump(analysis_small, open(f"{basename}_analysis_small.pcl", "wb"))

        for ending in file_endings:
            os.popen(f'rm {basename}{ending}')

        if return_analysis_small:
            return analysis_small

        else:
            return analysis_data

    def run_Bayes(self, Bbar, gL_min, *args, **kwargs):
        filename = f"Local/data/Bayes_{self.metadata}p{self.bayes_points}_s{self.scale}_B{Bbar:.2f}_gLm{gL_min:.1f}.pcl"
        try:
            results = pickle.load(open(filename, 'rb'))

        except Exception:
            results = self.run_pymultinest(Bbar, gL_min, *args, **kwargs)

            pickle.dump(results, open(filename, 'wb'))

        return results

    def run_Bayes_all(self, *args, **kwargs):
        self.evidence = numpy.zeros((len(self.Bbar_pieces), len(self.gL_mins)))
        self.bayes_best = numpy.zeros((len(self.Bbar_pieces), len(self.gL_mins)))
        self.bayes_best_xs = numpy.zeros((len(self.Bbar_pieces), len(self.gL_mins),
                                            self.n_params))

        for i, Bbar in enumerate(self.Bbar_pieces):
            for j, gL_min in enumerate(self.gL_mins):
                results = self.run_Bayes(Bbar, gL_min, *args, **kwargs)
                self.evidence[i, j] = results[0]
                self.bayes_best[i, j] = results[-1]['log_likelihood']
                self.bayes_best_xs[i, j] = results[-1]['parameters']

    def plot_posteriors(self, Bbar, gL_min, *args, show_fig=True, **kwargs):
        # Check if the files are present
        basename = (f"Local/data/Bayes_{self.metadata}p{self.bayes_points}")

        missing_files = []

        file_endings = ["ev.dat", "live.points", ".paramnames", "params.json", "phys_live.points",
                        "post_equal_weights.dat", "post_separate.dat", ".ranges", "resume.dat",
                        "stats.dat", "summary.txt", ".txt"]

        for ending in file_endings:
            missing_files.append(not os.path.exists(f'{basename}{ending}'))

        if sum(missing_files) > 0:
            self.run_Bayes(Bbar, gL_min, *args, delete_files=False, **kwargs)

        samples = loadMCSamples(f"Local/data/Bayes_{self.metadata}p{self.bayes_points}")

        g = plots_edit.get_subplot_plotter()
        g.triangle_plot(samples, filled=True)

        if show_fig:
            plt.show()

        else:
            plt.close()

    def plot_posteriors_all(self, *args, **kwargs):
        for i, Bbar in enumerate(self.Bbar_pieces):
            for j, gL_min in enumerate(self.gL_mins):
                if i == 8:
                    self.plot_posteriors(Bbar, gL_min, show_fig=True)
                else:
                    self.plot_posteriors(Bbar, gL_min, show_fig=False)


class CompareModels:
    def __init__(self, N, model_name, **kwargs):
        self.model1 = analysis(N, model_name=model_name, model_type=1, **kwargs)
        self.model2 = analysis(N, model_name=model_name, model_type=2, **kwargs)
        self.model3 = analysis(N, model_name=model_name, model_type=3, **kwargs)
        self.model4 = analysis(N, model_name=model_name, model_type=4, **kwargs)

        self.N = N
        self.model_name = model_name

    def GRID_plot(self, chisq_only=False, compare=(1, 2), test_fit_sig=False, show=False):
        models = {"1": self.model1,
                  "2": self.model2,
                  "3": self.model3,
                  "4": self.model4}

        model1 = models[str(compare[0])]
        model2 = models[str(compare[1])]

        # Now add in the frequentist data
        run_bootstrap = test_fit_sig

        model1.fit_all_gLmin_all_Bbar(run_bootstrap=run_bootstrap)
        model2.fit_all_gLmin_all_Bbar(run_bootstrap=run_bootstrap)

        ps1 = model1.p_values
        ps2 = model2.p_values

        chisqs1 = model1.chisqs
        chisqs2 = model2.chisqs
        del_chisqs = chisqs2 - chisqs1

        fig, ax = plt.subplots()

        if not chisq_only:
            model1.run_Bayes_all()
            model2.run_Bayes_all()
            del_bests = 2 * (model1.bayes_best - model2.bayes_best)

            # Test the accuracy of MULTINEST convergence
            test1 = numpy.abs(model1.chisqs + 2 * model1.bayes_best) < 1
            test2 = numpy.abs(model2.chisqs + 2 * model2.bayes_best) < 1

            acc_matrix = ((test1).astype(int) + (test2).astype(int)) / 2

            accuracy = 100 * numpy.mean(acc_matrix)  # As %

        cmap = colors.ListedColormap(['lightcoral', 'navajowhite', 'lightgrey',
                                      'lightgreen', 'cornflowerblue'])

        bounds = [-10 ** 9, -2, -1, 1, 2, 10 ** 9]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        gL_mins = model1.gL_mins
        Bbars = model1.Bbar_pieces

        if chisq_only:
            imshow_data = del_chisqs / 2

        else:
            imshow_data = del_bests / 2

        # Determine if the fits are solid - e.g. do they have <100% error bars on all params
        if test_fit_sig:
            stds1 = numpy.std(model1.full_results, axis=2)
            stds2 = numpy.std(model2.full_results, axis=2)

            ratio1 = numpy.abs(model1.means) / stds1
            ratio2 = numpy.abs(model2.means) / stds2

            # Find the worst value of the ratio - lowest abs of fit_param_value) / std(fit_param_value)
            ratio1 = numpy.min(ratio1, axis=2)
            ratio2 = numpy.min(ratio2, axis=2)

            insignificant_fit1 = ratio1 < 1
            insignificant_fit2 = ratio2 < 1

            # Only color the squares where the fit is si
            insignificant_fit = numpy.logical_or(insignificant_fit1, insignificant_fit2)

        # tell imshow about color map so that only set colors are used
        ax.imshow(imshow_data, interpolation='nearest', origin='lower', cmap=cmap, norm=norm,
                  aspect="auto", extent=[min(gL_mins), max(gL_mins), max(Bbars), min(Bbars)])

        ax.set_xlabel(r'$gL_{min}$')
        ax.set_ylabel(r'$\bar{B}$')

        ax.set_xticks(numpy.linspace(min(gL_mins), max(gL_mins), len(gL_mins) + 1))
        ax.set_yticks(numpy.linspace(min(Bbars), max(Bbars), len(Bbars) + 1))

        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)

        x_labels = ["", ] + [str(i) + ' ' * 20 for i in list(gL_mins)]
        y_labels = ["", ] + [str(i) + '\n' * 6 for i in list(Bbars)][::-1]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.grid(color='w', linestyle='-', linewidth=2)
        fig.set_size_inches(20, 10)

        del_Bbar = (max(Bbars) - min(Bbars)) / (len(Bbars))
        del_gL_min = (max(gL_mins) - min(gL_mins)) / (len(gL_mins))

        crossovers = numpy.full_like(Bbars, numpy.inf)
        crossovers2 = numpy.full_like(Bbars, numpy.inf)

        for i, Bbar in enumerate(Bbars):
            current_best = 0
            transitioned = False

            for j, gL_min in enumerate(gL_mins):
                # Show transitions from one model supported to the other
                new_best = 1 if chisqs2[i, j] > chisqs1[i, j] else -1

                if ((current_best == -1) and (new_best == 1)) and not transitioned:
                    transitioned = True
                    crossovers[i] = 0.5 * (gL_mins[j] + gL_mins[j - 1])
                    crossovers2[i] = 0.5 * (model1.find_median_gLmin(gL_mins[j]) +
                                            model1.find_median_gLmin(gL_mins[j - 1]))

                    width = del_gL_min / 20
                    height = del_Bbar

                    xpos = (j - 0.025) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)

                    ypos = (len(Bbars) - (i + 1)) * (max(Bbars) - min(Bbars)) / len(Bbars) + min(Bbars)

                    rect1 = patches.Rectangle((xpos, ypos), width, height, linewidth=1,
                                            edgecolor='purple', facecolor='purple', zorder=numpy.inf)

                    ax.add_patch(rect1)

                current_best = new_best
                if test_fit_sig:
                    if insignificant_fit[i, j]:
                        ypos = (len(Bbars) - (i + 1) + 0.15) * (max(Bbars) - min(Bbars)) / len(Bbars) + min(Bbars)

                        if insignificant_fit1[i, j]:
                            xpos = (j + 0.15) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)

                            ax.scatter(xpos, ypos, marker='s', facecolors=(0.2, 0.2, 0.2),
                                    edgecolors=None, s=35)

                        if insignificant_fit2[i, j]:
                            xpos = (j + 0.3) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)

                            ax.scatter(xpos, ypos, marker='^', facecolors=(0.2, 0.2, 0.2),
                                    edgecolors=None, s=35)

                xpos = (j + 0.5) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)
                ypos = (len(Bbars) - (i + 1) + 0.5) * (max(Bbars) - min(Bbars)) / len(Bbars) + min(Bbars)

                edgecolor1 = 'k' if (ps1[i, j] > 0.05 and ps1[i, j] < 0.95) else None
                edgecolor2 = 'k' if (ps2[i, j] > 0.05 and ps2[i, j] < 0.95) else None

                h1, h2 = ps1[i, j] * del_Bbar * 0.8, ps2[i, j] * del_Bbar * 0.8
                width = del_gL_min / 15

                rect1 = patches.Rectangle((xpos - width / 2, ypos - h1 / 2), width, h1, linewidth=1,
                                           edgecolor=edgecolor1, facecolor='orange')
                rect2 = patches.Rectangle((xpos + width / 2, ypos - h2 / 2), width, h2, linewidth=1,
                                           edgecolor=edgecolor2, facecolor='green')

                ax.add_patch(rect1)
                ax.add_patch(rect2)

                # Add in significance of fit symbols
                ypos = (len(Bbars) - (i + 1) + 0.8) * (max(Bbars) - min(Bbars)) / len(Bbars) + min(Bbars)

                # ax.scatter(xpos, ypos, marker='o', facecolors='orange', edgecolors='r', s=50)
                # ax.scatter(xpos, ypos, marker='^', facecolors='green', edgecolors='k', s=50)

                # if ps3[i, j] > 0.05:
                #     xpos = (j + 0.8) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)

                #     ax.scatter(xpos, ypos, marker='s', facecolors='purple', edgecolors='k', s=50)

                if not chisq_only:
                    if abs(acc_matrix[i, j] - 1) < 10 ** -12:
                        xpos = (j + 0.8) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)
                        ypos = (len(Bbars) - (i + 1) + 0.2) * (max(Bbars) - min(Bbars)) / len(Bbars) + min(Bbars)

                        ax.scatter(xpos, ypos, marker='*', facecolors='yellow', edgecolors='k', s=50)

                # Add in the delta_chisq_max as a check
                xpos = (j + 0.2) * (max(gL_mins) - min(gL_mins)) / len(gL_mins) + min(gL_mins)

                ypos = (len(Bbars) - (i + 1) + 0.5) * (max(Bbars) - min(Bbars)) / len(Bbars) + min(Bbars)

                # ax.text(xpos, ypos, f'{del_chisqs[i, j]:.0f}')

        pivot = numpy.median(crossovers)
        pivot2 = numpy.median(crossovers2)
        title = f'N = {self.N}, model = {self.model_name}, pivot = {pivot}, pivot2 = {pivot2}'

        if not chisq_only:
            title += f'points={model1.bayes_points}, scale={model1.scale}, accuracy={accuracy:.1f}%'

        plt.title(title)

        if chisq_only:
            plt.savefig(f'Local/graphs/BMA_study/Chisqs/combined_plot_compare{compare}_{model1.metadata}.pdf')

        else:
            plt.savefig(f'Local/graphs/BMA_study/Bayes/combined_plot_compare{compare}_{model1.metadata}_p{model1.bayes_points}_s{model1.scale}.pdf')

        if show:
            plt.show()

        else:
            plt.close('all')
