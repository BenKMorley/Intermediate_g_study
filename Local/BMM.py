# Import from the Core directory
import enum
import sys
import os

from numpy.lib import RankWarning
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')


from numpy.core.einsumfunc import _parse_possible_contraction
from Core.model_definitions import *
from scipy.optimize import minimize, least_squares
from scipy.linalg import logm
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


def prior_penalty(N, params):
    if N == 2:
        prior_lowers = [-0.4, 0, -20, 0, -15]
        prior_uppers = [0.4, 1, 20, 15, 15]

    if N == 4:
        prior_lowers = [-0.4, -0.4, 0, -20, 0, -15]
        prior_uppers = [0.4, 0.4, 1, 20, 15, 15]

    for i, param in enumerate(params):
        if prior_lowers[i] > params[i]:
            return numpy.inf

        if prior_uppers[i] < params[i]:
            return numpy.inf

    return 0


def make_res_function(N, m_s, g_s, L_s, Bbar_s):
    def res_function(x, cov_inv, model_function, return_details=False):
        # Caculate the residuals between the model and the data
        predictions = model_function(N, g_s, L_s, Bbar_s, *x)

        # Add an infinite penalty for being outside of the priors
        predictions += prior_penalty(N, x)

        residuals = m_s - predictions

        if abs(sum(residuals)) == numpy.inf:
            normalized_residuals = residuals

        else:
            normalized_residuals = numpy.dot(cov_inv, residuals)

        if return_details:
            return normalized_residuals, N, g_s, L_s, Bbar_s, x

        else:
            return normalized_residuals

    return res_function


## Parameters
def central(N, Bbar_s_in=None, plot=True, cut=False):
    input_h5_file = "h5data/Bindercrossings.h5"
    N_s_in = [N]
    g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]

    if N == 2:
        if Bbar_s_in is None:
            Bbar_s_in = [0.52, 0.53]
        param_names = ["alpha", "f0", "f1", "beta", "nu"]
        x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
        model = model1_1a

        # The uniform priors
        prior_lowers = [-0.4, 0, -20, 0, -15]
        prior_uppers = [0.4, 1, 20, 15, 15]

    if N == 4:
        if Bbar_s_in is None:
            Bbar_s_in = [0.42, 0.43]
        param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]
        x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
        model = model1_2a

        prior_lowers = [-0.4, -0.4, 0, -20, 0, -15]
        prior_uppers = [0.4, 0.4, 1, 20, 15, 15]

    L_s_in = [8, 16, 32, 48, 64, 96, 128]
    GL_max = numpy.inf
    print_info = True

    no_samples = 500

    GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
                           16, 19.2, 24, 25.6, 28.8, 32])
    GL_mins = numpy.array([28.8, 32])

    means = []
    stds = []
    probs = []

    # Try loading in all the data so we know how many points are being cut out
    samples_full, g_s_full, L_s_full, Bbar_s_full, m_s_full = load_h5_data(
        input_h5_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, 0, GL_max)

    for j, GL_min in enumerate(GL_mins):
        samples, g_s, L_s, Bbar_s, m_s = load_h5_data(input_h5_file, N_s_in, g_s_in, L_s_in,
                                                      Bbar_s_in, GL_min, GL_max)

        cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
        cov_1_2 = numpy.linalg.cholesky(cov_matrix)
        cov_inv = numpy.linalg.inv(cov_1_2)

        res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

        res_function([-1, 0, 0.4459, -0.02707, 1, 2 / 3], cov_inv, model)

        res = least_squares(res_function, x0, args=(cov_inv, model),
                            method="lm")

        chisq = chisq_calc(res.x, cov_inv, model, res_function)
        n_params = len(res.x)
        dof = g_s.shape[0] - n_params
        p = chisq_pvalue(dof, chisq)
        param_central = res.x
        N_cut = len(g_s_full) - len(g_s)
        AIC = chisq + 2 * n_params + 2 * N_cut

        # Apply a two-sided cut
        if cut:
            if p > 0.05 and p < 0.95:
                prob_model = numpy.exp(-0.5 * AIC)
            else:
                prob_model = 0
        else:
            prob_model = numpy.exp(-0.5 * AIC)

        means.append(param_central)
        probs.append(prob_model)

        if print_info:
            print("##############################################################")
            print(f"Config: N = {N}, Bbar_s = [{Bbar_s_in[0]}, {Bbar_s_in[1]}],"
                  f" gL_min = {GL_min}, gL_max = {GL_max}")
            print(f"chisq = {chisq}")
            print(f"chisq/dof = {chisq / dof}")
            print(f"pvalue = {p}")
            print(f"dof = {dof}")
            print(f"N_cut = {N_cut}")
            print(f"AIC = {AIC}")
            print(f"prob_model = {prob_model}")

        param_estimates = numpy.zeros((no_samples, n_params))

        for i in tqdm(range(no_samples)):
            m_s = samples[:, i]

            res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

            res = least_squares(res_function, x0, args=(cov_inv, model),
                                method="lm")

            if abs(res.x[2]) > 1:
                print("Hellooooo")
                res_function(res.x, cov_inv, model)

            param_estimates[i] = numpy.array(res.x)

        stds.append(numpy.std(param_estimates, axis=0))

    means = numpy.array(means)
    stds = numpy.array(stds)
    probs = numpy.array(probs)

    # Normalize the probabilities
    probs = probs / numpy.sum(probs)

    means_overall = []
    stds_overall = []

    ## Plot the Parameter estimates
    for i, param in enumerate(param_names):
        # Calculate the overall parameter estimates
        mean = numpy.average(means[:, i], weights=probs)
        var = numpy.average(stds[:, i] ** 2, weights=probs) +\
            numpy.average(means[:, i] ** 2, weights=probs) - mean ** 2
        std = numpy.sqrt(var)

        means_overall.append(mean)
        stds_overall.append(std)

        alphas = 0.1 + 0.89 * probs / max(probs)
        colors = numpy.asarray([(0, 0, 1, alpha) for alpha in alphas])

        if plot:
            plt.scatter(GL_mins, means[:, i], c=colors, edgecolors=colors, marker='_')

            for pos, ypt, err, color in zip(GL_mins, means[:, i], stds[:, i], colors):
                (_, caps, _) = plt.errorbar(pos, ypt, err, ls='', color=color, capsize=5, capthick=2, marker='_', lw=2)

            plt.fill_between([min(GL_mins), max(GL_mins)], [mean - std], [mean + std], alpha=0.1, color='k')
            plt.plot([min(GL_mins), max(GL_mins)], [mean, mean], color='k', label='BMM')
            plt.title(f"{param}")
            plt.xlabel(r"$gL_{min}$")
            plt.savefig(f"Local/graphs/BMM_{param}_N{N}_Bbars{Bbar_s_in}.pdf")
            plt.clf()

    ## Plot the relative weights of the contributions
    if plot:
        plt.plot(GL_mins, probs, color='k')
        plt.xlabel(r"$gL_{min}$")
        plt.savefig(f"Local/graphs/BMM_probs_N{N}_Bbars{Bbar_s_in}.pdf")

    plt.clf()

    return means_overall, stds_overall


# central(4, Bbar_s_in=[0.44, 0.46], cut=True)


class analysis(object):
    def __init__(self, N):
        self.N = N
        self.input_h5_file = "h5data/Bindercrossings.h5"
        self.N_s_in = [N]
        self.g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]

        if N == 2:
            self.Bbar_s = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
            # Bbar_s = [0.51, 0.52, 0.53]

            self.param_names = ["alpha", "f0", "f1", "beta", "nu"]
            self.x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
            self.model = model1_1a

            # The uniform priors
            self.prior_lowers = [-0.4, 0, -20, 0, -15]
            self.prior_uppers = [0.4, 1, 20, 15, 15]

        if N == 4:
            self.Bbar_s = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47]
            self.param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]
            self.x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
            self.model = model1_2a

            self.prior_lowers = [-0.4, -0.4, 0, -20, 0, -15]
            self.prior_uppers = [0.4, 0.4, 1, 20, 15, 15]

        self.Bbar_list = []
        for i in range(len(self.Bbar_s)):
            for j in range(i + 1, len(self.Bbar_s)):
                self.Bbar_list.append([self.Bbar_s[i], self.Bbar_s[j]])

        self.n_params = len(self.param_names)
        self.L_s_in = [8, 16, 32, 48, 64, 96, 128]
        self.GL_max = numpy.inf
        self.print_info = True
        self.no_samples = 500
        self.GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
                                    16, 19.2, 24, 25.6, 28.8, 32])

    ## Now try and do this using all Bbar combinations
    def run_all_Bbar(self):
        means = numpy.zeros((len(self.Bbar_list), len(self.GL_mins), self.n_params))
        stds = numpy.zeros((len(self.Bbar_list), len(self.GL_mins), self.n_params))
        probs = numpy.zeros((len(self.Bbar_list), len(self.GL_mins)))

        for k, Bbar_s_in in enumerate(self.Bbar_list):
            # Try loading in all the data so we know how many points are being cut out
            samples_full, g_s_full, L_s_full, Bbar_s_full, m_s_full = load_h5_data(
                self.input_h5_file, self.N_s_in, self.g_s_in, self.L_s_in, Bbar_s_in, 0, self.GL_max)

            for j, GL_min in enumerate(self.GL_mins):
                samples, g_s, L_s, Bbar_s, m_s = load_h5_data(self.input_h5_file, self.N_s_in, self.g_s_in, self.L_s_in,
                                                              Bbar_s_in, GL_min, self.GL_max)

                cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
                cov_1_2 = numpy.linalg.cholesky(cov_matrix)
                cov_inv = numpy.linalg.inv(cov_1_2)

                res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

                res = least_squares(res_function, self.x0, args=(cov_inv, self.model),
                                    method="lm")

                chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
                dof = g_s.shape[0] - self.n_params
                p = chisq_pvalue(dof, chisq)
                param_central = res.x
                N_cut = len(g_s_full) - len(g_s)
                AIC = chisq + 2 * self.n_params + 2 * N_cut
                prob_model = numpy.exp(-0.5 * AIC)

                means[k, j] = param_central
                probs[k, j] = prob_model

                if self.print_info:
                    print("##############################################################")
                    print(f"Config: N = {self.N}, Bbar_s = [{Bbar_s_in[0]}, {Bbar_s_in[1]}],"
                        f" gL_min = {GL_min}, gL_max = {self.GL_max}")
                    print(f"chisq = {chisq}")
                    print(f"chisq/dof = {chisq / dof}")
                    print(f"pvalue = {p}")
                    print(f"dof = {dof}")
                    print(f"N_cut = {N_cut}")
                    print(f"AIC = {AIC}")
                    print(f"prob_model = {prob_model}")

                param_estimates = numpy.zeros((self.no_samples, self.n_params))

                for i in tqdm(range(self.no_samples)):
                    m_s = samples[:, i]

                    res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

                    res = least_squares(res_function, self.x0, args=(cov_inv, self.model),
                                        method="lm")

                    param_estimates[i] = numpy.array(res.x)

                stds[k, j] = numpy.std(param_estimates, axis=0)

        # Save the results
        numpy.save(f"Data/BMM_N{self.N}_all_means.npy", means)
        numpy.save(f"Data/BMM_N{self.N}_all_stds.npy", stds)
        numpy.save(f"Data/BMM_N{self.N}_all_probs.npy", probs)

        return means, stds, probs

    def plot_all_Bbar(self):
        try:
            means = numpy.load(f"Data/BMM_N{self.N}_all_means.npy")
            stds = numpy.load(f"Data/BMM_N{self.N}_all_stds.npy")
            probs = numpy.load(f"Data/BMM_N{self.N}_all_probs.npy")

        except FileNotFoundError:
            means, stds, probs = self.run_all_Bbar()

        means = numpy.array(means)
        stds = numpy.array(stds)
        probs = numpy.array(probs)

        # Normalize the probabilities
        probs = probs / numpy.sum(probs)

        means_Bbars = numpy.zeros((len(self.Bbar_list), self.n_params))
        stds_Bbars = numpy.zeros((len(self.Bbar_list), self.n_params))

        for i in range(len(self.Bbar_list)):
            for j in range(self.n_params):
                probs_Bbar = probs[i] / numpy.sum(probs[i])

                means_Bbars[i, j] = numpy.average(means[i, :, j], weights=probs_Bbar)
                var = numpy.average(stds[i, :, j] ** 2, weights=probs_Bbar) +\
                    numpy.average(means[i, :, j] ** 2, weights=probs_Bbar) - means_Bbars[i, j] ** 2
                stds_Bbars[i, j] = numpy.sqrt(var)

        ## Plot the Parameter estimates
        for i, param in enumerate(self.param_names):
            # Calculate the overall parameter estimates
            mean = numpy.average(means[:, :, i], weights=probs)
            var = numpy.average(stds[:, :, i] ** 2, weights=probs) +\
                numpy.average(means[:, :, i] ** 2, weights=probs) - mean ** 2
            std = numpy.sqrt(var)

            alphas = 0.1 + 0.89 * numpy.sum(probs, axis=1) / max(numpy.sum(probs, axis=1))
            colors = numpy.asarray([(0, 0, 1, alpha) for alpha in alphas])

            plt.scatter(numpy.arange(len(self.Bbar_list)), means_Bbars[:, i], marker='_', color=colors)

            for pos, ypt, err, color in zip(range(len(self.Bbar_list)), means_Bbars[:, i], stds_Bbars[:, i], colors):
                (_, caps, _) = plt.errorbar(pos, ypt, err, ls='', capsize=5, capthick=2, marker='_', lw=2, color=color)

            plt.fill_between([0, len(self.Bbar_list) - 1], [mean - std], [mean + std], alpha=0.1, color='k')
            plt.plot([0, len(self.Bbar_list) - 1], [mean, mean], color='k', label='BMM')
            plt.title(f"{param}")
            plt.xticks(range(len(self.Bbar_list)), [str(Bbar) for Bbar in self.Bbar_list], rotation='vertical')
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.2)
            plt.savefig(f"Local/graphs/BMM_{param}_N{self.N}_All_Bbar.pdf")
            plt.clf()


my_analysis = analysis(4)
my_analysis.plot_all_Bbar()
