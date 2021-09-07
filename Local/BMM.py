# Import from the Core directory
import enum
import sys
import os

from numpy.lib import RankWarning
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')


from Core.model_definitions import *
from Core.bayesian_functions import run_pymultinest
from scipy.optimize import least_squares
from tqdm import tqdm
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


# Override the make_res_function to add a prior penalty
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


class analysis(object):
    def __init__(self, N, use_mutlinest=False):
        self.N = N
        self.input_h5_file = "h5data/Bindercrossings.h5"
        self.N_s_in = [N]
        self.g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
        self.use_multinest = use_mutlinest
        self.directory = "Data/"

        if N == 2:
            self.Bbar_s = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
            self.param_names = ["alpha", "f0", "f1", "beta", "nu"]
            self.x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
            self.model = model1_1a

            # The uniform priors
            self.prior_lowers = [-0.4, 0, -20, -15, 0]
            self.prior_uppers = [0.4, 1, 20, 15, 15]

        if N == 4:
            self.Bbar_s = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47]
            self.param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]
            self.x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
            self.model = model1_2a

            self.prior_lowers = [-0.4, -0.4, 0, -20, -15, 0]
            self.prior_uppers = [0.4, 0.4, 1, 20, 15, 15]

        self.prior_range = []
        for x in zip(self.prior_lowers, self.prior_uppers):
            lower, upper = tuple(x)
            self.prior_range.append([lower, upper])

        self.Bbar_list = []
        for i in range(len(self.Bbar_s)):
            for j in range(i + 1, len(self.Bbar_s)):
                self.Bbar_list.append([self.Bbar_s[i], self.Bbar_s[j]])

        self.n_params = len(self.param_names)
        self.L_s_in = [8, 16, 32, 48, 64, 96, 128]
        self.GL_max = numpy.inf
        self.print_info = True
        self.no_samples = 500

        # Set up the GL_mins
        GL_mins = numpy.outer(numpy.array(self.L_s_in), numpy.array(self.g_s_in)).flatten()
        GL_mins = numpy.sort(GL_mins)
        GL_mins = numpy.around(GL_mins, decimals=8)
        GL_mins = numpy.unique(GL_mins)
        GL_mins = numpy.sort(GL_mins)
        self.GL_mins = GL_mins[GL_mins <= 32]

    def run_Bbar_pair(self, Bbar_s_in, print_info=True, plot=True):
        means = numpy.zeros((len(self.GL_mins), self.n_params))
        stds = numpy.zeros((len(self.GL_mins), self.n_params))
        probs = numpy.zeros(len(self.GL_mins))

        # Try loading in all the data so we know how many points are being cut out
        samples_full, g_s_full, L_s_full, Bbar_s_full, m_s_full = load_h5_data(
            self.input_h5_file, self.N_s_in, self.g_s_in, self.L_s_in, Bbar_s_in, 0, self.GL_max)

        for j, GL_min in enumerate(self.GL_mins):
            samples, g_s, L_s, Bbar_s, m_s = load_h5_data(self.input_h5_file, self.N_s_in, self.g_s_in, self.L_s_in,
                                                          Bbar_s_in, GL_min, self.GL_max)

            N_cut = len(g_s_full) - len(g_s)

            if self.use_multinest:
                analysis, best_fit = run_pymultinest(self.prior_range, self.model, GL_min,
                    self.GL_max, self.n_params, f"{self.directory}MULTINEST/", self.N,
                    g_s, Bbar_s, L_s, samples, m_s, self.param_names,
                    return_analysis_small=True)

                # Use the Bayesian Evidence directly with a correction for N_cut
                probs[j] = numpy.exp(analysis[0] - N_cut)
                means[j] = best_fit['parameters']

                # Assume the distribution to be symmetric
                sigma_1_range = analysis[2]
                for i in range(self.n_params):
                    stds[j, i] = 0.5 * max(sigma_1_range[i]) - min(sigma_1_range[i])

                if self.print_info:
                    print(f"prob_model = {probs[j]}")

            else:
                cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
                cov_1_2 = numpy.linalg.cholesky(cov_matrix)
                cov_inv = numpy.linalg.inv(cov_1_2)

                res = least_squares(res_function, self.x0, args=(cov_inv, self.model),
                                    method="lm")

                chisq = chisq_calc(res.x, cov_inv, self.model, res_function)
                n_params = len(res.x)
                dof = g_s.shape[0] - n_params
                p = chisq_pvalue(dof, chisq)
                means[j] = res.x
                AIC = chisq + 2 * n_params + 2 * N_cut
                probs[j] = numpy.exp(-0.5 * AIC)

                if print_info:
                    print("##############################################################")
                    print(f"Config: N = {self.N}, Bbar_s = [{Bbar_s_in[0]}, {Bbar_s_in[1]}],"
                        f" gL_min = {GL_min}, gL_max = {self.GL_max}")
                    print(f"chisq = {chisq}")
                    print(f"chisq/dof = {chisq / dof}")
                    print(f"pvalue = {p}")
                    print(f"dof = {dof}")
                    print(f"N_cut = {N_cut}")
                    print(f"AIC = {AIC}")
                    print(f"prob_model = {probs[j]}")

                param_estimates = numpy.zeros((self.no_samples, n_params))

                for i in tqdm(range(self.no_samples)):
                    m_s = samples[:, i]

                    res_function = make_res_function(self.N, m_s, g_s, L_s, Bbar_s)

                    res = least_squares(res_function, self.x0, args=(cov_inv, self.model),
                                        method="lm")

                    if abs(res.x[2]) > 1:
                        print("Hellooooo")
                        res_function(res.x, cov_inv, self.model)

                    param_estimates[i] = numpy.array(res.x)

                stds.append(numpy.std(param_estimates, axis=0))

        means = numpy.array(means)
        stds = numpy.array(stds)
        probs = numpy.array(probs)

        means_overall = []
        stds_overall = []

        ## Plot the Parameter estimates
        for i, param in enumerate(self.param_names):
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
                plt.scatter(self.GL_mins, means[:, i], c=colors, edgecolors=colors, marker='_')

                for pos, ypt, err, color in zip(self.GL_mins, means[:, i], stds[:, i], colors):
                    (_, caps, _) = plt.errorbar(pos, ypt, err, ls='', color=color, capsize=5, capthick=2, marker='_', lw=2)

                plt.fill_between([min(self.GL_mins), max(self.GL_mins)], [mean - std], [mean + std], alpha=0.1, color='k')
                plt.plot([min(self.GL_mins), max(self.GL_mins)], [mean, mean], color='k', label='BMM')
                plt.title(f"{param}")
                plt.xlabel(r"$gL_{min}$")
                plt.savefig(f"Local/graphs/BMM_{param}_N{self.N}_Bbars{Bbar_s_in}.pdf")
                plt.clf()

        ## Plot the relative weights of the contributions
        if plot:
            plt.plot(self.GL_mins, probs / numpy.sum(probs), color='k')
            plt.xlabel(r"$gL_{min}$")
            plt.savefig(f"Local/graphs/BMM_probs_N{self.N}_Bbars{Bbar_s_in}.pdf")

        plt.clf()

        return means_overall, stds_overall, probs

    ## Now try and do this using all Bbar combinations
    def run_all_Bbar(self):
        means = numpy.zeros((len(self.Bbar_list), len(self.GL_mins), self.n_params))
        stds = numpy.zeros((len(self.Bbar_list), len(self.GL_mins), self.n_params))
        probs = numpy.zeros((len(self.Bbar_list), len(self.GL_mins)))

        for k, Bbar_s_in in enumerate(self.Bbar_list):
            for j, GL_min in enumerate(self.GL_mins):
                means[k], stds[k], probs[k] = self.run_Bbar_pair(Bbar_s_in, plot=False)

        # Save the results
        numpy.save(f"{self.directory}BMM_N{self.N}_all_means.npy", means)
        numpy.save(f"{self.directory}BMM_N{self.N}_all_stds.npy", stds)
        numpy.save(f"{self.directory}BMM_N{self.N}_all_probs.npy", probs)

        return means, stds, probs

    def plot_all_Bbar(self):
        try:
            means = numpy.load(f"{self.directory}BMM_N{self.N}_all_means.npy")
            stds = numpy.load(f"{self.directory}BMM_N{self.N}_all_stds.npy")
            probs = numpy.load(f"{self.directory}BMM_N{self.N}_all_probs.npy")

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


my_analysis = analysis(2, use_mutlinest=True)
my_analysis.run_Bbar_pair([0.52, 0.53])

my_analysis = analysis(4, use_mutlinest=True)
my_analysis.run_Bbar_pair([0.52, 0.53])
