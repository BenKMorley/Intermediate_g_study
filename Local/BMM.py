import enum
import sys
import os

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')
from Core.model_definitions import *
from Core.bayesian_functions import run_pymultinest, likelihood_maker


from numpy.lib import RankWarning
from scipy.optimize import least_squares
from tqdm import tqdm
import matplotlib.pyplot as plt
import pymultinest


class analysis(object):
    def __init__(self, N, Gaussian_priors={}, label=""):
        self.N = N
        self.input_h5_file = "h5data/Bindercrossings.h5"
        self.N_s_in = [N]
        self.g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
        self.directory = "Data/"
        self.MULTINEST_directory = f"{self.directory}MULTINEST/"
        self.label = label

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

        # Gaussian Priors which change the chi-squared calculation
        self.Gaussian_priors = Gaussian_priors

    def prior_penalty(self, N, params):
        for i, param in enumerate(params):
            if self.prior_lowers[i] > params[i]:
                return numpy.inf

            if self.prior_uppers[i] < params[i]:
                return numpy.inf

        return 0

    def make_res_function(self, m_s, g_s, L_s, Bbar_s, cov_inv):
        def res_function(x, return_details=False):
            # Caculate the residuals between the model and the data
            predictions = self.model(self.N, g_s, L_s, Bbar_s, *x)

            # Add an infinite penalty for being outside of the priors
            predictions += self.prior_penalty(self.N, x)

            residuals = m_s - predictions

            if abs(sum(residuals)) == numpy.inf:
                normalized_residuals = residuals

            else:
                normalized_residuals = numpy.dot(cov_inv, residuals)

            # Add the Gaussian priors to the residuals
            for param in self.Gaussian_priors:
                idx = numpy.argwhere(numpy.array(self.param_names) == param)[0][0]

                mu, sigma = self.Gaussian_priors[param]
                penalty = (x[idx] - mu) / sigma

                normalized_residuals = numpy.append(normalized_residuals, penalty)

            if return_details:
                return normalized_residuals, self.N, g_s, L_s, Bbar_s, x

            else:
                return normalized_residuals

        return res_function

    def run_Bbar_pair(self, Bbar_s_in, print_info=True, plot=True, points=400, use_multinest=False):
        means = numpy.zeros((len(self.GL_mins), self.n_params))
        stds = numpy.zeros((len(self.GL_mins), self.n_params))
        probs = numpy.zeros(len(self.GL_mins))
        chisqs = numpy.zeros(len(self.GL_mins))
        AICs = numpy.zeros(len(self.GL_mins))

        # Try loading in all the data so we know how many points are being cut out
        samples_full, g_s_full, L_s_full, Bbar_s_full, m_s_full = load_h5_data(
            self.input_h5_file, self.N_s_in, self.g_s_in, self.L_s_in, Bbar_s_in, 0, self.GL_max)

        for j, GL_min in enumerate(self.GL_mins):
            samples, g_s, L_s, Bbar_s, m_s = load_h5_data(self.input_h5_file, self.N_s_in, self.g_s_in, self.L_s_in,
                                                          Bbar_s_in, GL_min, self.GL_max)

            N_cut = len(g_s_full) - len(g_s)

            cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
            cov_1_2 = numpy.linalg.cholesky(cov_matrix)
            cov_inv = numpy.linalg.inv(cov_1_2)

            res_function = self.make_res_function(m_s, g_s, L_s, Bbar_s, cov_inv)

            if use_multinest:
                analysis, best_fit = self.run_pymultinest(res_function, GL_min, return_analysis_small=True,
                                                          n_live_points=points)

                # Use the Bayesian Evidence directly with a correction for N_cut
                probs[j] = numpy.exp(analysis[0] - N_cut)
                means[j] = best_fit['parameters']

                # Assume the distribution to be symmetric
                sigma_1_range = analysis[2]
                for i in range(self.n_params):
                    stds[j, i] = 0.5 * (max(sigma_1_range[i]) - min(sigma_1_range[i]))

                if self.print_info:
                    print(f"prob_model = {probs[j]}")

            else:
                res = least_squares(res_function, self.x0, method="lm")

                normalized_residuals = res_function(res.x)
                chisq = numpy.sum(normalized_residuals ** 2)
                n_params = len(res.x)
                dof = g_s.shape[0] - n_params
                p = chisq_pvalue(dof, chisq)
                means[j] = res.x
                AIC = chisq + 2 * n_params + 2 * N_cut
                probs[j] = numpy.exp(-0.5 * AIC)
                chisqs[j] = chisq
                AICs[j] = AIC

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

                    res_function = self.make_res_function(m_s, g_s, L_s, Bbar_s, cov_inv)

                    res = least_squares(res_function, self.x0, method="lm")

                    param_estimates[i] = numpy.array(res.x)

                stds[j] = numpy.std(param_estimates, axis=0)

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

                plt.fill_between([min(self.GL_mins), max(self.GL_mins)], [mean - std], [mean + std],
                                  alpha=0.1, color='k', label="BMM Fit")

                # Use the gL_min = 12.8 point for reference
                j = numpy.argwhere(numpy.abs(self.GL_mins - 12.8) < 10 ** -10)
                plt.fill_between([min(self.GL_mins), max(self.GL_mins)], [means[j, i][0][0] - stds[j, i][0][0]],
                                 [means[j, i][0][0] + stds[j, i][0][0]], alpha=0.1, color='k', label="Central Fit")

                plt.plot([min(self.GL_mins), max(self.GL_mins)], [mean, mean], color='k', label='BMM')
                plt.title(f"{param}")
                plt.xlabel(r"$gL_{min}$")
                plt.savefig(f"Local/graphs/BMM_{param}_N{self.N}_Bbars{Bbar_s_in}_{self.label}.pdf")
                plt.clf()

        ## Plot the relative weights of the contributions
        if plot:
            plt.plot(self.GL_mins, probs / numpy.sum(probs), color='k')
            plt.xlabel(r"$gL_{min}$")
            plt.savefig(f"Local/graphs/BMM_probs_N{self.N}_Bbars{Bbar_s_in}_{self.label}.pdf")

        plt.clf()

        ## Plot the chisq and the AIC
        if plot:
            plt.plot(self.GL_mins, chisqs, color='b', label='chisq')
            plt.xlabel(r"$gL_{min}$")
            plt.ylim(0, 50)
            plt.legend()
            plt.savefig(f"Local/graphs/BMM_chisq_N{self.N}_Bbars{Bbar_s_in}_{self.label}.pdf")

        plt.clf()

        if plot:
            plt.plot(self.GL_mins, AICs, color='b', label='AIC')
            plt.xlabel(r"$gL_{min}$")
            plt.ylim(100, 150)
            plt.legend()
            plt.savefig(f"Local/graphs/BMM_AIC_N{self.N}_Bbars{Bbar_s_in}_{self.label}.pdf")

        plt.clf()

        return means, stds, probs

    def freq_priors_multinest(self, Bbar_s_in):
        means, stds, probs = self.run_Bbar_pair(Bbar_s_in)

        sigmas = 10 ** numpy.linspace(-2, 4, 10)

        for sigma in sigmas:
            gauss_priors = {}
            for i, param in enumerate(self.param_names):
                gauss_priors[param] = (means[i], stds[i] * sigma)

            self.Gaussian_priors = gauss_priors

            x, y, probs = self.run_Bbar_pair(Bbar_s_in, sigma, use_multinest=True)

            plt.plot(self.GL_mins, probs, label=f"sigma = {sigma}")

        plt.show()

    ## Now try and do this using all Bbar combinations
    def run_all_Bbar(self):
        means = numpy.zeros((len(self.Bbar_list), len(self.GL_mins), self.n_params))
        stds = numpy.zeros((len(self.Bbar_list), len(self.GL_mins), self.n_params))
        probs = numpy.zeros((len(self.Bbar_list), len(self.GL_mins)))

        for k, Bbar_s_in in enumerate(self.Bbar_list):
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

    def run_pymultinest(self, res_function, GL_min, prior_name="", n_live_points=400,
                        INS=False, clean_files=True, sampling_efficiency=0.3,
                        return_analysis_small=False, tag="", keep_GLmax=True, seed=-1,
                        param_names_latex=None):
        directory = self.MULTINEST_directory

        # Calculate log-likelihood and prior functions for use by MULTINEST
        def loglike(cube, ndim, nparams):
            params = []
            for i in range(nparams):
                params.append(cube[i])

            normalized_residuals = res_function(params)
            chisq = numpy.sum(normalized_residuals ** 2)

            return -0.5 * chisq

        def prior(cube, ndim, nparams):
            for i in range(len(self.prior_range)):
                cube[i] = cube[i] * (self.prior_range[i][1] - self.prior_range[i][0]) +\
                                     self.prior_range[i][0]

        # The filenames are limited to 100 charachters in the MUTLTINEST code, so
        # sometimes it is necessary to save charachters
        if keep_GLmax:
            basename = (f"{directory}{self.model.__name__}{tag}_prior{prior_name}" +
                        f"_N{self.N}_GLmin{GL_min:.1f}_GLmax{self.GL_max:.1f}" +
                        f"_p{n_live_points}")
        else:
            basename = (f"{directory}{self.model.__name__}{tag}_prior{prior_name}_N" +
                        f"{self.N}_GLmin{GL_min:.1f}_p{n_live_points}")

        # Save the priors into a file
        pickle.dump(self.prior_range, open(f"{directory}{tag}priors_{prior_name}_N{self.N}" +
                                    f"_GLmin{GL_min:.1f}_GLmax{self.GL_max:.1f}" +
                                    f"_p{n_live_points}.pcl", "wb"))

        # Run the MULTINEST sampler
        print("##################################################################")
        print(f" gL_min = {GL_min}, gL_max = {self.GL_max}, model = {self.model.__name__}")
        print("Initiating MULTINEST")
        pymultinest.run(loglike, prior, self.n_params,
                        outputfiles_basename=basename, resume=False,
                        n_live_points=n_live_points,
                        sampling_efficiency=sampling_efficiency,
                        evidence_tolerance=0.1, importance_nested_sampling=INS,
                        seed=seed)
        print("MULTINEST run complete")

        # save parameter names
        f = open(basename + '.paramnames', 'w')
        for i in range(len(self.param_names)):
            if param_names_latex is None:
                f.write(f"{self.param_names[i]}\n")
            else:
                f.write(f"{self.param_names[i]}\t{param_names_latex[i]}\n")

        f.close()

        # Save the prior ranges
        f = open(basename + '.ranges', 'w')
        for i in range(len(self.param_names)):
            f.write(f"{self.param_names[i]} {self.prior_range[i][0]} {self.prior_range[i][1]}\n")

        f.close()

        # Also save as a json
        # json.dump(self.param_names, open(f'{basename}params.json', 'w'))

        # Get information about the MULTINEST run
        analysis = pymultinest.Analyzer(outputfiles_basename=basename,
                                        n_params=self.n_params)
        stats = analysis.get_stats()

        # Extract the log-evidence and its error
        E, delta_E = stats['global evidence'], stats['global evidence error']

        # Extract parameter estimates from the posterior distributions
        sigma_1_range = [analysis.get_stats()['marginals'][i]['1sigma'] for i in
                        range(self.n_params)]
        sigma_2_range = [analysis.get_stats()['marginals'][i]['2sigma'] for i in
                        range(self.n_params)]
        median = [analysis.get_stats()['marginals'][i]['median'] for i in
                  range(self.n_params)]

        # Extract the points sampled by MULTINEST
        posterior_data = analysis.get_equal_weighted_posterior()

        # Find the parameter estimates at the MAP
        best_fit = analysis.get_best_fit()

        # Collate data for saving/returning
        analysis_data = [E, delta_E, sigma_1_range, sigma_2_range, posterior_data,
                        median]

        if not clean_files:
            pickle.dump(analysis_data, open(f"{basename}_analysis.pcl", "wb"))

        # Make a cut down version for the purpose of quicker transfer
        analysis_small = [E, delta_E, sigma_1_range, sigma_2_range, median]

        pickle.dump(analysis_small, open(f"{basename}_analysis_small.pcl", "wb"))

        if clean_files:
            # Remove the remaining saved files to conserve disk space
            os.popen(f'rm {basename}ev.dat')
            os.popen(f'rm {basename}live.points')
            os.popen(f'rm {basename}.paramnames')
            # os.popen(f'rm {basename}params.json')
            os.popen(f'rm {basename}phys_live.points')
            os.popen(f'rm {basename}post_equal_weights.dat')
            os.popen(f'rm {basename}post_separate.dat')
            os.popen(f'rm {basename}.ranges')
            os.popen(f'rm {basename}resume.dat')
            os.popen(f'rm {basename}stats.dat')
            os.popen(f'rm {basename}summary.txt')
            os.popen(f'rm {basename}.txt')

        return analysis_small, best_fit
