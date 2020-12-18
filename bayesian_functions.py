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
import pymultinest
import os
import pickle
import json
from multiprocessing import Pool, current_process


# Function that returns likelihood function for use in MULTINEST
def likelihood_maker(n_params, cov_inv, model, res_function):
    """
        INPUTS :
        --------
        n_params: int, Number of parameters in the fit anzatz (model)
        cov_inv: The square-root of the inverse of the covariance matrix
        model: A function that has the following callable signature
        model(N, g_s, L_s, Bbar_s, *x), where N is the rank of the field
            symmetry, g_s are the coupling constants, L_s are the lattice sizes
            and Bbar_s are the Bbar intercept values. *x are the parameters of
            the fit, and is of length n_params
        res_function: residual function: See
            model_definitions.make_res_function

        OUTPUTS :
        ---------
        loglike: Function of the log of the likelihood ready for MULTINEST
    """
    def loglike(cube, ndim, nparams):
        params = []
        for i in range(n_params):
            params.append(cube[i])

        chisq = chisq_calc(params, cov_inv, model, res_function)

        return -0.5 * chisq

    return loglike


# Function that returns a uniform prior for use in MULTINEST
def prior_maker(prior_range):
    """
        INPUTS :
        --------
        prior_range: a list of lists, or numpy array, of shape (n_params, 2),
        where the ith component contains the upper and lower bounds of the
        uniform prior of the ith parameter of the fit (floats or ints)

        OUTPUTS :
        ---------
        prior: prior function as used by MULTINEST
    """
    def prior(cube, ndim, nparams):
        for i in range(len(prior_range)):
            cube[i] = cube[i] * (prior_range[i][1] - prior_range[i][0]) +\
                                prior_range[i][0]

    return prior


def run_pymultinest(prior_range, model, GL_min, GL_max, n_params, directory,
                    N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                    prior_name="", n_live_points=400, INS=False,
                    clean_files=False, sampling_efficiency=0.3,
                    return_analysis_small=False, tag="", keep_GLmax=True,
                    seed=-1, param_names_latex=None):
    """
        See documentation for pymultinest.run()

        INPUTS :
        --------
        prior_range: Function giving uniform prior ranges for use by MULTINEST
        model: A function that has the following callable signature
            model(N, g_s, L_s, Bbar_s, *x), where N is the rank of the field
            symmetry, g_s are the coupling constants, L_s are the lattice sizes
            and Bbar_s are the Bbar intercept values. *x are the parameters of
            the fit, and is of length n_params
        GL_min: float, Minimum value of g * L that is used in the fit
        GL_max: float, Maximum value of g * L that is used in the fit
        n_params: int, number of fit parameters in the anzatz
        directory: string, Where the MULTINEST output will be stored
        N: int, rank of the SU(N) field matrices
        g_s: 1D array of floats of length s, where s is the number of data
            points used in the fit. Each element is a value of the coupling, g.
        L_s: 1D array of ints of length s, where s is the number of data points
            used in the fit. Each element is a value of the lattice size, L.
        samples: 2D array of floats of size (s, no_samples), where s is the
            number of data points used in the fit, and no_samples is the number
            of bootstrap samples used for finding the Binder crossing points.
            Each element is a value of the Binder crossing mass, m
        param_names: list of strings of length n_params. Each entry is the name
            of the corresponding parameter
        prior_name: string, useful for distinguishing results that used
            different priors
        n_live_points: int, number of points active at any time in the
            MULTINEST algorithm. Increasing this increases the accuracy and
            computational time of the algorithm
        INS: Bool, if True multinest will use importance nested sampling method
        clean_files: Bool, if True then the large data files of MULTINEST
            samples will be deleted after completion of the algorithm. This is
            useful for reducing hard-drive load if doing many runs
        sampling_efficiency: See MULTINEST documentation. float between 0 and
            1.  Here, 0.3 is used as this is the reccomended value for
            Bayesian Evidence
        return_analysis_small: Bool, if True return only key values of the run,
            explicitly... [E, delta_E, sigma_1_range, sigma_2_range, median],
            where E is the log-evidence, delta_E is the error in the
            log-evidence as estimated by MULTINEST (not always accurate),
            sigma_1_range and sigma_2_range are the 1 and 2 sigma band
            estimates of parameter values from the posterior distribution
            respectively, median is the median parameter estimate, also from
            the posterior distribution.
        tag: string, change this to label the files associated with a run
            uniquely
        keep_GLmax: Bool, if True the value of GL_max will be kept in the
            metadata of the filename. You may wish to make this False as
                MULTINEST has a 100 charachter limit on filenames, and GL_max
                isn't particularly important in the analysis done here
        seed: int, starting seed for the random MULTINEST algorithm
        param_names_latex: list of strings, gives the LaTeX expression for each
            parameter for plotting

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
    # Create the output directory if it doesn't already exist
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Obtain the covariance matrix
    cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    # Obtain a function for getting the normalized residuals
    res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

    # Calculate log-likelihood and prior functions for use by MULTINEST
    likelihood_function = likelihood_maker(n_params, cov_inv, model,
                                           res_function)
    prior = prior_maker(prior_range)

    # The filenames are limited to 100 charachters in the MUTLTINEST code, so
    # sometimes it is necessary to save charachters
    if keep_GLmax:
        basename = (f"{directory}{model.__name__}{tag}_prior{prior_name}" +
                    f"_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}" +
                    f"_p{n_live_points}")
    else:
        basename = (f"{directory}{model.__name__}{tag}_prior{prior_name}_N" +
                    f"{N}_GLmin{GL_min:.1f}_p{n_live_points}")

    # Save the priors into a file
    pickle.dump(prior_range, open(f"{directory}{tag}priors_{prior_name}_N{N}" +
                                  f"_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}" +
                                  f"_p{n_live_points}.pcl", "wb"))

    # Run the MULTINEST sampler
    Bbar_s_set = list(set(Bbar_s))
    print("##################################################################")
    print(f"Config: N = {N}, Bbar_s = [{Bbar_s_set[0]}, {Bbar_s_set[1]}],"
          f" gL_min = {GL_min}, gL_max = {GL_max}, model = {model.__name__}")
    print("Initiating MULTINEST")
    pymultinest.run(likelihood_function, prior, n_params,
                    outputfiles_basename=basename, resume=False,
                    n_live_points=n_live_points,
                    sampling_efficiency=sampling_efficiency,
                    evidence_tolerance=0.1, importance_nested_sampling=INS,
                    seed=seed)
    print("MULTINEST run complete")

    # save parameter names
    f = open(basename + '.paramnames', 'w')
    for i in range(len(param_names)):
        if param_names_latex is None:
            f.write(f"{param_names[i]}\n")
        else:
            f.write(f"{param_names[i]}\t{param_names_latex[i]}\n")

    f.close()

    # Save the prior ranges
    f = open(basename + '.ranges', 'w')
    for i in range(len(param_names)):
        f.write(f"{param_names[i]} {prior_range[i][0]} {prior_range[i][1]}\n")

    f.close()

    # Also save as a json
    json.dump(param_names, open(f'{basename}params.json', 'w'))

    # Get information about the MULTINEST run
    analysis = pymultinest.Analyzer(outputfiles_basename=basename,
                                    n_params=n_params)
    stats = analysis.get_stats()

    # Extract the log-evidence and its error
    E, delta_E = stats['global evidence'], stats['global evidence error']

    # Extract parameter estimates from the posterior distributions
    sigma_1_range = [analysis.get_stats()['marginals'][i]['1sigma'] for i in
                     range(n_params)]
    sigma_2_range = [analysis.get_stats()['marginals'][i]['2sigma'] for i in
                     range(n_params)]
    median = [analysis.get_stats()['marginals'][i]['median'] for i in
              range(n_params)]

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
        os.popen(f'rm {basename}params.json')
        os.popen(f'rm {basename}phys_live.points')
        os.popen(f'rm {basename}post_equal_weights.dat')
        os.popen(f'rm {basename}post_separate.dat')
        os.popen(f'rm {basename}.ranges')
        os.popen(f'rm {basename}resume.dat')
        os.popen(f'rm {basename}stats.dat')
        os.popen(f'rm {basename}summary.txt')
        os.popen(f'rm {basename}.txt')

    if return_analysis_small:
        return analysis_small, best_fit

    else:
        return analysis, best_fit
