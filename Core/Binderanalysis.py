###############################################################################
# Source file: ./Binderanalysis.py
#
# Copyright (C) 2020
#
# Author: Andreas Juettner juettner@soton.ac.uk
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
###############################################################################

# custom libraries for this project
from parameters import *
from MISC import *

# generic python libraries
import numpy as np
import scipy.optimize as opt
import random
import h5py
import os
import sys
import pdb
from parse import parse
import warnings
import matplotlib.pyplot as plt

# strict: raise exceptions in case of warnings
np.seterr(all='warn')
warnings.filterwarnings('error')


# begin class definition ######################################################
class Critical_analysis():
    """
        Critical_analysis() takes MCMC data for phi^2, M^2, M^4, constructes
        the Binder cumulant and determines its crossing points with a
        particular choice Bbar under reweighting. The code has been used for
        the analysis presented in "Nonperturbative infrared finiteness in
        super-renormalisable scalar quantum field theory"
        https://arxiv.org/abs/2009.14768

        List of methods:
        * h5load_data(): Loads MCMC simulation data for given values of ag,
            L / a and N from hdf5 data file
        * compute_tauint_and_bootstrap_indices(): Computes the integrated
            autocorrelation time for primary observables and determines
            required binsizes and bins primary data into bins
        * compute_overlap(m): Computes the overlap of the MC distribution of
            phi^2 on available ensembles with the value of <phi^2> at m^2 if it
            exists. Otherwise the value of <phi^2> at m^2 is estimated from a
            linear inter/extrapolation from simulation data
        * RWbinit_N(dat,Nbinl):  Bins MCMC data in dat into bins of size Nbinl
        * refresh_nested_bsindices():   Computes boostrap indices for L0 and L1
            boostrap (for details see header of function "reweigth_Binder" and
            "find_Binder_crossing")
        * bootit(f,dat): Computes boostrap of function f on data in dat
        * bootit_fix(f,dat,bsindices):
        * B(x): Helper function that computes <M^2>^2/<M^4> under reweighting
        * reweight_Binder(m,ind,nested_ind): Computes the Binder cumulant at
            m^2 by reweighting of existing simulation data. ind and nested_ind
            are boostrap indices for the L0 and L1 bootstrap
        * find_Binder_crossing(mmax,mmin):
            Finds the crossing point between B(m^2) and a fixed value of
            self.Bbar. The search is restricted to mmin < m^2 < mmax
        * h5store_results(self): Writes result to hdf5 file
    """

    def __init__(self, N, g, L, direc='h5data/', filename='MCMC_data_full.h5', Nboot=500):
        """
            Here we allocate some basic variables and seed the RNG
        """
        np.random.seed(int(N * g * 912123))

        self.N = N
        self.g = g
        self.L = L
        self.phi2 = {}
        self.M2 = {}
        self.M4 = {}
        self.actualm0sqlist = []
        self.actualNm0sq = 0
        self.Ntraj = []

        # time std of phi^2 distribution to include in reweighting
        self.frac_of_dist = 1
        self.Nbin_tauint = []
        self.tauint = []
        self.Nbin_min = 50
        self.rng_nest = np.random.RandomState()
        self.Nboot = Nboot
        self.msq_final = 0.
        self.dmsq_final = 0.
        self.datadir = direc
        self.MCMCdatafile = filename
        self.resultsfile = 'Bindercrossings.h5'
        self.therm = 10000  # Configurations to remove due to thermalization

    def h5load_data(self):
        """
            import MCMC data from from hd5 data file
        """
        # open data file
        filename = self.datadir + self.MCMCdatafile
        if os.path.isfile(filename):     # check if file exists, exit if not
            f = h5py.File(filename, 'r')

        else:
            print("Can't find data file ", filename)
            exit()

        # some IO to command line
        print(separator)
        print("Loading data for N=%d, ag=%.2f, L/a=%d" % (self.N, self.g,
                                                          self.L))
        print(separator)

        # get hdf5 group and determine available simulated masses
        dat = f.get('N=%d/g=%.2f/L=%d' % (self.N, self.g, self.L))

        # Only extract data points in the range we're interested in
        sn = 'su%d_%g' % (self.N, self.g)
        iLin = np.where(np.array(Ls) == int(self.L))[0][0]
        if sn in mlims:
            if mlims[sn][iLin]:
                xmin = -mlims[sn][iLin][1]
                xmax = -mlims[sn][iLin][0]

            else:
                xmin = - np.inf
                xmax = + np.inf

        else:
            xmin = - np.inf
            xmax = + np.inf

        # list of input bare masses
        self.actualm0sqlist = [float(parse('msq={}', i)[0]) for i in dat if
                               float(parse('msq={}', i)[0]) < xmax and
                               float(parse('msq={}', i)[0]) > xmin]

        # number of masses
        self.actualNm0sq = len(self.actualm0sqlist)

        # now loop over available masses and, in each case, assign values for
        # M^2, M^4 and phi^2 measurements to class variables
        print("Found %d msq values:" % (len(self.actualm0sqlist)))

        for ii in range(self.actualNm0sq):
            self.M2[str(ii)], self.M4[str(ii)], self.phi2[str(ii)] = \
                f.get('N=%d/g=%.2f/L=%d/msq=%.8f' % (self.N, self.g, self.L,
                                                     self.actualm0sqlist[ii]))

            # Remove thermalization steps
            self.M2[str(ii)] = self.M2[str(ii)][self.therm:]
            self.M4[str(ii)] = self.M4[str(ii)][self.therm:]
            self.phi2[str(ii)] = self.phi2[str(ii)][self.therm:]

            self.Ntraj.append(len(self.M2[str(ii)]))

            print(ii, "with %7d trajectories" % (len(self.M2[str(ii)])))

        print("Data loaded")
        print(separator)

    def compute_tauint_and_bootstrap_indices(self):
        """
            Compute integrated autocorrelation time based on Uli Wolff's
            `Monte Carlo Errors with less errors`
            https://arxiv.org/abs/hep-lat/0306017
        """
        print(separator)
        print("Computing max. tauint of M^2, M^4, phi^4: ")

        for i in range(self.actualNm0sq):
            # compute tau_int for M^2
            uwfn = UWerr(np.array(self.M2[str(i)]).T, 1.5, '')
            uwres_M2 = uwfn.doit()

            # compute tau_int for M^4
            uwfn = UWerr(np.array(self.M4[str(i)]).T, 1.5, '')
            uwres_M4 = uwfn.doit()

            # compute tau_int for phi^2
            uwfn = UWerr(np.array(self.phi2[str(i)]).T, 1.5, '')
            uwres_phi2 = uwfn.doit()

            # compute largest tau_int
            tau_intl = int(np.max([uwres_M2[3], uwres_M4[3], uwres_phi2[3]]))
            self.tauint.append(tau_intl)

            # set a minimum binsize of Nbin_min, otherwise 4*tau_int
            if 4 * tau_intl <= self.Nbin_min:
                tau_intl = self.Nbin_min
            else:
                tau_intl = 4 * tau_intl

            self.Nbin_tauint.append(tau_intl)

            print("msq=%.7f (tau_int)_max=%.0f using bin-size %d" %
                  (self.actualm0sqlist[i], self.tauint[-1], tau_intl))

        print(separator)

        # now that we know the binning size we can generate boostrap indices
        # for all bins of L1 boostratp for central value
        self.L1_bsindices = []

        for i in range(self.Nboot):  # start boostrap
            L1tmp = []

            # generate boostrap indices for all input masses
            for j in range(len(self.actualm0sqlist)):
                N = self.phi2[str(j)].shape[0]  # number of trajectories
                Nbins = int(np.floor(N / self.Nbin_tauint[j]))  # number of bins
                L1tmp.append(np.random.randint(0, Nbins, size=(Nbins)))

            self.L1_bsindices.append(L1tmp)

        print("Generated bootstrap indices")
        print(separator)

    def plot_tr_phi2_distributions(self):
        plt.close('all')
        fig, ax = plt.subplots()

        for i in range(self.actualNm0sq):
            ax.hist(self.phi2[str(i)], bins=20, label=rf"m^2 = m{self.actualm0sqlist[i]}", alpha=0.4)

        plt.legend()

        return fig, ax

    def compute_overlap(self, msq):
        """
            This is relevant for reweighting:
            Determine overlap of phi^2-distributions of simulation points with
            various m^2 in order to assess whether reweighting appropriate. It
            computes list with indices in m0sqlist which lie within acceptable
            radius which can be set by class variable self.frac_of_dist. An
            ensemble enters the reweighting if its distribution of phi^2
            overlaps with the target-mass within self.frac_of_dist*sigma
        """
        prefac = (self.L ** 3) * (self.N / self.g)

        # compute mean value and standard deviation of phi^2
        self.simulated = [np.mean(self.phi2[str(i)]) for i in
                          range(self.actualNm0sq)]
        self.dsimulated = [np.std(self.phi2[str(i)]) for i in
                           range(self.actualNm0sq)]

        # determine nearest simulation points
        above = np.where(msq > -np.array(self.actualm0sqlist))[0]
        below = np.where(msq <= -np.array(self.actualm0sqlist))[0]
        indexists = np.where(-1. * msq == np.array(self.actualm0sqlist))[0]

        # now interpolate/extrapolate central value of phi^2 in the bare in
        # put mass
        if len(indexists) > 0:
            # if m agrees with simulated point then take just this and do
            # nothing
            res1 = np.array(flatten([self.simulated[i] for i in indexists]))

        else:
            # otherwise interpolate/extrapolate <phi2> to simulation point
            if len(above) > 0 and len(below) > 0:  # interpolate
                alpha = (self.simulated[above[0]] - self.simulated[below[-1]])\
                    / (self.actualm0sqlist[below[-1]] -
                       self.actualm0sqlist[above[0]])

                res1 = self.simulated[below[-1]] + \
                    alpha * (self.actualm0sqlist[below[-1]] + msq)

            elif len(below) == 0 and len(above) > 0:  # extapolate
                alpha = (self.simulated[above[1]] - self.simulated[above[0]])\
                    / (self.actualm0sqlist[above[0]] -
                       self.actualm0sqlist[above[1]])

                res1 = self.simulated[above[0]] + \
                    alpha * (self.actualm0sqlist[above[0]] + msq)

            elif len(below) > 0 and len(above) == 0:  # extrapolate
                alpha = (self.simulated[below[-1]] -
                         self.simulated[below[-2]]) \
                    / (self.actualm0sqlist[below[-2]] -
                       self.actualm0sqlist[below[-1]])

                res1 = self.simulated[below[-1]] +\
                    alpha * (self.actualm0sqlist[below[-1]] + msq)

        # now determine which phi^2 distributions of simulated bare masses
        # overlap with the one we just inter/extrapolated to within fraction
        # self.frac_of_dist of the standard deviation
        x = np.array(self.simulated)
        dx = np.array(self.dsimulated)
        ind = np.where((res1 < x + self.frac_of_dist * dx) &
                       (res1 > x - self.frac_of_dist * dx))[0]

        return ind

    def RWbinit_N(self, dat, Nbinl):
        """
            bin data after reweighting explicit binning as arg ##3
            - dat is Ntraj x Nobs array
            - Nbinl is the number of trajectories to be binned into one bin
        """
        # Determine number of bins
        Nbins = np.floor(dat.shape[0] / Nbinl)

        # do the binning
        newdat = np.array([])
        if Nbinl == 1:
            newdat = dat

        else:
            for n in range(int(Nbins)):
                newdat = np.r_[newdat, np.array([np.mean(dat[n * Nbinl:(n + 1)
                                                               * Nbinl], 0)])]

        return newdat

    def refresh_L0_bsindices(self):
        """
            Generate new set of L0 boostrap indices
        """
        rn = []

        # generate boostrap indices for all input masses
        for j in range(len(self.actualm0sqlist)):
            N = self.phi2[str(j)].shape[0]  # number of trajectories
            Nbins = int(np.floor(N / self.Nbin_tauint[j]))  # number of bins

            rn.append(self.rng_nest.randint(0, Nbins,
                                            size=(Nbins, self.Nboot)))

        return rn

    def bootit(self, f, dat, bsindices, *args, **kwargs):
        """
            A simple boostrap routine, takes function f and applies dat
            - f is function
            - dat is Nmeas x Nobs array
            - bsindices is Nboot x Nobs array
            Returns central value and standard deviation
        """
        res = f(dat)
        res_bs = np.array([])
        N = dat.shape[0]
        av0s = np.zeros(self.Nboot)
        av1s = np.zeros(self.Nboot)
        av2s = np.zeros(self.Nboot)

        for i in range(self.Nboot):
            wol = bsindices[:, i]
            # kwargs = {'return_extra': True}
            kwargs = {}

            resl  = f(dat[wol, :], *args, **kwargs)
            # resl, av0, av1, av2 = f(dat[wol, :], *args, **kwargs)

            # av0s[i] = av0
            # av1s[i] = av1
            # av2s[i] = av2

            res_bs = np.r_[res_bs, np.array([resl])]

        if res == np.nan:
            return np.nan, np.nan

        else:
            dres = np.sqrt(np.real(np.sum((np.array(res_bs) - res)**2, 0))) \
                / np.sqrt(self.Nboot)

            return res, dres

    def B(self, x, return_extra=False):
        """
            Helper function: Compute the ratio in the Binder cumulant including
            reweighting
            - x is N x 3 array with
                1st column <reweighting factor>
                2nd column <M2> reweighted
                3rd column <M5> reweighted
        """
        # the reweighting factor can get huge and require arithmetics beyond
        # double precision fpa in this case an exception is raised adn np.nan
        # returned to be dealt with later
        if return_extra:
            try:
                av = np.mean(x, 0)

                B = av[0] * av[2] / av[1] ** 2

                if B > 1:
                    print("Hello")

                return B, av[0], av[1], av[2]

            except Exception:
                return np.nan

        else:
            try:
                av = np.mean(x, 0)
                return av[0] * av[2] / av[1] ** 2

            except Exception:
                return np.nan

    def reweight_Binder(self, msq, L1bs, L0bs, sigma=False):
        """
            This function computes the reweighted Binder cumulant
            - msq is target bare mass squared for for reweighting
            - L1bs/L0bs are L1/L0 bootstrap indices
        """
        # compute reweighting factor for each ensemble, then reweight, then bin
        iinclude = self.compute_overlap(msq)

        res = []
        dres = []
        # loop over masses to be included in current reweighting
        for i in iinclude:
            # In the following:
            # the reweighting factor can get huge and require arithmetics
            # beyond double precision fpa in this case an exception is raised
            # and np.nan returned to be dealt with later. The nan values are
            # filtered out later and excluded from the analysis

            # compute the unbinned reweighting factor
            try:
                RWfac = np.exp(-(msq + self.actualm0sqlist[i]) * (self.L**3) *
                                (self.N / self.g) *
                               np.array(self.phi2[str(i)]))

            except Exception:
                RWfac = np.nan * np.array(self.phi2[str(i)])

            # Flag contributions where the exponent is so negative the exponent
            # goes to 0, or so large it goes to infinity.
            if min(RWfac == 0) or max(RWfac == np.inf):
                RWfac = np.nan * np.array(self.phi2[str(i)])

            # binning for reweighting factor, phi^2, M^2 and M^4
            try:
                RW_fac_bin = self.RWbinit_N(RWfac, self.Nbin_tauint[i])

            except Exception:
                RW_fac_bin = self.RWbinit_N(np.nan * RWfac, self.Nbin_tauint[i])

            try:
                RW_2_bin = self.RWbinit_N(RWfac * np.array(self.M2[str(i)]),
                                          self.Nbin_tauint[i])

            except Exception:
                RW_2_bin = self.RWbinit_N(np.nan * np.array(self.M2[str(i)]),
                                          self.Nbin_tauint[i])

            try:
                RW_4_bin = self.RWbinit_N(RWfac * np.array(self.M4[str(i)]),
                                          self.Nbin_tauint[i])
            except Exception:
                RW_4_bin = self.RWbinit_N(np.nan * np.array(self.M4[str(i)]),
                                          self.Nbin_tauint[i])

            # collate binned quantities into array
            datl = np.array([])
            datl = np.array([RW_fac_bin])              # reweighting factor
            datl = np.r_[datl, np.array([RW_2_bin])]    # <M2>
            datl = np.r_[datl, np.array([RW_4_bin])]    # <M4>

            # pick data for current bootstrap (L1 bootstrap indices)
            # can only be done after data combined with
            # reweighting factor
            datl = datl[:, L1bs[i]]

            # run the bootstrap for the Binder comulant with L0 bootstrap
            # indices
            resl, dresl = self.bootit(self.B, datl.T, L0bs[i])

            res.append(resl)
            dres.append(dresl)

        # filter out occurences of NaNs and produce weighted average over
        # reweighted results for Binder cumulant

        ## Always filter out Nan!
        dres0 = [dx for x, dx in zip(res, dres) if
                    ((not np.isnan(x)) and (dx != 0.))]

        res0 = [x for x, dx in zip(res, dres) if
                ((not np.isnan(x)) and (dx != 0.))]

        if len(res0) > 0 and len(dres0) > 0:
            denom = np.sum(1. / np.array(dres0) ** 2)
            numer = np.sum(np.array(res0) / np.array(dres0) ** 2)

            B = 1 - self.N * 1. / 3 * numer / denom

            if sigma:
                sigma = np.sqrt((1 / denom)) * self.N / 3

        else:
            B = np.nan
            simga = np.nan

        # The denominator gives the estimate of the bootstrap variance
        if sigma:
            return B - self.Bbar, sigma

        else:
            return B - self.Bbar

    def find_Binder_crossing(self, mmax, mmin):
        """
            find crossing of 1st Binder cumulant with Bbar
            under bootstrap restricted to interval mmax...mmin

            There is an inner and an outer bootstrap:
            - The outer bootstrap (layer L1)  determines the stat. error on
                the value of m^2 at the crossing of the reweighted Binder
                cumulant with Bbar.
            - The inner bootstrap (layer L0) runs for each iteration of the
                minimiser within one bootstrap iteration. In order to reweight
                to a given point of m^2 we compute the reweighted result from
                nearby simulation point under the inner bootstrap and then
                determine the value of the Binder cumulant at that point by
                means of a weighted average.
        """
        print(separator)
        print("Starting determination of crossing point of Binder cumulant"
              "with Bbar=%.2f" % (self.Bbar))

        # Fill list with trivial L1 bootstrap indices (i.e. 0,1,2,3,5,...)
        # for central value and use L0_bsindices for the inner bootstrap
        self.rng_nest.seed(int(189123 * self.L / 3 * self.g * self.N))
        L1bs = []

        for j in range(len(self.actualm0sqlist)):
            N = self.phi2[str(j)].shape[0]
            L1bs.append(np.arange(int(np.floor(N / self.Nbin_tauint[j]))))
        L0bs = self.refresh_L0_bsindices()

        # compute central value for crossing
        try:
            mcsq, r = opt.brentq(self.reweight_Binder, mmin, mmax,
                                 args=(L1bs, L0bs), full_output=True,
                                 xtol=1e-6)

        except ValueError:
            print("No crossing point found -- aborting")
            exit()

        print("Crossing point central value found at mc^2=%e" % (mcsq))

        # compute crossing under Bootstrap
        print("Now starting bootstrap with %d samples:" % self.Nboot)
        bres = []

        for i in range(self.Nboot):  # start boostrap
            # assign the set of L0 bootstrap indices for the ith L1 boostrap
            # sample
            L1bs = self.L1_bsindices[i]
            L0bs = self.refresh_L0_bsindices()
            try:
                mcsq_i, r = opt.brentq(self.reweight_Binder, mmin, mmax,
                                       args=(L1bs, L0bs), full_output=True,
                                       xtol=1e-6)

            except ValueError:
                print("No crossing point found -- aborting")
                exit()

            bres.append(mcsq_i)
            print("bs sample %4d: mc^2=%e" % (i, mcsq_i))

        # compute the BS error
        dmcsq = np.sqrt(np.real(np.sum((np.array(bres) - mcsq) ** 2, 0)))\
            / np.sqrt(self.Nboot)

        print("result %.6f +/- %.6f" % (mcsq, dmcsq))
        print(separator)

        self.msq_final = mcsq
        self.dmsq_final = dmcsq
        self.msq_bins_final = bres

    def h5store_results(self):
        """
            store final result and information on underlying data into results
            file
        """
        f = h5py.File(self.datadir + self.resultsfile, 'a')
        key = 'N=%d/g=%.2f/L=%d/Bbar=%.3f' % (self.N, self.g, self.L,
                                              self.Bbar)

        if key in f:  # remove data under key if exists
            del f[key]

        g = f.create_group(key)
        g.create_dataset('central', data=self.msq_final)
        g.create_dataset('std', data=self.dmsq_final)
        g.create_dataset('bs_bins', data=self.msq_bins_final)
        g.attrs['masses'] = self.actualm0sqlist
        g.attrs['Ntraj'] = self.Ntraj
        g.attrs['tauint'] = self.tauint

        f.close()
# end class definition ########################################################


def compute_Bindercrossing(N, g, Bbar, Lin, direc="h5data/", filename="MCMC_data_full.h5", Nboot=500):
    """
        ######################################################################
        # Main routine for computing crossing of reweighted Binder cumulant  #
        # with Bbar value.                                                   #
        #                                                                    #
        # Analysis code underlying arXiv:2009.14768 ``Nonperturbative        #
        # infrared finiteness in super-renormalisable scalar quantum field   #
        # theory'' by: Guido Cossu, Luigi Del Debbio, Andreas Juettner,      #
        # Ben Kitching-Morely, Joseph K.L. Lee, Henrique Bergallo Rocha,     #
        # Kostas Skenderis                                                   #
        #                                                                    #
        # Analysis code by Ben Kitching-Morley and                           #
        # Andreas Juettner (juettner@soton.ac.uk)                            #
        ######################################################################
    """
    print(compute_Bindercrossing.__doc__)
    check_exists('L/a', Lin)
    check_exists('ag', g)
    check_exists('N', N)
    iLin = np.where(np.array(Ls) == int(Lin))[0][0]

    # instantiate analysis class
    run1 = Critical_analysis(N, g, Ls[iLin], direc=direc, filename=filename, Nboot=Nboot)

    # choose Bbar
    run1.Bbar = Bbar

    # load simulation data for current value of N, g
    run1.h5load_data()

    # Determine interval over which solver runs. Choose range between
    # smallest/largest simulated m^2 or, if defined in parameters.py,
    # custom choice
    xmin = min(run1.actualm0sqlist)
    xmax = max(run1.actualm0sqlist)

    sn = 'su%d_%g' % (N, g)

    # tailored min and max to extrapolate beyond simulatedpoints by means of
    # reweighting
    if sn in mlims:
        if mlims[sn][iLin]:
            xmin = -mlims[sn][iLin][1]
            xmax = -mlims[sn][iLin][0]

    # compute integrated autocorrelation time for basic quantities for each
    # ensemble and bin data correspondingly
    run1.compute_tauint_and_bootstrap_indices()

    # start the actual determination of m^2 where the Binder cumulant assumes
    # the value Bbar
    run1.find_Binder_crossing(-xmin, -xmax)
    run1.h5store_results()


if __name__ == "__main__":
    # For passing in arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('N', metavar="N", type=int)
    parser.add_argument('g', metavar="ag", type=float)
    parser.add_argument('Bbar', metavar="Bbar", type=float)
    parser.add_argument('L', metavar="L / a", type=int)

    parser.add_argument('-therm', metavar="No. therm steps", type=int, default=argparse.SUPPRESS)       # No. of thermalization steps
    parser.add_argument('-Nboot', metavar="No. boot samples", type=int, default=argparse.SUPPRESS)      # No. of boot samples
    parser.add_argument('-direc', metavar="Directory", type=str, default=argparse.SUPPRESS)             # Directory
    parser.add_argument('-filename', metavar="Filename", type=str, default=argparse.SUPPRESS)               # Filename

    args = parser.parse_args()

    # Extract any optional arguments
    kwargs = vars(parser.parse_args())
    del kwargs['N']
    del kwargs['g']
    del kwargs['Bbar']
    del kwargs['L']

    ###########################################################################
    # call main routine
    compute_Bindercrossing(args.N,              # N
                           args.g,              # ag
                           args.Bbar,           # Bbar
                           args.L,              # L / a
                           **kwargs)            # If not using defaults
    ###########################################################################
