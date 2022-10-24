import pdb
import h5py
import numpy
import sys
import os
from multiprocessing import Pool


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.Binderanalysis import Critical_analysis
from Core.parameters import *
import scipy.optimize as opt


def test_binder():
    # Try to reproduce results from random selection of 10 Binder crossings
    s = 10
    boot_no = 5  # Number of bootstrap samples to test
    widths = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    Ns = [2, 3, 4, 5]
    numpy.random.seed(438243)

    for i in range(s):
        # Obtain a random choice of the saved results
        w = numpy.random.choice(widths)
        N_ = numpy.random.choice(Ns)

        # width=0.4,   N=3,   g=0.60,   L=128,    Bbar=0.460
        # w = 0.8

        filename = f'h5data/width/width_{w:.1f}.h5'
        f = h5py.File(filename)
        print(filename)
        print(w)

        # N_ = 5

        gs = param_dict[N_]["g_s"]
        Ls = param_dict[N_]["L_s"]
        Bbars = param_dict[N_]["Bbar_list"]

        g = numpy.random.choice(gs)
        L = numpy.random.choice(Ls)
        Bbar = numpy.random.choice(Bbars)

        # g = 0.6
        # L = 96
        # Bbar = 0.52

        print('===============================================')
        print(f'width={w:.1f},   N={N_},   g={g:.2f},   L={L},    Bbar={Bbar:.3f}')

        key = 'N=%d/g=%.2f/L=%d/Bbar=%.3f' % (N_, g, L, Bbar)

        central = f[key]['central'][()]
        bins = f[key]['bs_bins'][()]
        f.close()

        # Attempt to reproduce those results
        iLin = numpy.where(numpy.array(Ls) == int(L))[0][0]

        # instantiate analysis class
        run1 = Critical_analysis(N_, g, Ls[iLin], width=w)

        # choose Bbar
        run1.Bbar = Bbar

        # load simulation data for current value of N, g, L
        run1.h5load_data()

        # Determine interval over which solver runs. Choose range between smallest/largest simulated
        # m^2 or, if defined in parameters.py, custom choice
        xmin = min(run1.actualm0sqlist)
        xmax = max(run1.actualm0sqlist)

        sn = 'su%d_%g' % (N_, g)

        # tailored min and max to extrapolate beyond simulatedpoints by means of reweighting
        if sn in mlims:
            if mlims[sn][iLin]:
                xmin = -mlims[sn][iLin][1]
                xmax = -mlims[sn][iLin][0]

        mmin, mmax = -xmax, -xmin

        # compute integrated autocorrelation time for basic quantities for each ensemble and bin data
        # correspondingly
        run1.compute_tauint_and_bootstrap_indices()

        # Fill list with trivial L1 bootstrap indices (i.e. 0,1,2,3,5,...)
        # for central value and use L0_bsindices for the inner bootstrap
        run1.rng_nest.seed(int(189123 * run1.L / 3 * run1.g * run1.N))
        L1bs = []

        for j in range(len(run1.actualm0sqlist)):
            N = run1.phi2[str(j)].shape[0]
            L1bs.append(numpy.arange(int(numpy.floor(N / run1.Nbin_tauint[j]))))

        L0bs = run1.refresh_L0_bsindices()

        # compute central value for crossing
        print(f'sn = {sn}')
        print(f'iLin = {iLin}')
        print(f'xmin = {xmin}')
        # print(L0bs)
        # print(L1bs)

        mcsq, r = opt.brentq(run1.reweight_Binder, mmin, mmax, args=(L1bs, L0bs),
                                full_output=True, xtol=1e-6)

        print(f'Central Old Value: {central}')
        print(f'Central New Value: {mcsq}')

        assert mcsq == central

        ms = numpy.zeros(boot_no)

        for i in range(boot_no):  # start boostrap
            # assign the set of L0 bootstrap indices for the ith L1 boostrap sample
            L1bs = run1.L1_bsindices[i]

            if not run1.freeze_inner_boot:
                L0bs = run1.refresh_L0_bsindices()

            ms[i], r = opt.brentq(run1.reweight_Binder, mmin, mmax, args=(L1bs, L0bs),
                                    full_output=True, xtol=1e-6)

            print(f'Bootstrap = {i}')
            print(f'Old Value : {bins[i]}')
            print(f'New Value : {ms[i]}')

        assert numpy.array_equal(ms, bins[:boot_no])


test_binder()
