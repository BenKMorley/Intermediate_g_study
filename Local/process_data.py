import h5py
import os
import re
import numpy
import sys

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.Binderanalysis import Critical_analysis
from Core.MISC import *


def process(filename, size=100001):
    if os.path.isfile(filename):
        data = h5py.File(filename, "a")

    else:
        print("File Does not exist")
        raise FileNotFoundError

    # Now we know which configurations we have available lets read them in
    for N_key in data.keys():
        dict1 = data[N_key]

        for g_key in dict1.keys():
            dict2 = dict1[g_key]

            for L_key in dict2.keys():
                N = int(re.findall(r'\d+', N_key)[0])
                g = float(re.findall(r'\d+.\d+', g_key)[0])
                L = int(re.findall(r'\d+', L_key)[0])

                Successful_Readin = False

                while not Successful_Readin:
                    System = Critical_analysis(N, g, L)
                    System.MCMCdatafile = filename
                    System.datadir = ''
                    System.min_traj = size
                    System.h5load_data()

                    value = System.compute_tauint_and_bootstrap_indices()

                    if value is None:
                        break

                    else:
                        del data[MCMC_conv_N(N)][MCMC_conv_g(g)][MCMC_conv_L(L)][MCMC_conv_m(-value)]
