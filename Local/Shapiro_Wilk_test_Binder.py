import numpy
import sys
import os
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.model_definitions import *
from Core.bayesian_functions import *
from Core.parameters import param_dict, seperator, h5_dict

for N in [2, 3, 4]:
    print(seperator)
    print(f'N = {N}')
    N_s = [N]
    Bbar_s = param_dict[N]["Bbar_list"]
    g_s = param_dict[N]["g_s"]
    L_s = param_dict[N]["L_s"]
    L_s = [96]
    g_s = [0.1]
    N_s = [2]
    Bbar_s = [0.55]
    h5_data_file = f"h5data/Bindercrossings.h5"
    with h5py.File(h5_data_file, 'r') as f:
        for g in g_s:
            p_s = []

            for L in L_s:
                for Bbar in Bbar_s:
                    success = True
                    try:
                        data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}'][f'Bbar={Bbar:.3f}']['bs_bins'][()]

                    except Exception:
                        success = False

                    if success:
                        W, p = shapiro(data)
                        if p < 10 ** -5:
                            print(f"g = {g}, L = {L}, Bbar = {Bbar}: p = {p}")
                            plt.hist(data)
                            pdb.set_trace()

    h5_data_file = f"h5data/Binder_N2_data.h5"
    with h5py.File(h5_data_file, 'r') as f:
        for g in g_s:
            p_s = []

            for L in L_s:
                for Bbar in Bbar_s:
                    success = True
                    try:
                        data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}'][f'Bbar={Bbar:.3f}']['bs_bins'][()]

                    except Exception:
                        success = False

                    if success:
                        W, p = shapiro(data)
                        if p < 10 ** -5:
                            print(f"g = {g}, L = {L}, Bbar = {Bbar}: p = {p}")
                            plt.hist(data)