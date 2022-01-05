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
from Core.parameters import param_dict, seperator, h5data_dir

for N in [2, 4]:
    print(seperator)
    print("RUNNING FOR OLD DATA SET")
    print(f'N = {N}')
    N_s = [N]
    Bbar_s = param_dict[N]["Bbar_list"]
    g_s = param_dict[N]["g_s"]
    L_s = param_dict[N]["L_s"]
    Bbar_s = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
    h5_data_file = f"{h5data_dir}Bindercrossings.h5"
    h5_data_file2 = f"{h5data_dir}Bindercrossings_new_old_bounds.h5"

    with h5py.File(h5_data_file, 'r') as f:
        with h5py.File(h5_data_file2, 'r') as f2:
            for g in g_s:
                p_s = []

                for L in L_s:
                    for Bbar in Bbar_s:
                        success = True
                        try:
                            data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}'][f'Bbar={Bbar:.3f}']['bs_bins'][()]
                            data2 = f2[f'N={N}'][f'g={g:.2f}'][f'L={L}'][f'Bbar={Bbar:.3f}']['bs_bins'][()]

                        except Exception:
                            success = False

                        if success:
                            W, p = shapiro(data)
                            if p < 10 ** -5:
                                bins = plt.hist(data, label='Old Data', bins=20)
                                plt.hist(data2, label='New Data', bins=bins[1])
                                plt.title(f"g = {g}, L = {L}, Bbar = {Bbar}: p = {p}")
                                plt.legend()
                                plt.show()
