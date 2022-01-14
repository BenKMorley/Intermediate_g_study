import numpy
import sys
import os
import re
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# Import from the Core directory
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.model_definitions import *
from Core.bayesian_functions import *
from Core.parameters import param_dict, seperator, h5data_dir


def count_non_normal(filename, N):
    total = 0
    p0_05 = 0
    p0_01 = 0
    p0_001 = 0
    p0_0001 = 0
    p0_00000000001 = 0

    with h5py.File(filename, 'r') as f:
        # We only want to analyse a single N
        assert f'N={N}' in f.keys(), "The data file needs to contain data for this N"

        data = f[f'N={N}']
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

                    samples = B_data['bs_bins'][()]
                    W, p = shapiro(samples)

                    total += 1

                    if p < 0.05:
                        p0_05 += 1
                    if p < 0.01:
                        p0_01 += 1
                    if p < 0.001:
                        p0_001 += 1
                    if p < 0.0001:
                        p0_0001 += 1
                    if p < 0.00000000001:
                        p0_00000000001 += 1

    return p0_05, p0_01, p0_001, p0_0001, p0_00000000001


for width in numpy.arange(10) / 10:
    print(count_non_normal(f'h5data/width/width_{width:.1f}.h5', 2))
