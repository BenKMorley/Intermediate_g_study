import pdb
import h5py
import os
import re
import numpy
import sys

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.MISC import *


def update(filename, N_s=None, g_s=None, L_s=None, m_s=None, OR=10,
           base_dir=f"/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor", size=100001):
    if os.path.isfile(filename):
        old_data = h5py.File(filename, "a")

    else:
        old_data = h5py.File(filename, "w")

    available_data = {}

    if g_s is None:
        files = os.popen(f'ls {base_dir}')
        for name in files:
            if len(re.findall(r'g\d+.\d+', name)) != 0:
                g = float(re.findall(r'\d+.\d+', name)[0])

                if g not in available_data.keys():
                    available_data[g] = {}

    else:
        for g in g_s:
            if g not in available_data.keys():
                available_data[g] = {}

    # Figure out which configurations we need to extract
    for g in available_data.keys():
        sub_dict = available_data[g]
        sub_dir = f"{base_dir}/{GRID_conv_g(g)}"
        files = os.popen(f'ls {sub_dir}')

        if N_s is None:
            for name in files:
                if len(re.findall(r'\d+', name)) != 0:
                    value = int(re.findall(r'\d+', name)[0])

                    if value not in sub_dict:
                        sub_dict[value] = {}

        else:
            for N in N_s:
                for name in files:
                    if len(re.findall(rf'{N}', name)) != 0:
                        sub_dict[N] = {}

        for N in sub_dict.keys():
            sub_dict2 = sub_dict[N]
            sub_dir2 = sub_dir + "/" + GRID_conv_N(N)

            if L_s is None:
                try:
                    files = os.popen(f'ls {sub_dir2}')

                except Exception:
                    files = []

                for name in files:
                    if len(re.findall(r'\d+', name)) != 0:
                        value = int(re.findall(r'\d+', name)[0])

                        if value not in sub_dict2:
                            sub_dict2[value] = {}

            else:
                for L in L_s:
                    if L not in sub_dict2:
                        sub_dict2[L] = {}

            for L in sub_dict2.keys():
                sub_dict3 = sub_dict2[L]
                sub_dir3 = sub_dir2 + "/" + GRID_conv_L(L)

                if m_s is None:
                    try:
                        files = os.popen(f'ls {sub_dir3}')

                    except Exception:
                        files = []

                    for name in files:
                        if len(re.findall(r'm2-\d+\.\d+', name)) != 0:
                            value = float(re.findall(r'-\d+\.\d+', name)[0])

                            sub_dict3[value] = {}

                        elif len(re.findall(r'm2\d+\.\d+', name)) != 0:
                            value = float(re.findall(r'\d+\.\d+', name.lstrip('m2'))[0])

                            sub_dict3[value] = {}

                else:
                    for m in m_s:
                        sub_dict3[m] = {}

    # Now we know which configurations we have available lets read them in
    for g in available_data.keys():
        dict1 = available_data[g]

        for N in dict1.keys():
            dict2 = dict1[N]

            for L in dict2.keys():
                dict3 = dict2[L]

                keys = list(dict3.keys())

                for m in keys:
                    dict4 = dict3[m]
                    file_root = f"{base_dir}/{GRID_conv_g(g)}/{GRID_conv_N(N)}/" +\
                        f"{GRID_conv_L(L)}/{GRID_conv_m(m)}/mag/cosmhol-hbor-{GRID_conv_N(N)}" +\
                        f"_{GRID_conv_L(L)}_{GRID_conv_g(g)}_{GRID_conv_m(m)}_or{OR}"

                    if MCMC_conv_N(N) not in old_data.keys():
                        old_data.create_group(MCMC_conv_N(N))

                    if MCMC_conv_g(g) not in list(old_data[MCMC_conv_N(N)].keys()):
                        old_data[MCMC_conv_N(N)].create_group(MCMC_conv_g(g))

                    if MCMC_conv_L(L) not in list(old_data[MCMC_conv_N(N)][MCMC_conv_g(g)].keys()):
                        old_data[MCMC_conv_N(N)][MCMC_conv_g(g)].create_group(MCMC_conv_L(L))

                    too_small = False
                    data_not_found = False

                    NgL = old_data[MCMC_conv_N(N)][MCMC_conv_g(g)][MCMC_conv_L(L)]

                    if MCMC_conv_m(m) in list(NgL.keys()):
                        print(f"{MCMC_conv_N(N)}, {MCMC_conv_g(g)}, {MCMC_conv_L(L)}," +
                              f"{MCMC_conv_m(m)} data already present")

                        # Make sure the data contains the desired number of configs
                        data_size = NgL[MCMC_conv_m(m)].shape[1]

                        if data_size < size:
                            print(f"Exisiting data present too small: Expected {size} got" +
                                  f"{data_size}")
                            print(f"Deleting entry")

                            del NgL[MCMC_conv_m(m)]

                            too_small = True

                    else:
                        data_not_found = True

                    if data_not_found | too_small:
                        print(f"found new data for {N}, {g}, {L}, {m}")

                        ## Find all .dat files for a given run
                        start_configs = []
                        directory, file_start = os.path.split(file_root)

                        Files_not_found = False
                        try:
                            files = os.listdir(directory)
                            files = list(filter(lambda x: x.endswith('.dat') and
                                                f'{file_start}_phi2.' in x, files))

                        except Exception:
                            Files_not_found = True

                        if (not Files_not_found) and (len(files) == 0):
                            Files_not_found = True

                        if Files_not_found:
                            print("FILES NOT FOUND!")

                        else:
                            for f in files:
                                start_configs.append(int(re.findall(r'\d+.dat', f)[0][:-4]))

                            prev_length = -1
                            prev_start = -1

                            # Subtract 1 for each section due to overlapp,
                            # however for first contribution there is no
                            # overlapp so start at 1
                            lengths = []
                            data_pieces = []

                            start_configs = numpy.sort(start_configs)

                            inconsistant_data = False
                            for start in start_configs:
                                new_data_piece = numpy.loadtxt(file_root + f"_phi2.{start}.dat")
                                current_length = new_data_piece.shape[0]
                                data_pieces.append(new_data_piece)
                                lengths.append(current_length)

                                if prev_length != -1:
                                    delta_start = prev_start + prev_length - start
                                    # if len(new_data_piece[:prev_length - start]) <= 0:
                                    #     print(f"No overlapp between data sets")
                                    #     inconsistant_data = True

                                    if delta_start > 0:
                                        try:
                                            diff = numpy.sum(numpy.abs(new_data_piece[:delta_start]
                                                                - old_data_piece[-delta_start:]))

                                        except Exception:
                                            pdb.set_trace()

                                        # Check that the boundaries match (boundaries can be >
                                        # 1 config)
                                        if diff != 0:
                                            print(f"Data not consistant! (config {start})")
                                            pdb.set_trace()
                                            inconsistant_data = True

                                prev_start = start
                                prev_length = current_length
                                old_data_piece = new_data_piece

                            total_length = start + current_length

                            if total_length < size:
                                print(f"Not enough data to read in: Expected {size}, got" +
                                      f"{total_length}")

                            # If the data is consistant and of the expected size then read it in
                            elif not inconsistant_data:
                                phi2 = numpy.zeros(total_length)
                                m2 = numpy.zeros(total_length)
                                m4 = numpy.zeros(total_length)

                                for i, start in enumerate(start_configs):
                                    phi2[start: start + lengths[i]] = numpy.loadtxt(file_root +
                                                                              f"_phi2.{start}.dat")
                                    m2[start: start + lengths[i]] = numpy.loadtxt(file_root +
                                                                              f"_m2.{start}.dat")
                                    m4[start: start + lengths[i]] = numpy.loadtxt(file_root +
                                                                              f"_m4.{start}.dat")

                                NgL = old_data[MCMC_conv_N(N)][MCMC_conv_g(g)][MCMC_conv_L(L)]

                                NgL.create_dataset(MCMC_conv_m(m), (3, len(phi2)), dtype='<f8')

                                NgL[MCMC_conv_m(m)][0] = m2
                                NgL[MCMC_conv_m(m)][1] = m4
                                NgL[MCMC_conv_m(m)][2] = phi2

    return old_data
