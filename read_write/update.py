import h5py
import os
import re
import pdb
import numpy


N_s = [3]
g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
filename = f"../h5data/MCMC_data_full.h5"


def GRID_convention_m(m):
    return f"m2{m}".rstrip('0')

def MCMC_convention_m(m):
    return f"msq={-m:.8f}"

def GRID_convention_g(g):
    return f"g{g}".rstrip('0').rstrip('.')

def MCMC_convention_g(g):
    return f"g={g:.2f}"

def update(filename, N_s=None, g_s=None, L_s=None, m_s=None, OR=10, base_dir=f"/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor"):
    if os.path.isfile(filename):
        old_data = h5py.File(filename, "a")
    
    else:
        old_data = h5py.File(filename, "w")

    available_data = {}

    if g_s is None:
        files = os.popen(f'ls {base_dir}')
        for name in files:
            if len(re.findall(r'g\d+.\d+', name)) != 0:
                value = float(re.findall(r'\d+.\d+', name)[0])

                if g not in available_data.keys():
                    available_data[g] = {}

    else:
        for g in g_s:
            if g not in available_data.keys():
                available_data[g] = {}

    # Figure out which configurations we need to extract
    for g in available_data.keys():
        sub_dict = available_data[g]
        sub_dir = f"{base_dir}/{GRID_convention_g(g)}"
        files = os.popen(f'ls {sub_dir}')

        if N_s is None:
            for name in files:
                if len(re.findall(r'\d+', name)) != 0:
                    value = int(re.findall(r'\d+', name)[0])

                    if f"N{value}" not in sub_dict:
                        sub_dict[f"N{value}"] = {}

        else:
            for N in N_s:
                for name in files:
                    if len(re.findall(rf'{N}', name)) != 0:
                        sub_dict[f"su{N}"] = {}

        for N in sub_dict.keys():
            sub_dict2 = sub_dict[N]
            sub_dir2 = sub_dir + "/" + N

            if L_s is None:
                try:
                    files = os.popen(f'ls {sub_dir2}')

                except Exception:
                    files = []

                for name in files:
                    if len(re.findall(r'\d+', name)) != 0:
                        value = int(re.findall(r'\d+', name)[0])

                        if f"L{value}" not in sub_dict2:
                            sub_dict2[f"L{value}"] = {}

            else:
                for L in L_s:
                    if f"L{L}" not in sub_dict2:
                        sub_dict2[f"L{L}"] = {}

            for L in sub_dict2.keys():
                sub_dict3 = sub_dict2[L]
                sub_dir3 = sub_dir2 + "/" + L

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
                            value = float(re.findall(r'\d+\.\d+', name)[0])

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
                    file_root = f"{base_dir}/{GRID_convention_g(g)}/{N}/{L}/{GRID_convention_m(m)}/mag/cosmhol-hbor-{N}_{L}_{GRID_convention_g(g)}_{GRID_convention_m(m)}_or{OR}"

                    #pdb.set_trace()
                    if N not in old_data.keys():
                        old_data.create_group(N)

                    #pdb.set_trace()
                    if MCMC_convention_g(g) not in list(old_data[N].keys()):
                        old_data[N].create_group(MCMC_convention_g(g))

                    #pdb.set_trace()
                    if L not in old_data[N][MCMC_convention_g(g)].keys():
                        old_data[N][MCMC_convention_g(g)].create_group(L)

                    #pdb.set_trace()
                    if MCMC_convention_m(m) not in list(old_data[N][MCMC_convention_g(g)][L].keys()):
                        old_data[N][MCMC_convention_g(g)][L].create_group(MCMC_convention_m(m))

                        print(f"found new data for {N}, {g}, {L}, {m}")
                        #pdb.set_trace()

                        try:
                            old_data[N][MCMC_convention_g(g)][L][MCMC_convention_m(m)]['phi2'] = numpy.loadtxt(file_root + "_phi2.0.dat")
                            old_data[N][MCMC_convention_g(g)][L][MCMC_convention_m(m)]['m2'] = numpy.loadtxt(file_root + "_m2.0.dat")
                            old_data[N][MCMC_convention_g(g)][L][MCMC_convention_m(m)]['m4'] = numpy.loadtxt(file_root + "_m4.0.dat")

                        except Exception:
                            print("FILES NOT FOUND!")
                            del old_data[N][MCMC_convention_g(g)][L][MCMC_convention_m(m)]
                    
                    else:
                        print(f"{N}, {MCMC_convention_g(g)}, {L}, {MCMC_convention_m(m)} data already present")

    return old_data


update(filename, N_s=N_s, g_s=g_s)
