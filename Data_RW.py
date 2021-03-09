import sqlite3
import h5py
import pdb
import numpy
import os
import re

from mass_array import get_masses


def g_string(g):
    return f"{g:.1f}".rstrip('0')


def get_SQL_data(N, L, g, OR):
    directory = f"/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor-dbtest/g{g:.1f}/su{N}/L{L}/"

    masses = []
    files = os.popen(f'ls {directory}')
    for name in files:
        if len(re.findall(r'-\d+\.\d+', name)) != 0:
            masses.append(float(re.findall(r'-\d+\.\d+', name)[0]))

    phi2 = {}
    m2 = {}
    m4 = {}
    num_entries = {}
    found = {}

    for m in masses:
        file_name = f"cosmhol-scalar-hbor-su{N}_L{L}_g" + f"{g:.1f}".rstrip('0') + f"_m2{float(m):.5f}".rstrip('0') + f"_or{OR}_database.0.db"

        conn = sqlite3.connect(f"{directory}/m2{m:.5f}/mag/{file_name}")

        cur = conn.execute('select * from Observables')

        names = list(map(lambda x: x[0], cur.description))
        
        found = True

        try:
            phi2[m] = cur.execute('select phi2 from Observables').fetchall()
            m2[m] = cur.execute('select m2 from Observables').fetchall()
            m4[m] = cur.execute('select m4 from Observables').fetchall()

            num_entries[m] = len(m2[m])

        except:
            print(f"Data File not found: N={N}, g={g}, L={L}, m={m}")
            Found = False

    return phi2, m2, m4, num_entries, masses, found


def get_raw_data_flexible(N_s=None, g_s=None, L_s=None, m_s=None, OR=10, base_dir=f"/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor", change_sign=False):
    ensemble_dict = {}

    if g_s is None:
        files = os.popen(f'ls {base_dir}')
        for name in files:
            if len(re.findall(r'g\d+.\d+', name)) != 0:
                value = float(re.findall(r'\d+.\d+', name)[0])
                ensemble_dict[f"g{value:.1f}"] = {}

    else:
        for g in g_s:
            ensemble_dict[f"g{g:.1f}"] = {}

    # Figure out which configurations we need to extract
    for g in ensemble_dict.keys():
        sub_dict = ensemble_dict[g]
        sub_dir = f"{base_dir}/{g}"

        if N_s is None:
            files = os.popen(f'ls {sub_dir}')
            for name in files:
                if len(re.findall(r'\d+', name)) != 0:
                    value = int(re.findall(r'\d+', name)[0])
                    sub_dict[f"N{value}"] = {}

        else:
            for N in N_s:
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
                        sub_dict2[f"L{value}"] = {}

            else:
                for L in L_s:
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
                            if change_sign == False:
                                value = -float(re.findall(r'\d+\.\d+', name)[0])
                                sub_dict3[f"m2{value:.5f}"] = {}

                            else:
                                value = float(re.findall(r'\d+\.\d+', name)[0])
                                sub_dict3[f"m2{value:.5f}"] = {}

                        elif len(re.findall(r'm2\d+\.\d+', name)) != 0:
                            if change_sign == False:
                                value = float(re.findall(r'\d+\.\d+', name)[0])
                                sub_dict3[f"m2{value:.5f}"] = {}

                            else:
                                value = -float(re.findall(r'\d+\.\d+', name)[0])
                                sub_dict3[f"m2{value:.5f}"] = {}

                else:
                    for m in m_s:
                        sub_dict3[f"m2{m:.5f}"] = {}

    # Extract the configurations
    for g in ensemble_dict.keys():
        dict1 = ensemble_dict[g]

        for N in dict1.keys():
            dict2 = dict1[N]

            for L in dict2.keys():
                dict3 = dict2[L]

                keys = list(dict3.keys())

                for m in keys:
                    dict4 = dict3[m]
                    file_root = f"{base_dir}/{g}/{N}/{L}/" + m.rstrip('0') + f"/mag/cosmhol-hbor-{N}_{L}_{g}_{m}".rstrip('0') + f"_or{OR}"

                    print("Loading in data from: " + file_root)

                    try:
                        dict4['phi2'] = numpy.loadtxt(file_root + "_phi2.0.dat")
                        dict4['m2'] = numpy.loadtxt(file_root + "_m2.0.dat")
                        dict4['m4'] = numpy.loadtxt(file_root + "_m4.0.dat")

                    except Exception:
                        print("FILES NOT FOUND!")
                        del dict3[m]

    return ensemble_dict


def get_raw_data(N, L, g, OR=10, sub_dir="cosmhol-hbor"):
    base_dir = f"/rds/project/dirac_vol4/rds-dirac-dp099/{sub_dir}/g{g:.1f}/su{N}/L{L}"

    masses = []
    files = os.popen(f'ls {base_dir}')
    for name in files:
        if len(re.findall(r'-\d+\.\d+', name)) != 0:
            masses.append(float(re.findall(r'-\d+\.\d+', name)[0]))

    phi2 = {}
    m2 = {}
    m4 = {}
    num_entries = {}
    found = {}

    for m in masses:
        file_prefix = f"cosmhol-hbor-su{N}_L{L}_g{g}_m2{m}_or{OR}"
        in_dir = f"{base_dir}/m2{m}/mag"

        found[m] = True

        try:
            phi2[m] = numpy.loadtxt(f"{in_dir}/{file_prefix}_phi2.0.dat")
            m2[m] = numpy.loadtxt(f"{in_dir}/{file_prefix}_m2.0.dat")
            m4[m] = numpy.loadtxt(f"{in_dir}/{file_prefix}_m4.0.dat")

            num_entries[m] = len(m2[m])

        except:
            print(f"Data File not found: N={N}, g={g}, L={L}, m={m}")
            found[m] = False

    return phi2, m2, m4, num_entries, masses, found


def get_raw_data_one_m(N, L, g, m, OR=10, sub_dir="cosmhol-hbor"):
    base_dir = f"/rds/project/dirac_vol4/rds-dirac-dp099/{sub_dir}/g{g:.1f}/su{N}/L{L}/" + f"m2{m:.5f}".rstrip('0')  + "/mag"

    file_prefix = f"cosmhol-hbor-su{N}_L{L}_g{g}_" + f"m2{m:.5f}".rstrip('0') + f"_or{OR}"
    in_dir = f"{base_dir}/m2{m}/mag"

    found = True

    try:
        phi2 = numpy.loadtxt(f"{in_dir}/{file_prefix}_phi2.0.dat")
        m2 = numpy.loadtxt(f"{in_dir}/{file_prefix}_m2.0.dat")
        m4 = numpy.loadtxt(f"{in_dir}/{file_prefix}_m4.0.dat")

        num_entries = len(m2)

    except:
        print(f"Data File not found: N={N}, g={g}, L={L}, m={m}")
        found = False

    return phi2, m2, m4, num_entries, masses, found


def write_data_to_MCMC(N, L, g, m, phi2, m2, m4, num_entries, rewrite_data=False, filename="MCMC_test.h5"):
    with h5py.File(filename, "a") as f:
        m2 = numpy.array(m2)
        m4 = numpy.array(m4)
        phi2 = numpy.array(phi2)

        assert len(m2) == num_entries
        assert len(m4) == num_entries
        assert len(phi2) == num_entries

        if m2.shape != (num_entries, ):
            assert m2.shape == (num_entries, 1)
            m2.reshape(num_entries)

        if m4.shape != (num_entries, ):
            assert m4.shape == (num_entries, 1)
            m4.reshape(num_entries)

        if phi2.shape != (num_entries, ):
            assert phi2.shape == (num_entries, 1)
            phi2.reshape(num_entries)

        if f"N={N}" not in f.keys():
            N_level = f.create_group(f"N={N}")
        else:
            N_level = f[f"N={N}"]

        if f"g={g:.2f}" not in N_level.keys():
            g_level = N_level.create_group(f"g={g:.2f}")
        else:
            g_level = N_level[f"g={g:.2f}"]

        if f"L={L}" not in g_level.keys():
            L_level = g_level.create_group(f"L={L}")
        else:
            L_level = g_level[f"L={L}"]


        # If this key is present then the data has already been written in
        if f"msq={float(m):.8f}" not in L_level.keys():
            print(f"About to write data for N = {N}, L = {L}, g = {g}, m = {m}")
            data = L_level.create_dataset(f"msq={float(m):.8f}", (3, num_entries), dtype='f')

            data[0] = m2
            data[1] = m4
            data[2] = phi2

        elif rewrite_data:
            print(f"About to write data for N = {N}, L = {L}, g = {g}, m = {m}")
            data = L_level.create_dataset(f"msq={float(m):.8f}", (3, num_entries), dtype='f')

            data[0] = m2
            data[1] = m4
            data[2] = phi2

        else:
            print("Data aleady in file - continuing without rewrite")
