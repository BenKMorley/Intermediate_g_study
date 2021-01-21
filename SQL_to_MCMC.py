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
    files = os.popen('ls')
    for name in files:
        if re.findall(r'-\d+\.\d+', name)[0] is not None:
            masses.append(float(re.findall(r'-\d+\.\d+', string)[0]))

    phi2 = {}
    m2 = {}
    m4 = {}
    num_entries = {}

    for m in masses:
        file_name = f"cosmhol-scalar-hbor-su{N}_L{L}_g" + f"{g:.1f}".rstrip('0') + f"_m2{float(m):.5f}".rstrip('0') + f"_or{OR}_database.0.db"

        conn = sqlite3.connect(f"{directory}{file_name}")

        cur = conn.execute('select * from Observables')

        names = list(map(lambda x: x[0], cur.description))

        phi2[m] = cur.execute('select phi2 from Observables').fetchall()
        m2[m] = cur.execute('select m2 from Observables').fetchall()
        m4[m] = cur.execute('select m4 from Observables').fetchall()

        num_entries[m] = len(m2[m])

    return phi2, m2, m4, num_entries, masses


def write_data_to_MCMC(N, L, g, m, phi2, m2, m4, num_entries, rewrite_data=False):
    with h5py.File("MCMC_test.h5", "a") as f:
        assert len(m2) == num_entries
        assert len(m4) == num_entries
        assert len(phi2) == num_entries

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

            data[0] = numpy.array(m2)[:, 0]
            data[1] = numpy.array(m4)[:, 0]
            data[2] = numpy.array(phi2)[:, 0]

        elif rewrite_data:
            print(f"About to write data for N = {N}, L = {L}, g = {g}, m = {m}")
            data = L_level.create_dataset(f"msq={float(m):.8f}", (3, num_entries), dtype='f')

            data[0] = numpy.array(m2)[:, 0]
            data[1] = numpy.array(m4)[:, 0]
            data[2] = numpy.array(phi2)[:, 0]

        else:
            print("Data aleady in file - continuing without rewrite")


N = 2
g = 1
OR = 10
num_m = 20
L = 16

phi2, m2, m4, num_entries, masses = get_SQL_data(N, L, g, OR)

for m in masses:
    phi2_ = phi2[m]
    m2_ = m2[m]
    m4_ = m4[m]
    num_entries_ = num_entries[m]
    print(f"Retrieving data for N = {N}, L = {L}, g = {g}, m = {m}")

    write_data_to_MCMC(N, L, g, m, phi2, m2, m4, num_entries)
