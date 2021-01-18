import sqlite3
import h5py
import pdb
from mass_array import get_masses


def g_string(g):
    if abs(g - 1) < 10 ** -10:
        return "1."
    else:
        return f"{g:.1f}"


def get_SQL_data(N, L, g, m, OR):
    directory = f"/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor-dbtest/g{g:.2f}/su{N}/L{L}/m2{m}/"
    file_name = f"cosmhal-scalar-hbor-su{N}-L{L}_g{g_string(g)}_m2{m}_or{OR}_database.0.db"

    pdb.set_trace()
    conn = sqlite3.connect(f"{directory}{file_name}")

    cur = conn.cursor()

    names = list(map(lambda x: x[0], cur.description))

    phi2 = cur.execute('select phi2 from Observables').fetchall()
    m2 = cur.execute('select m2 from Observables').fetchall()
    m4 = cur.execute('select m4 from Observables').fetchall()

    num_entries = len(m2)

    return phi2, m2, m4, num_entries


def write_data_to_MCMC(N, L, g, m, phi2, m2, m4, num_entries):
    f = h5py.File("MCMC_test.h5", "w")

    assert len(m2) == num_entries
    assert len(m4) == num_entries
    assert len(phi2) == num_entries

    if f"N={N}" not in f.keys():
        N_level = f.create_group(f"N={N}")
    else:
        N_level = f[f"N={N}"]

    if f"g={g:.2f}" not in f.keys():
        g_level = N_level.create_group(f"g={g:.2f}")
    else:
        g_level = N_level[f"g={g:.2f}"]

    if f"L={L}" not in f.keys():
        L_level = g_level.create_group(f"L={L}")
    else:
        L_level = g_level[f"L={L}"]

    # If this key is present then the data has already been written in
    if f"msq={m:.8f}" not in f.keys():
        m_level = L_level.create_group(f"msq={m:.8f}")
        data = m_level.create_dataset((3, num_entries))

        data[0] = m2
        data[1] = m4
        data[2] = phi2

    elif rewrite_data:
        m_level = L_level[f"msq={m:.8f}"]
        data = m_level.create_dataset((3, num_entries))

        data[0] = m2
        data[1] = m4
        data[2] = phi2

    else:
        print("Data aleady in file - continuing without rewrite")


N = 2
L = 16
g = 1
OR = 10
num_m = 20

masses = get_masses(N, g, L, num_m)

for m in masses:
    phi2, m2, m4, num_entries = get_SQL_data(N, L, g, m, OR)

    write_data_to_MCMC(N, L, g, m, phi2, m2, m4, num_entries)
