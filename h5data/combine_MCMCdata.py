import h5py
import sys

file1 = sys.argv[0]
file2 = sys.argv[1]

with h5py.File(file1) as f1:
    with h5py.File(file2) as f2:
        for N_key in f2.keys():
            N = int(re.findall(r'\d+', N_key)[0])

            if f"N={N}" not in f1.keys():
                N_level1 = f1.create_group(f"N={N}")
            else:
                N_level1 = f1[f"N={N}"]

            N_level2 = f2[f"N={N}"]

            for g_key in N_level2.keys():
                g = float(re.findall(r'\d+\.\d+', g_key)[0])

                if f"g={g:.2f}" not in N_level2.keys():
                    g_level1 = N_level1.create_group(f"g={g:.2f}")
                else:
                    g_level1 = N_level1[f"g={g:.2f}"]

                g_level2 = N_level2[f"g={g:.2f}"]

                for L_key in g_level2:
                    L = int(re.findall(r'\d+', L_key)[0])

                    if f"L={L}" not in g_level1.keys():
                        L_level1 = g_level1.create_group(f"L={L}")
                    else:
                        L_level1 = g_level1[f"L={L}"]

                    L_level2 = g_level2[f"L={L}"]

                    for m_key in L_level2:
                        m = float(re.findall(r'\d+\.\d+', m_key)[0])

                        if f'msq={m:.8f}' not in L_level1.keys():
                            L_level1[f'msq={m:.8f}'] = L_level2[f'msq={m:.8f}']
