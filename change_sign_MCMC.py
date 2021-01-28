import h5py
import pdb
import numpy
import os
import re


def change_sign(MCMC_data_file, output_filename):
    with h5py.File(MCMC_data_file, "r") as f:
        with h5py.File(output_filename, "w") as f_new:
            for N_key in f.keys():
                N = int(re.findall(r'\d+', N_key)[0])

                if f'N={N}' not in f_new:
                    f_new.create_group(f'N={N}')

                for g_key in f[f'N={N}'].keys():
                    g = float(re.findall(r'\d+\.\d+', g_key)[0])

                    if f'g={g:.2f}' not in f_new[f'N={N}']:
                        f_new[f'N={N}'].create_group(f'g={g:.2f}')

                    for L_key in f[f'N={N}'][f'g={g:.2f}']:
                        L = int(re.findall(r'\d+', L_key)[0])

                        if f'L={L}' not in f_new[f'N={N}'][f'g={g:.2f}']:
                            f_new[f'N={N}'][f'g={g:.2f}'].create_group(f'L={L}')

                        for m_key in f[f'N={N}'][f'g={g:.2f}'][f'L={L}']:
                            m = float(re.findall(r'\d+\.\d+', m_key)[0])

                            f_new[f'N={N}'][f'g={g:.2f}'][f'L={L}'][f'm={-m:.8f}'] = numpy.array(f[f'N={N}'][f'g={g:.2f}'][f'L={L}'][f'msq={m:.8f}'])
                        

change_sign("h5data/MCMCdata.h5", "h5data/MCMCdata_flipped_sign.h5")
