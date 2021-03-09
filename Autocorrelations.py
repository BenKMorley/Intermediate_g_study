import h5py
import re
import numpy
import pdb
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('tkagg')


def get_autocorrelation_time(N, g, L):
    with h5py.File("MCMC_test.h5", "r") as f:
        data = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

        masses = []
        Auto = []

        for string in data.keys():
            m = float(re.findall(r'-\d+\.\d+', string)[0])
            # pdb.set_trace()

            mass_data = data[f'msq={m:.8f}']

            M2 = mass_data[0]
            M4 = mass_data[1]

            aut = []
            aut.append(numpy.mean(M2 ** 2) - numpy.mean(M2))
            i = 1

            while (numpy.mean(M2[i:] * M2[:-i]) - numpy.mean(M2) ** 2) > 0:
                aut.append(numpy.mean(M2[i:] * M2[:-i]) - numpy.mean(M2) ** 2)

                i += 1

            log_aut = numpy.log(aut)

            try:
                x = numpy.polyfit(numpy.arange(0, len(aut)), log_aut, 1)[0]
                masses.append(m)
                Auto.append(- 1 / x)

            except:
                print("Can't fit data - not enough valid points")

    return masses, Auto

g = 1
L = 32
N = 2

masses, Auto = get_autocorrelation_time(N, g, L)
