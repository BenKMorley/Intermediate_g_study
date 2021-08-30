# Import from the Core directory
import enum
import sys
import os
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')


from numpy.core.einsumfunc import _parse_possible_contraction
from Core.model_definitions import *
from scipy.optimize import minimize, least_squares
from scipy.linalg import logm
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


## Parameters
input_h5_file = "h5data/Bindercrossings.h5"
N_s_in = [2]
N = 2
g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
Bbar_s_in = [0.52, 0.53]
L_s_in = [8, 16, 32, 48, 64, 96, 128]
GL_max = numpy.inf
print_info = True
model = model1_1a
param_names = ["alpha", "f0", "f1", "beta", "nu"]
x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
no_samples = 100

GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
                       16, 19.2, 24, 25.6, 28.8, 32])

means = []
stds = []
probs = []

# Try loading in all the data so we know how many points are being cut out
samples_full, g_s_full, L_s_full, Bbar_s_full, m_s_full = load_h5_data(
    input_h5_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, 0, GL_max)

for j, GL_min in enumerate(GL_mins):
    samples, g_s, L_s, Bbar_s, m_s = load_h5_data(input_h5_file, N_s_in, g_s_in, L_s_in,
                                                  Bbar_s_in, GL_min, GL_max)

    cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

    res = least_squares(res_function, x0, args=(cov_inv, model),
                        method="lm")

    chisq = chisq_calc(res.x, cov_inv, model, res_function)
    n_params = len(res.x)
    dof = g_s.shape[0] - n_params
    p = chisq_pvalue(dof, chisq)
    param_central = res.x
    N_cut = len(g_s_full) - len(g_s)
    AIC = chisq + 2 * n_params + 2 * N_cut
    prob_model = numpy.exp(-0.5 * AIC)

    means.append(param_central)
    probs.append(prob_model)

    if print_info:
        print("##############################################################")
        print(f"Config: N = {N}, Bbar_s = [{Bbar_s_in[0]}, {Bbar_s_in[1]}],"
              f" gL_min = {GL_min}, gL_max = {GL_max}")
        print(f"chisq = {chisq}")
        print(f"chisq/dof = {chisq / dof}")
        print(f"pvalue = {p}")
        print(f"dof = {dof}")
        print(f"N_cut = {N_cut}")
        print(f"AIC = {AIC}")
        print(f"prob_model = {prob_model}")

    param_estimates = numpy.zeros((no_samples, n_params))

    for i in tqdm(range(no_samples)):
        m_s = samples[:, i]

        res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

        res = least_squares(res_function, x0, args=(cov_inv, model),
                            method="lm")

        param_estimates[i] = numpy.array(res.x)

    stds.append(numpy.std(param_estimates, axis=0))


means = numpy.array(means)
stds = numpy.array(stds)
probs = numpy.array(probs)


# Calculate the overall parameter estimates
mean = numpy.average(means, axis=0, weights=probs)
std = numpy.average(stds, axis=0, weights=probs) +\
    numpy.average(means ** 2, axis=0, weights=probs) - mean ** 2

## Plot the Parameter estimates
for i, param in enumerate(param_names):
    (_, caps, _) = plt.errorbar(GL_mins, means[:, i], stds[:, i], ls='', color='blue', capsize=5, marker='_')
    for cap in caps:
        cap.set_color('blue')
        cap.set_markeredgewidth(1)

    plt.fill_between([min(GL_mins), max(GL_mins)], [mean[i] - std[i]], [mean[i] + std[i]], alpha=0.1, color='k')
    plt.plot([min(GL_mins), max(GL_mins)], [mean[i], mean[i]], color='k', label='BMM')
    plt.title(f"{param}")
    plt.xlabel(r"$gL_{min}$")
    plt.savefig(f"Local/graphs/BMM_{param}_N{N}_Bbars{Bbar_s_in}.pdf")
    plt.clf()
