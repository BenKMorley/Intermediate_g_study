import sys
import os
from math import floor
from multiprocessing import Pool


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *
from Local.Plot_Binder import plot_Binder


models = ["A", "B", "C"]
Ns = [2, 3, 4, 5]
model_type = 1
const_Bbar = True
show_plot = False
plot_crossings = True
plot_histograms = False
parallel = True
scale_with_fit = True
w = 0.5
crossings_file = f"width/width_{w:.1f}.h5"
make_data = False
gL_min = 7.9


# Code from https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
def set_size(w, h, ax):
    """ w, h: width, height in inches """
    L = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - L)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


if make_data:
    run_params = []

    for N in Ns:
        for model in models:
            a = analysis(N, model, model_type=model_type)

            for g in set(a.g_s):
                for L in set(a.L_s):
                    run_params.append((g, L, N, model))

    def run(run_params):
        g, L, N, model = run_params
        a = analysis(N, model, model_type=model_type)
        a.fit_all_gLmin_all_Bbar()
        a.BMM_overall(perform_checks=True)

        m_c = a.mass_mean
        nu = a.mean[-1]

        fig, ax = plt.subplots()

        ax = plot_Binder(N, [g], [L], min_gL=0.7, max_gL=76.9, reweight=True, ax=ax,
                plot_lims=plot_lims, plot_crossings=plot_crossings, m_crit=m_c,
                nu=nu, plot_histograms=plot_histograms, legend=False, width=w,
                crossings_file=crossings_file, plot_Bounds=plot_Bounds, scale_with_fit=True)

        plt.close('all')

    if parallel:
        p = Pool(os.cpu_count())
        p.map(run, run_params)

    else:
        for params in run_params:
            run(params)

for N in Ns:
    for i, model in enumerate(models):
        fig, ax = plt.subplots()

        ax = plot_Binder(N, min_gL=gL_min, reweight=True, ax=ax, model=model,
            plot_crossings=plot_crossings, plot_histograms=plot_histograms, legend=False, width=w,
            crossings_file=crossings_file, scale_with_fit=scale_with_fit)

        x_axis_string = r'$((m_c^2 - m^2) / g^2) x^{1 / \nu}'
        if model in ["B", "C"]:
            x_axis_string += r'+ c(\bar{B})x^{-\omega}'

        if model == "C":
            x_axis_string += r'+ e(\bar{B})x^{-\epsilon}'

        x_axis_string += r'$'

        ax.set_xlabel(x_axis_string)

        if N == 2:
            ax.set_xlim(-3, 3)

        if N == 3:
            ax.set_xlim(-3, 3)
            ax.set_ylim(0.38, 0.52)

        if N == 4:
            ax.set_xlim(-3, 3)
            ax.set_ylim(0.35, 0.67)

        if N == 5:
            ax.set_xlim(-3, 3)
            ax.set_xlim()

        fig.set_size_inches(8, 6)
        plt.savefig(f"Thesis_results/Binder_plots/N{N}_model{model}_gLmin{gL_min}.pdf")
