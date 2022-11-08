from ast import arg
from ensurepip import bootstrap
import sys
import os
from multiprocessing import Pool

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

N_s = [2, 3, 4, 5]
model = "A"  # The Bayesian Evidence integrator struggles if correction-to-scaling are added
model_types = [1, 2]
run_params = []
parallel = True
graph_lims = (4, 50)
bayes_points = 400


for N in N_s:
    for model_type in model_types:
        run_params.append((N, model, model_type))


def run(run_params):
    N, model, model_type = run_params
    a = analysis(N, model, model_type=model_type)
    a.fit_all_gLmin_all_Bbar(run_bootstrap=False)


if parallel:
    p = Pool(os.cpu_count())
    p.map(run, run_params)

else:
    for params in run_params:
        run(params)


def plot_Bayes_freq(gL_mins, ps1, ps2, E, graph_lims):
    color_blind_palette = [(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115),
                        (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]

    color_blind_palette = [(a / 255, b / 255, c / 255) for a, b, c in color_blind_palette]

    # Do N = 2 First
    min_value = numpy.min(E)
    max_value = numpy.max(E)

    fig, ax = plt.subplots()

    ax.scatter(gL_mins, E, label=r"$\mathcal{E}$", marker='o', color='none', edgecolor='k')

    # Show the Jeffrey's scale
    plt.fill_between([min(gL_mins) / 2, max(gL_mins) * 2], [-1, -1], [1, 1], color=color_blind_palette[7], alpha=0.5, label='Insignificant')
    plt.fill_between([min(gL_mins) / 2, max(gL_mins) * 2], [1, 1], [2, 2], color=color_blind_palette[4], alpha=0.2, label='Strong')
    plt.fill_between([min(gL_mins) / 2, max(gL_mins) * 2], [2, 2], [max_value * 1.1, max_value * 1.1], color=color_blind_palette[2], alpha=0.2, label='Decisive')
    plt.fill_between([min(gL_mins) / 2, max(gL_mins) * 2], [-2, -2], [-1, -1], color=color_blind_palette[4], alpha=0.2)
    plt.fill_between([min(gL_mins) / 2, max(gL_mins) * 2], [-10, -10], [-2, -2], color=color_blind_palette[2], alpha=0.2)

    ax = plt.gca()
    ax.legend(loc=(0.02, 0.4), framealpha=1)
    ax.set_ylabel(r"$\mathcal{E}$")
    ax.set_ylim(min_value + 1, max_value + 1)
    ax.set_xlim(min(gL_mins) - 1, max(gL_mins) + 1)
    ax.tick_params(direction='in')
    ax.set_xlabel(r"$gL_{min}$")
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_xlim(*graph_lims)
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$p-value$')
    ax2.scatter(gL_mins, ps1, marker='s', label=r"$\Lambda_{IR} \propto g$", color='none',
                edgecolor=color_blind_palette[1])
    ax2.scatter(gL_mins, ps2, marker='^', label=r"$\Lambda_{IR} \propto \frac{1}{L}$",
                color='none', edgecolor=color_blind_palette[5])

    # Show our acceptance threshold (alpha)
    ax2.plot(numpy.array(gL_mins) * 2 - 10, [0.05, ] * len(gL_mins), color='grey')
    ax2.plot(numpy.array(gL_mins) * 2 - 10, [0.95, ] * len(gL_mins), color='grey')

    fig.tight_layout()
    ax2.tick_params(direction='in')
    ax2.legend(loc=(0.02, 0.25), framealpha=1)
    ax.set_ylim(0.5, max(E) * 1.1)
    ax.set_yscale('symlog', linthreshy=2.5)
    ax2.annotate(r"$p = 0.05$", xy=(25, 0.07), color='grey')
    ax2.annotate(r"$p = 0.95$", xy=(25, 0.91), color='grey')


for N in N_s:
    a1 = analysis_Bayes(N, model, model_type=1, bayes_points=bayes_points)
    a2 = analysis_Bayes(N, model, model_type=2, bayes_points=bayes_points)

    gmin, gmax = graph_lims
    keep = numpy.logical_and(a1.gL_mins > gmin - 0.01,
                             a1.gL_mins < gmax + 0.01)
    gL_mins = a1.gL_mins[keep]

    a1.fit_all_gLmin_all_Bbar(run_bootstrap=False)
    a2.fit_all_gLmin_all_Bbar(run_bootstrap=False)

    p_values1 = numpy.max(a1.p_values, axis=0)
    p_values2 = numpy.max(a2.p_values, axis=0)

    max_dof = max(a1.dofs)

    # We use the best fit of the log(g) for this plot. For more values see the grid plots
    arg_min = numpy.argmax(p_values1 > 0.05)
    B_arg_min = numpy.argmax(a1.p_values[:, arg_min])

    ps1 = a1.p_values[B_arg_min][keep]
    ps2 = a2.p_values[B_arg_min][keep]

    # Now generate the Bayesian Evidence
    E1s = []
    E2s = []
    Bbar = a1.Bbar_list[B_arg_min]

    for gL_min in gL_mins:
        # Divide by log(10) to put the results into log base 10.
        E1s.append(a1.run_pymultinest(Bbar, gL_min)[0] / numpy.log(10))
        E2s.append(a2.run_pymultinest(Bbar, gL_min)[0] / numpy.log(10))

    E = [E1s[i] - E2s[i] for i in range(len(E1s))]

    plot_Bayes_freq(gL_mins, ps1, ps2, E, graph_lims)

    plt.savefig(f'Thesis_results/Combo_Plot_N{N}.pdf')
    plt.title(f'N = {N}, Bbar = {Bbar}')
    plt.close()
