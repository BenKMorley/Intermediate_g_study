import sys
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

model = "B"
N = 2
chisq_only = True
font_size = 15

a = CompareModels(N, model)

fig, ax = plt.subplots()

ax = a.GRID_plot(chisq_only=chisq_only, compare=(1, 2), show=True, ax_in=ax)

ax.set_xlabel(r'$gL_\textrm{min}$', fontsize=font_size)
ax.set_ylabel(r'$B$', fontsize=font_size, rotation=0)

gL_mins = a.model1.gL_mins
Bbars = a.model1.Bbar_pieces

ax.set_xticks(numpy.linspace(min(gL_mins), max(gL_mins), len(gL_mins) + 1))
ax.set_yticks(numpy.linspace(min(Bbars), max(Bbars), len(Bbars) + 1))

minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)

x_labels = ["", ] + [f"{str(i)}" for i in list(gL_mins)]
y_labels = ["", ] + [str(i) + '\n' * 6 for i in list(Bbars)][::-1]

ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# ax.tick_params(axis='both', which='major', labelsize=font_size)
# ax.tick_params(axis='both', which='major', labelsize=font_size)

plt.title(rf'$N = {N}$, model={model}', fontsize=font_size * 1.5)
fig.set_size_inches(12, 8)

plt.savefig(f'Thesis_results/compare_plot_N{N}_model{model}.pdf')
plt.show()
