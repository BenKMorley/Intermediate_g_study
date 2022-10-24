import sys
import os
from multiprocessing import Pool
import numpy

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *
from Local.Plot_Binder import plot_Binder

const_Bbar = True
show_plot = False
plot_crossings = True
plot_histograms = True
parallel = False
samples = 100
g = 0.1
L = 96
N = 2
mlims = [-0.03133, -0.03117]


def width_subplot(width, g, L, ax, N, legend=False):
    L_s = [L]
    g_s = [g]
    color_dict = {'bs_bins_(0, 1, 2)': 'g',
                  'bs_bins_(0, 1, 2, 3)': 'r',
                  'bs_bins_(1, 2, 3)': 'b'}

    plot_Binder(N, g_s, L_s, crossings_file=f'width/width_{width:.1f}.h5', width=width, ax=ax,
                plot_histograms=plot_histograms, plot_crossings=plot_crossings, min_gL=0.7,
                no_reweight_samples=samples, legend=legend, use_128=True, plot_lims=mlims,
                color_dict=color_dict)

    ax.set_title(f'width = {width:.1f}')
    Bbar_s = param_dict[N]["Bbar_list"]


widths = [0, 0.1, 0.2, 0.3]
h = 2
w = 2

fig, axes = plt.subplots(h, w)
a = numpy.array(axes)
a = a.reshape(h * w)

for i in range(h * w):
    width_subplot(widths[i], g, L, a[i], N, i == 0)

    if i // w != h - 1:
        a[i].axes.xaxis.set_visible(False)

    if i % w != 0:
        a[i].axes.yaxis.set_visible(False)

    a[i].set_xlim(*mlims)
    a[i].set_ylim(0.54, 0.58)

fig.set_size_inches(18, 12)
plt.savefig(f'Thesis_results/reweighting_smoothing.pdf')
plt.show()
