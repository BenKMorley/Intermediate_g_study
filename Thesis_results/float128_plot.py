import sys
import os

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *
from Local.Plot_Binder import plot_Binder

plt.rcParams.update({"text.usetex": True})

N = 4
model = "A"
const_Bbar = True
show_plot = False
plot_crossings = False
plot_histograms = False
parallel = True
legend = False
scale_with_fit = True
w = 0
crossings_file = f"width/width_{w:.1f}.h5"
make_data = False
scale_with_fit = False
gL_min = 3.9
g_s = [0.5]
L_s = [128]
c1 = 'g'
c2 = 'r'

fig, ax = plt.subplots()

# Plot with float128
ax = plot_Binder(N, g_s=g_s, L_s=L_s, min_gL=gL_min, reweight=True, ax=ax, model=model,
    plot_crossings=plot_crossings, plot_histograms=plot_histograms, legend=legend, width=w,
    crossings_file=crossings_file, scale_with_fit=scale_with_fit, color=c1)


# Plot with float64
ax = plot_Binder(N, g_s=g_s, L_s=L_s, min_gL=gL_min, reweight=True, ax=ax, model=model,
    plot_crossings=plot_crossings, plot_histograms=plot_histograms, legend=legend, width=w,
    crossings_file=crossings_file, scale_with_fit=scale_with_fit, use_128=False, color=c2)

ax.set_ylim(0.4, 0.6)

font_size = 10

p = ax.fill(numpy.NaN, numpy.NaN, 'g', alpha=0.2)
p2 = ax.fill(numpy.NaN, numpy.NaN, 'r', alpha=0.2)
ax.legend(handles=[p[0], p2[0]], labels=['float128', 'float64'], loc='lower left',
          fontsize=font_size)

ax.set_title(rf'$L = {L_s[0]}$, $g = {g_s[0]}$, $N = {N}$', fontsize=font_size * 1.5)
ax.set_xlabel(r'$m^2$', fontsize=font_size)
ax.set_ylabel(r'$B$', fontsize=font_size, rotation=0)

ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)

ax.set_xlim(-0.2229, -0.2217)
fig.set_size_inches(6, 4)
plt.savefig('Thesis_results/float128_plot.pdf')
plt.show()
