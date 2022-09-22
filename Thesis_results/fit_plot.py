from lib2to3 import refactor
import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

# Parameters
N = 2
model = "A"
model_type = 1
gLmin = 12.8
Bbar = 0.52

a = analysis(N, model, model_type=1)

a.fit_all_gLmin_all_Bbar()
a.fit(Bbar, gLmin, plot=True)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.savefig(f"Thesis_results/IR_fit_example.pdf")
