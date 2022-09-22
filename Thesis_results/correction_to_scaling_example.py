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
N = 4
model = "B"
model_type = 1
gLmin = 24

a = analysis(N, model, model_type=1)

a.fit_all_gLmin_all_Bbar()
a.BMM_specific_gLmin(gLmin, params=['c1'], refactor=True, plot=True)
plt.savefig(f"Thesis_results/Correction_to_scaling_coefficient.pdf")
