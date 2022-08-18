import sys
import os


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.run_analysis_script2 import *

# Parameters
N = 2
model = "B"
model_type = 1
NBbar = 2
Bbar = 0.52

a = analysis(N, model, NBbar=NBbar, model_type=1)

a.fit_all_gLmin_all_Bbar()
a.BMM_plot_gLmin(0.52, show=True, params=['nu'])
