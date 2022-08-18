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

a = analysis(N, model, NBbar=NBbar, model_type=model_type)
a.fit_all_gLmin_all_Bbar()
a.BMM_plot_overall(show=True, params=['nu'])
