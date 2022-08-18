import sys
import os
from multiprocessing import Pool


# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Local')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Local')

from Local.analysis_class import *

models = ["B"]
N_s = [2]
parallel = False
chisq_only = True
run_params = []

for N in N_s:
    for model in models:
        run_params.append((N, model))


def run(run_params):
    N, model = run_params
    a = CompareModels(N, model)
    a.GRID_plot(chisq_only=chisq_only, compare=(1, 2), show=True)


if parallel:
    p = Pool(os.cpu_count())
    results = p.map(run, run_params)
    p.close()

else:
    for run_param in run_params:
        run(run_param)
