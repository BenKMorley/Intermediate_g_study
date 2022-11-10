import os


# Make necessary directories for the project
for directory in ['Local/data', 'Local/graphs', 'Thesis_results/Binder_plots']:
    if not os.path.isdir(directory):
        os.makedirs(directory)
