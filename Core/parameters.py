###############################################################################
# Source file: ./parameters.py
#
# Copyright (C) 2020
#
# Author: Andreas Juettner juettner@soton.ac.uk
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# See the full license in the file "LICENSE" in the top level distribution
# directory
###############################################################################

# list of available volumes
from model_definitions import model1_1a, model1_2a, model2_1a, model2_2a
seperator = "##############################################################"


Ls = [8, 16, 32, 48, 64, 96, 128]
gs = [0.1, 0.2, 0.3, 0.5, 0.6, 1.0]
Ns = [2, 3, 4]

# interval on which solver should iterate, in case of emptly list solver will
# run between smallest and largest simulated mass. For each dictionary entry
# the lists corresponds to L/a=8,16,32,48,64,96,128
mlims = {}

# N=2 ####
mlims['su2_0.1'] = [
            [-0.044, -0.012],
            [-0.036, -0.024],
            [-0.0325, -0.029],
            [-0.0320, -0.02950],
            [-0.0318, -0.03025],
            [-0.0315, -0.0308],
            [-0.0315, -0.0309]]

mlims['su2_0.2'] = [
            [-0.080, -0.035],
            [-0.07, -0.051],
            [-0.065, -0.057],
            [-0.063, -0.0595],
            [-0.063, -0.06],
            [-0.06275, -0.0613],
            [-0.0625, -0.0615]]

mlims['su2_0.3'] = [
            [-0.113, -0.058],
            [-0.098, -0.081],
            [-0.096, -0.0880],
            [-0.094, -0.0895],
            [-0.0933, -0.0905],
            [-0.0935, -0.0915],
            [-0.0932, -0.0920]]

mlims['su2_0.5'] = [
            [-0.19, -0.10],
            [-0.162, -0.1375],
            [-0.158, -0.145],
            [-0.155, -0.149],
            [],
            [-0.1538, -0.1515],
            [-0.154, -0.152]]

mlims['su2_0.6'] = [
            [-.20, -.12],
            [-.2, -.16],
            [-0.186, -0.174],
            [-0.184, -0.1785],
            [-0.184, -0.180],
            [-0.184, -0.1813],
            [-0.18375, -0.18225]]


mlims['su2_0.7'] = [
            [],
            [],
            [-0.212, -0.209],
            [-0.213, -0.21125],
            [-0.21310, -0.2121]]

# N=3 ####
mlims['su3_0.1'] = [
    [],
    [-0.097, -0.055],
    [],
    [],
    [],
    [],
    []
]


mlims['su3_0.2'] = [
    [],
    [-0.097, -0.055],
    [],
    [],
    [],
    [],
    []
]

mlims['su3_0.3'] = [
    [],
    [-0.139, 0],
    [],
    [],
    [],
    [],
    []
]

mlims['su3_0.5'] = [
    [],
    [-0.139, 0],
    [],
    [],
    [],
    [],
    [-0.2075, -0.202]
]

mlims['su3_0.6'] = [
    [],
    [-0.139, 0],
    [],
    [],
    [],
    [],
    []
]


# N=4 ####
mlims['su4_0.1'] = [
            [-0.069, -0.025],
            [-0.053, -0.037],
            [-0.048, -0.041],
            [-0.047, -0.043],
            [-0.0465, -0.044],
            [-0.0458, -0.0448],
            [-0.0456297, -0.04495]]


mlims['su4_0.2'] = [
            [-0.123, -0.06],
            [-0.100, -0.075], [],
            [-0.0917, -0.0873],
            [-0.0912, -0.0880],
            [-0.09075, -0.0893],
            [-0.09025, -0.0897]]

mlims['su4_0.3'] = [
            [-0.174, -0.09],
            [-.145, -.122],
            [-.1375, -.1295],
            [-0.1365, -0.13],
            [-0.136, -0.132],
            [-0.1350, -0.1335],
            [-.135, -.1340]]

mlims['su4_0.5'] = [
            [-0.27, -0.162],
            [-0.24, -0.195],
            [-0.226, -0.2125],
            [-.224, -.218],
            [-0.2230, -0.220], [],
            [-0.22292, -0.22175]]

mlims['su4_0.6'] = [
            [-0.31, -0.20],
            [],
            [-0.270, -0.25],
            [-0.27, -0.26],
            [-0.2669, -0.262],
            [-0.2665, -0.264],
            [-0.26670, -0.265]]


# For the Bayesian analysis
alpha_range = [-0.4, 0.4]
f0_range = [0, 1]
f1_range = [-20, 20]
beta_range = [-15, 15]
nu_range = [0, 15]

# Parameters used in publication_results.py analysis
param_name_dict = {}
param_name_dict[model1_1a.__name__] = ["alpha", "f0", "f1", "beta", "nu"]
param_name_dict[model1_2a.__name__] = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

prior_range_dict = {}
prior_range_dict[model1_1a.__name__] = [alpha_range, f0_range, f1_range, beta_range, nu_range]
prior_range_dict[model1_2a.__name__] = [alpha_range, alpha_range, f0_range, f1_range, beta_range,
                                        nu_range]

param_dict = {}

# N = 2
param_dict[2] = {}

param_dict[2]["model_1"] = model1_1a
param_dict[2]["model_2"] = model2_1a
param_dict[2]["param_names"] = param_name_dict[param_dict[2]["model_1"].__name__]
param_dict[2]["Central_Bbar"] = [0.52, 0.53]
param_dict[2]["Bbar_list"] = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
param_dict[2]["g_s"] = [0.1, 0.2, 0.3, 0.5, 0.6]
param_dict[2]["L_s"] = [8, 16, 32, 48, 64, 96, 128]
param_dict[2]["x0"] = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
param_dict[2]["gL_max"] = 32
param_dict[2]["gL_central"] = {"model_1": 12.8, "model_2": 24}
param_dict[2]["h5_data_file"] = "Bindercrossings.h5"
param_dict[2]["MCMC_data_file"] = "MCMCdata.h5"
param_dict[2]["prior_range"] = prior_range_dict[param_dict[2]["model_1"].__name__]
param_dict[2]["therm"] = 0


# N = 3
param_dict[3] = {}

param_dict[3]["model_1"] = model1_1a
param_dict[3]["model_2"] = model2_1a
param_dict[3]["param_names"] = param_name_dict[param_dict[3]["model_1"].__name__]
param_dict[3]["Central_Bbar"] = [0.43, 0.44]
param_dict[3]["Bbar_list"] = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48]
param_dict[3]["g_s"] = [0.1, 0.2, 0.3, 0.5, 0.6]
param_dict[3]["L_s"] = [8, 16, 32, 48, 64, 96, 128]
param_dict[3]["x0"] = [0, 0.5, -0.03, 1, 2 / 3]
param_dict[3]["gL_max"] = 32
param_dict[3]["gL_central"] = {"model_1": 12.8, "model_2": 24}
param_dict[3]["h5_data_file"] = "Binder_N3_data.h5"
param_dict[3]["MCMC_data_file"] = "MCMC_N3_data.h5"
param_dict[3]["prior_range"] = prior_range_dict[param_dict[2]["model_1"].__name__]
param_dict[3]["therm"] = 10000


# N = 4
param_dict[4] = {}

param_dict[4]["model_1"] = model1_2a
param_dict[4]["model_2"] = model2_2a
param_dict[4]["param_names"] = param_name_dict[param_dict[4]["model_1"].__name__]
param_dict[4]["Central_Bbar"] = [0.42, 0.43]
param_dict[4]["Bbar_list"] = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47]
param_dict[4]["g_s"] = [0.1, 0.2, 0.3, 0.5, 0.6]
param_dict[4]["L_s"] = [8, 16, 32, 48, 64, 96, 128]
param_dict[4]["x0"] = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
param_dict[4]["gL_max"] = 32
param_dict[4]["gL_central"] = {"model_1": 12.8, "model_2": 32}
param_dict[4]["h5_data_file"] = "Bindercrossings.h5"
param_dict[4]["MCMC_data_file"] = "MCMCdata.h5"
param_dict[4]["prior_range"] = prior_range_dict[param_dict[2]["model_1"].__name__]
param_dict[4]["therm"] = 0


h5data_dir = "h5data/"
