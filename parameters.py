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
Ls = [8, 16, 32, 48, 64, 96, 128]
gs = [0.1, 0.2, 0.3, 0.5, 0.6, 1.0]
Ns = [2, 4]

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
            [-0.0315, -0.0307],
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
