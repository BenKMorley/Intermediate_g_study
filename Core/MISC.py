###############################################################################
# Source file: ./MISC.py
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
import numpy
from math import floor
import sys
from parameters import *

separator = "##########################################################"


# Routines for computing the integrated autocorrelation time
eps = sys.float_info.epsilon


# Naming conventions
def GRID_convention_m(m):
    return f"m2{m}".rstrip('0')


def MCMC_convention_m(m):
    return f"msq={-m:.8f}"


def GRID_convention_N(N):
    return f"su{N}"


def MCMC_convention_N(N):
    return f"N={N}"


def GRID_convention_L(L):
    return f"L{L}"


def MCMC_convention_L(L):
    return f"L={L}"


def GRID_convention_g(g):
    return f"g{g}".rstrip('0').rstrip('.')


def MCMC_convention_g(g):
    return f"g={g:.2f}"


def calc_gL_mins(g_s, L_s):
    return numpy.sort(list(set(numpy.outer(g_s, L_s).reshape(len(g_s) * len(L_s))))).round(2)


class UWerr():
    def __init__(self, Data, Stau, Name, function=[], *varargin):
        """
            Compute integrated autocorrelation time based on U. Wolff's "Monte
            Carlo Errors with less errors" http://arxiv.org/abs/hep-lat/0306017

            implementation of
            Data:
                columns:	different observables
                lines: 	consecutive measurements

            Stau:
                Stau=0:	no autocorrelations
        """
        self.Data = Data
        self.Stau = Stau
        self.Name = Name
        self.function = function
        self.varargin = varargin
        self.dim = Data.shape
        self.N = self.dim[0]

    def doit(self):
        if len(self.dim) == 2:
            self.Nobs = self.dim[1]
        else:
            self.Nobs = 1

        # means of primary observables
        v = numpy.mean(self.Data, 0)

        # means of secondary observables
        if self.function == []:  # if only primary observables
            fv = v
        else:
            if self.varargin:
                fv = self.function(v, self.varargin)
            else:
                fv = self.function(v)

        # derivative with respect to primary observables:
        D = []
        if self.function == []:
            delta = self.Data - v

        else:
            dum = numpy.array(v)
            h = numpy.std(self.Data, 0) / numpy.sqrt(self.N)
            D = h * 0.

            for i in range(self.Nobs):
                if h[i] == 0:
                    D[i] = 0

                else:
                    dum[i] = v[i] + h[i]

                    if self.varargin:
                        D[i] = self.function(dum, self.varargin)

                    else:
                        D[i] = self.function(dum)

                    dum[i] = v[i] - h[i]

                    if self.varargin:
                        D[i] = D[i] - self.function(dum, self.varargin)

                    else:
                        D[i] = D[i] - self.function(dum)

                    dum[i] = numpy.array(v[i])
                    D[i] = D[i] / (2. * h[i])

            delta = numpy.dot((self.Data - v), D)

        Gamma = numpy.zeros(int(numpy.floor(1. * self.N / 2)))

        try:
            Gamma[0] = numpy.mean(delta ** 2)

        except RuntimeWarning:
            print("Overflow: Gamma[0] = inf")
            Gamma[0] = numpy.inf

        if Gamma[0] == 0:
            print("UWerr: data contains no no fluctuations: Gamma[0]=0")
            exit()

        if self.Stau == 0:
            Wop = 0
            tmax = 0
            doGamma = 0

        else:
            tmax = int(numpy.floor(1. * self.N / 2.))
            doGamma = 1
            Gint = 0

        t = 1
        t = 1

        while t <= tmax:
            try:
                Gamma[t] = numpy.sum(delta[0: -(t)] * delta[t:]) / (self.N - t)
            except RuntimeWarning:
                Gamma[t] = numpy.inf

            if doGamma == 1:
                Gint = Gint + Gamma[t] / Gamma[0]
                if Gint <= 0:
                    tauW = eps

                else:
                    tauW = self.Stau / (numpy.log((Gint + 1.) / Gint))

                try:
                    gW = numpy.exp(-1. * t / tauW) - tauW / numpy.sqrt(t * self.N)

                except RuntimeWarning:
                    # Underflow
                    if tauW < 1:
                        gW = -1

                    # Overflow
                    else:
                        gW = 1

                if gW < 0:
                    Wopt = t
                    tmax = numpy.min([tmax, 2 * t])
                    doGamma = 0

            t = t + 1

        if doGamma == 1:
            print("UWerr: windowing condition failed up to W=%d\n" % (tmax))
            Wopt = tmax

        Gamma = Gamma[:t]
        GammaOpt = Gamma[0] + 2. * numpy.sum(Gamma[1: Wopt + 1])

        if GammaOpt <= 0:
            print("UWerr: Gamma pathological with error below zero")
            raise ValueError

        Gamma = Gamma + GammaOpt / self.N
        GammaOpt = Gamma[0] + 2. * numpy.sum(Gamma[1: Wopt])
        dv = numpy.sqrt(GammaOpt / self.N)
        ddv = dv * (numpy.sqrt((Wopt + .5) / self.N))
        rho = Gamma / Gamma[0]
        tauint = (numpy.cumsum(rho)[Wopt] - 0.5)
        dtauint = tauint * 2 * numpy.sqrt((Wopt - tauint + 0.5) / self.N)

        return (fv, dv, ddv, tauint, dtauint, Wopt)


# routine to flatten nested lists or arrays
def flatten(x):
    """
        flatten(sequence) -> list

        Returns a single, flat list which contains all elements retrieved
        from the sequence and all recursively contained sub-sequences
        (iterables).

        Examples:
        >>> [1, 2, [3,4], (5,6)]
        [1, 2, [3, 4], (5, 6)]
        >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
        [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]
    """
    result = []

    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))

        else:
            result.append(el)

    return result


# utility to check whether inumpyut parameters match existing simulations
def check_exists(ss, x):
    """
        Checks if simulation data for a particular inumpyut value of ag, L/a and N
        exists, exists if not
    """
    if ss == 'L/a':
        isin = x in Ls
        available = Ls

    elif ss == 'ag':
        isin = x in gs
        available = gs

    elif ss == 'N':
        isin = x in Ns
        available = Ns

    if not isin:
        print("Value of %s=%s not available" % (ss, x))
        print("Available values are ", ' '.join([str(i) for i in available]))
        exit()


def disperr3(val, dval):
    """
        Helper routine for nicely printing results with error bars.
        Based on MATLAB script by Roland Hoffmann
    """
    n = len(val)

    if n != len(dval):
        print("val and dval must have the same length!")
        print(val, dval)
        print("exiting")
        exit()

    dig = 2
    res = n * ['']
    for i in range(n):
        if dval[i] == 0. and val[i] == 0.:
            res[i] = "0"

        elif numpy.isnan(val[i]) or numpy.isnan(dval[i]):
            res[i] = "nan"

        elif dval[i] == 0. and val[i] != 0.:
            value = "%d" % val[i]
            res[i] = value

        elif dval[i] < 1:
            location = int(numpy.floor(numpy.log10(dval[i])))
            append_err = "(" + str(int(numpy.round(dval[i] * 10 **
                                                (-location + dig - 1)))) + ")"

            if numpy.abs(val[i]) < 1e-100:
                val[i] = 0.
                location = 1

            valformat = "%." + str(-location + dig - 1) + "f"
            sval = valformat % val[i]
            res[i] = sval + append_err

        elif dval[i] >= 1:
            digits = min(0, int(numpy.ceil(numpy.log10(dval[i])) - 1)) + 1
            error = numpy.around(dval[i], digits)
            value = numpy.around(val[i], digits)
            serr = "%." + str(digits) + "f(%." + str(digits) + "f)"
            serr = serr % (value, error)
            res[i] = serr  # str(value)+"("+str(error)+")"

        else:
            digits = max(0, int(numpy.ceil(numpy.log10(dval[i])) - 1))
            error = int(round(dval[i] / 10 ** digits) * 10 ** digits)
            value = round(val[i] / 10 ** digits) * 10 ** digits
            res[i] = str(value) + "(" + str(error) + ")"

    return res


def weight(z, f, v):
    """
        Calculate the weight used for reweighting to a given mass given the mean of the
        distribution we're reweighting to is z sigma away from the mean we're reweighting from
        where sigma is the standard deviation of the distribution we are reweighting from.

        INPUTS:
        -------
        z:  float
        f:  float, this is the upper bound on deviation between the means of the extrapolated
            tr(phi^2) distribution and the tr(phi^2) mean of the mass point we are reweighting
            from, for this mass to be used in a reweighting. This variable is called
            self.frac_of_dist.
        v:  float, the width of the transition from full inclusion (weight = 0) to no inclusion
            (weight = 1)
    """
    assert v >= 0, "we need a finite width to the distribution"
    assert v <= f, "The transition needs to fit in the interval [0, f]"

    if v == 0:
        if z <= f:
            return 1

        else:
            return 0

    else:
        a = 2 / v ** 2

        if z < f - v:
            return 1

        elif z <= f - v / 2:
            return 1 - a * (z - (f - v)) ** 2

        elif z <= f:
            return a * (f - z) ** 2

        else:
            return 0


def weight_centered(z, f, v):
    """
        Calculate the weight used for reweighting to a given mass given the mean of the
        distribution we're reweighting to is z sigma away from the mean we're reweighting from
        where sigma is the standard deviation of the distribution we are reweighting from.

        INPUTS:
        -------
        z:  float, number of sigma difference
        f:  float, this is the deviation between the means of the extrapolated
            tr(phi^2) distribution and the tr(phi^2) mean of the mass point we are reweighting
            from, for this mass to be used in a reweighting. This is the centre of the distribution
            which has a weight of 0.5. This variable is called self.frac_of_dist.
        v:  float, the width of the transition from full inclusion in units of sigma
    """
    assert v >= 0, "we need a finite width to the distribution"
    assert v <= 2 * f, "The transition needs to fit in the interval [0, f]"

    if v == 0:
        if z <= f:
            return 1

        else:
            return 0

    else:
        a = 2 / v ** 2

        if z < f - v / 2:
            return 1

        elif z < f:
            return 1 - a * (z - (f - v / 2)) ** 2

        elif z < f + v / 2:
            return a * (f + v / 2 - z) ** 2

        else:
            return 0


def nice_string_print(mean, std):
    """
        Produce a string that represents a mean value with standard deviation such that the
        first two significant figures of the standard deviation are kept, and the appropriate
        number of s.f. of the mean are kept such that the smallest digit matches the second
        digit of the standard deviation

        Examples:
        ---------
        mean = 584
        std = 63
        output: "584(63)"

        mean = 687.245
        std = 8.6
        output: "687.2(8.6)"

        mean = 0.643
        std = 21
        output: "0(21)"
    """
    digit = floor(numpy.log10(abs(mean)))

    # We want the standard deviation rounded to 2 s.f.
    digit = floor(numpy.log10(abs(std)))
    std = numpy.round(std, -digit + 1)

    # Recaculate the digit for the edge case of a std that rounds up to 100
    digit = floor(numpy.log10(abs(std + sys.float_info.epsilon)))

    # Now round the mean to the same level as the standard deviation
    mean = numpy.round(mean, -digit + 1)

    # For the purpose of nice printing we want the std to print as a two digit number
    # unless it should contain a decimal point
    if mean == 0:
        digit2 = 0

    else:
        digit2 = floor(numpy.log10(abs(mean)))

    # We need to find the trailing zeros on the mean
    int_mean = mean * 10 ** (-digit + 1)
    factor = 10
    trailing_zeros = 0

    if digit != 0 and digit2 != 0:
        while(int_mean // factor != 0):
            if int_mean % factor == 0:
                factor *= 10
                trailing_zeros += 1

            else:
                break

    if mean == 0:
        if (digit < 2):
            string = f"0({std})"

        elif (digit > 10):
            string = f"0({std:.2g})"

        else:
            string = f"0({int(std)})"

    elif digit == 0:
        string = f"{mean}{'0' * trailing_zeros}({std:.1f})"

    elif digit >= 1:
        string = f"{int(mean)}({int(std)})"

    else:
        std *= 10 ** (-digit + 1)
        string = f"{mean}{'0' * trailing_zeros}({std:.0f})"

    return string
