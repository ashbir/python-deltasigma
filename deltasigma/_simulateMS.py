# -*- coding: utf-8 -*-
# _simulateMS.py
# Module providing the simulateMS function for mismatch shaping.
# This file is part of python-deltasigma.
#
# python-deltasigma is a 1:1 Python replacement of Richard Schreier's
# MATLAB delta sigma toolbox (aka "delsigma"), upon which it is heavily based.
# The delta sigma toolbox is (c) 2009, Richard Schreier.
#
# python-deltasigma is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LICENSE file for the licensing terms.

from __future__ import print_function

from warnings import warn

from ._simulateMS_python import simulateMS as _simulateMS_python

warned = False

def simulateMS(v, M=16, mtf=None, d=0., dw=None, sx0=None):
    """Simulate a mismatch-shaping DAC.

    This function simulates a mismatch-shaping DAC using a pure Python
    implementation. It is not performance-optimized.

    Parameters:
    * v: A vector of the digital input values.
    * M: The number of elements.
    * mtf: The mismatch-shaping transfer function (zpk object).
    * d: Dither uniformly distributed in [-d,d] is added to sy.
    * dw: A vector of DAC element weights.
    * sx0: A matrix whose columns are the initial states of the ESL.

    Returns:
    * sv: An MxN matrix whose columns are the selection vectors.
    * sx: An orderxM matrix containing the final state of the ESL.
    * sigma_se: The rms value of the selection error.
    * max_sx: The maximum absolute value of the state for all modulators.
    * max_sy: The maximum absolute value of the input to the VQ.
    """
    global warned
    if not warned:
        warn("The Python implementation of simulateMS is not performance-optimized.")
        warned = True
    return _simulateMS_python(v, M, mtf, d, dw, sx0)
