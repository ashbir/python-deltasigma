# -*- coding: utf-8 -*-
# _simulateMS_python.py
# Module providing the CPython simulateMS function
# Copyright 2025 The python-deltasigma contributors
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

"""Module providing the simulateMS() function for mismatch shaping.
"""

from __future__ import division

import numpy as np
from ._utils import _get_zpk

def _select_element(v, sy, dw):
    """Select elements for a mismatch-shaping DAC.

    This is a Python port of the selectElement MATLAB function.
    It uses a greedy algorithm which is fast but may not be optimal
    for the general subset-sum problem with non-unit elements.

    Parameters:
    * v: The desired output value.
    * sy: The vector of desired usage for each element.
    * dw: The vector of element weights.

    Returns:
    * sv: The selection vector.
    """
    M = len(dw)
    
    # The problem of finding the selection vector `sv` such that `sv.dot(dw) == v`
    # is the subset-sum problem, which is NP-hard.
    # The original MATLAB code uses a recursive search, which is slow.
    # A common and practical approach is to use a greedy algorithm,
    # especially when element weights are close to uniform.

    # Case 1: All elements are unit elements.
    if np.all(dw == 1):
        # We need to select `n_ones` elements.
        # `v = n_ones * 1 + (M - n_ones) * (-1) = 2*n_ones - M`
        # `n_ones = (v + M) / 2`
        n_ones = int(round((v + M) / 2.0))
        sv = -np.ones(M)
        # Select the elements with the highest desired usage (sy).
        indices = np.argsort(sy)[::-1][:n_ones]
        sv[indices] = 1
        return sv

    # Case 2: Non-unit elements. Use a greedy approach.
    v_rem = (v + np.sum(dw)) / 2.
    sv = -np.ones(M)
    # Prioritize elements with higher `sy` values.
    order = np.argsort(sy)[::-1]
    for i in order:
        if v_rem >= dw[i]:
            sv[i] = 1
            v_rem -= dw[i]
    
    return sv

def simulateMS(v, M=16, mtf=None, d=0., dw=None, sx0=None):
    """Simulate a mismatch-shaping DAC in pure Python.

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
    if mtf is None:
        # Default MTF is z-1
        mtf = ([1.], [0.], 1.)
    
    zeros, poles, _ = _get_zpk(mtf)
    order = len(poles)

    if sx0 is None and order > 0:
        sx0 = np.zeros((order, M))
    elif order == 0:
        sx0 = np.zeros((0, M))

    if dw is None:
        dw = np.ones(M)

    # B/A = MTF-1
    if order > 0:
        num = np.poly(zeros)
        den = np.poly(poles)
        A = -np.real(den[1:])
        B = np.real(num[1:]) + A
    else:
        A, B = np.array([]), np.array([])

    N = len(v)
    sv = np.zeros((M, N))

    sx = sx0.copy()
    max_sx = np.max(np.abs(sx)) if sx.size > 0 else 0
    max_sy = 0
    sum_se2 = 0

    for i in range(N):
        # Compute the sy vector.
        sy = B.dot(sx) if order > 0 else np.zeros(M)
        # Normalize sy for a minimum value of zero.
        sy = sy - np.min(sy)
        # Add dither
        dithered_sy = sy + d * (2 * np.random.rand(M) - 1)
        # Pick the elements
        sv[:, i] = _select_element(v[i], dithered_sy, dw)
        # Compute the selection error.
        se = sv[:, i] - sy
        # Compute the new sx matrix
        if order > 0:
            sxn = A.dot(sx) + se
            sx = np.vstack((sxn, sx[:-1, :]))
        
        # Keep track of some statistics.
        sum_se2 += np.sum((se - np.mean(se))**2)
        if sx.size > 0:
            max_sx = np.max((max_sx, np.max(np.abs(sx[0, :]))))
        max_sy = np.max((max_sy, np.max(np.abs(sy))))

    sigma_se = np.sqrt(sum_se2 / (M * N)) if (M * N) > 0 else 0
    
    return sv, sx, sigma_se, max_sx, max_sy
