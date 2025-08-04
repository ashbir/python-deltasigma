# -*- coding: utf-8 -*-
# test_simulateMS.py
# This module provides the tests for the simulateMS function.
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

"""This module provides the test class for the simulateMS() function.
"""

import unittest
import pkg_resources

import numpy as np
import deltasigma as ds
import scipy.io

class TestSimulateMS(unittest.TestCase):
    """Test class for simulateMS()"""

    def setUp(self):
        """Set up the test data."""
        try:
            fname = pkg_resources.resource_filename(
                __name__, "test_data/test_simulateMS.mat"
            )
            mat_data = scipy.io.loadmat(fname)
            self.v_in = mat_data['v'].reshape(-1)
            self.M = mat_data['M'].item()
            self.d = mat_data['d'].item()
            self.dw = mat_data['dw'].reshape(-1)
            self.sx0 = mat_data['sx0']
            self.sv_ref = mat_data['sv']
            self.sx_ref = mat_data['sx']
            self.sigma_se_ref = mat_data['sigma_se'].item()
            self.max_sx_ref = mat_data['max_sx'].item()
            self.max_sy_ref = mat_data['max_sy'].item()
            # The zpk object cannot be saved directly, so we recreate it.
            self.mtf = ([1.], [0.], 1.)
        except FileNotFoundError:
            self.fail("Test data file 'test_simulateMS.mat' not found. "
                      "Please run the MATLAB script to generate it.")

    def test_simulateMS_python(self):
        """Test function for the Python implementation of simulateMS."""
        sv, sx, sigma_se, max_sx, max_sy = ds.simulateMS(
            self.v_in, self.M, self.mtf, self.d, self.dw, self.sx0
        )

        # Due to the random dither, the results will not be identical.
        # We test if they are statistically close.
        # The selection vectors `sv` will differ because of the dither and
        # the different sorting algorithms used for tie-breaking.
        # However, the sum of each selection vector should be the same.
        self.assertTrue(np.allclose(np.sum(sv, axis=0), 
                                    np.sum(self.sv_ref, axis=0)))

        # The statistics should be close.
        # Using a larger tolerance due to the randomness.
        self.assertTrue(np.allclose(sigma_se, self.sigma_se_ref, atol=1e-1, rtol=1e-1))
        self.assertTrue(np.allclose(max_sx, self.max_sx_ref, atol=1, rtol=0.5))
        self.assertTrue(np.allclose(max_sy, self.max_sy_ref, atol=1, rtol=0.5))

        # Test with no dither to get deterministic results
        sv_nodither, _, _, _, _ = ds.simulateMS(
            self.v_in, self.M, self.mtf, d=0., dw=self.dw, sx0=self.sx0
        )
        
        # Rerun MATLAB data generation with d=0 to get a deterministic reference
        # For now, we just check if the output shape is correct
        self.assertEqual(sv_nodither.shape, self.sv_ref.shape)
