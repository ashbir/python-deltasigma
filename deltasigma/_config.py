# -*- coding: utf-8 -*-
# _config.py
# Module providing configuration switches
# Copyright 2013 Giuseppe Venturini
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

"""Module providing configuration switches.
"""

import os
import sys
from warnings import warn

import numpy as np

# should synthesizeNTF run the optimization routine?
optimize_NTF = True

# how many iterations should be allowed in NTF synthesis?
# see synthesizeNTF() for more
itn_limit = 500

# debug
_debug = False

# get blas information to compile the cython extensions
blas_info = {}
if len(blas_info) == 0 and _debug:
    warn("Numpy did not detect the BLAS library in the system")
# Let's make an educated guess
guessed_include = None
if 'linux' in sys.platform or 'darwin' in sys.platform:
    # Common locations for cblas.h on Linux and macOS
    search_dirs = [
        '/usr/include',
        '/usr/include/cblas',
        '/usr/include/openblas',
    ]
    # Add architecture-specific paths for Debian/Ubuntu derivatives
    if 'linux' in sys.platform:
        try:
            import platform
            arch = platform.machine()
            arch_dir = '/usr/include/%s-linux-gnu' % arch
            if os.path.isdir(arch_dir):
                search_dirs.append(arch_dir)
        except ImportError:
            pass  # platform module not available, skip

    for d in search_dirs:
        if os.path.isfile(os.path.join(d, 'cblas.h')):
            guessed_include = d
            break

# wrap it up: numpy or user-set environment var or a lucky guess on our side is
# needed to get the cblas.h header path. If not found, simulateDSM() will use
# a CPython implementation (slower).
"""
setup_args = {"script_args":(["--compiler=mingw32"]
                             if sys.platform == 'win32' else [])}
"""
setup_args = {"script_args":[]}
lib_include = [np.get_include()]
if "include_dirs" not in blas_info and "BLAS_H" not in os.environ and \
   'nt' not in os.name and not guessed_include:
    warn("Cannot find the path for 'cblas.h'. You may set it using the environment variable "
         "BLAS_H.\nNOTE: You need to pass the path to the directories were the "
         "header files are, not the path to the files.")
else:
    if "include_dirs" in blas_info:
        lib_include = lib_include + blas_info.get("include_dirs")
    elif "BLAS_H" in os.environ:
        lib_include = lib_include + [os.environ["BLAS_H"]]
    elif guessed_include:
        lib_include = lib_include + [guessed_include]
    else:
        pass # we're on windows
setup_args.update({"include_dirs":list(set(lib_include))})
