"""
Copyright 2019 Sigvald Marholm <marholm@marebakken.com>

This file is part of localreg.

localreg is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

localreg is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with localreg.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

def rms_error(true, pred):
    """Root-mean-square of error"""
    e = pred-true
    return np.sqrt(np.sum(e**2)/len(e))

def rms_rel_error(true, pred):
    """Root-mean-square of relative error"""
    e = (pred-true)/true
    return np.sqrt(np.sum(e**2)/len(e))

def max_abs_error(true, pred):
    """Maximum absolute error"""
    e = pred-true
    return np.max(np.abs(e))

def max_rel_error(true, pred):
    """Maximum relative value"""
    e = (pred-true)/true
    return np.max(np.abs(e))

def mean_abs_error(true, pred):
    """Mean absolute error"""
    e = pred-true
    return np.mean(np.abs(e))

def mean_rel_error(true, pred):
    """Mean relative error"""
    e = (pred-true)/true
    return np.mean(np.abs(e))

def error_bias(true, pred):
    """Bias in error"""
    e = pred-true
    return np.mean(e)

def rel_error_bias(true, pred):
    """Bias in relative error"""
    e = (pred-true)/true
    return np.mean(e)

def error_std(true, pred):
    """Standard deviation in error"""
    e = pred-true
    return np.std(e)

def rel_error_std(true, pred):
    """Standard deviation in relative error"""
    e = (pred-true)/true
    return np.std(e)

