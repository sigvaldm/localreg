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
# TODO
#
#   One could consider making the kernels callable objects. These objects could
#   then have a member function without if-testing, which is faster in case it
#   is known that all datapoints are to be included. This is the case when
#   frac!=None. It could also have a property for its width?
#

import numpy as np
import logging
from . import rbf

logger = logging.getLogger('localreg')
logging.basicConfig()

def polyfit(x, y, x0, weights=None, degree=2):

    if len(x)==0:
        return np.nan*np.ones_like(x0)

    if weights is None:
        weights = np.ones_like(x)

    s = np.sqrt(weights)

    X = x[:, None]**np.arange(degree + 1)
    X0 = x0[:, None]**np.arange(degree + 1)

    lhs = X*s[:, None]
    rhs = y*s

    # This is what NumPy uses for default from version 1.15 onwards,
    # and what 1.14 uses when rcond=None. Computing it here ensures
    # support for older versions of NumPy.
    rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)

    beta = np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]

    return X0.dot(beta)

def localreg(x, y, x0=None, degree=2, kernel=rbf.epanechnikov, width=1, frac=None):

    if x0 is None: x0=x

    y0 = np.zeros_like(x0)

    if frac is None:

        for i, xi in enumerate(x0):

            weights = kernel(np.abs(x-xi)/width)

            # Filter out the datapoints with zero weights.
            # Speeds up regressions with kernels of local support.
            inds = np.where(np.abs(weights)>1e-10)[0]

            y0[i] = polyfit(x[inds], y[inds], np.array([xi]),
                            weights[inds], degree=degree)

    else:

        N = int(frac*len(x))

        for i, xi in enumerate(x0):

            dist = np.abs(x-xi)
            inds = np.argsort(dist)[:N]
            width = dist[inds][-1]

            weights = kernel(dist[inds]/width)

            y0[i] = polyfit(x[inds], y[inds], np.array([xi]),
                            weights, degree=degree)

    if np.any(np.isnan(y0)):
        logger.warning("Kernel do not always span any data points")

    return y0
