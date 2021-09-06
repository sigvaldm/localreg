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
#   frac!=None. It could also have a property for its radius?
#

import numpy as np
import logging
import itertools as it
from . import rbf

logger = logging.getLogger('localreg')
logging.basicConfig()

def polyfit(x, y, x0, weights=None, degree=2):

    if len(x.shape) == 1:
        x = x.reshape(-1,1) # No copy. Only makes a view with different shape.

    if len(x0.shape) == 1:
        x0 = x0.reshape(-1,1) # No copy.

    n_samples, n_indeps = x.shape
    n_samples_out, _ = x0.shape

    if len(x)==0:
        tmp = np.nan*np.ones(n_samples_out)
        return tmp

    if weights is None:
        weights = np.ones(n_samples)

    s = np.sqrt(weights)

    # Multivariate bases (1, x, y, x*y, ...) are represented as tuples of exponents.
    B = it.product(*it.repeat(np.arange(degree+1), n_indeps)) # Cartesian product
    B = np.array(list(filter(lambda a: sum(a)<=degree, B)))
    # B = sorted(B, key=sum) # not really necessary

    X = np.ones((n_samples, len(B)))
    X0 = np.ones((n_samples_out, len(B)))

    for i in range(len(B)):

        # Un-optimized:
        # for j in range(x.shape[1]):
        #     X[:,i] *= x[:,j]**B[i,j]
        #     X0[:,i] *= x0[:,j]**B[i,j]

        # Optimized away for-loop:
        X[:,i] = np.product(x[:,:]**B[i,:], axis=1)
        X0[:,i] = np.product(x0[:,:]**B[i,:], axis=1)

    lhs = X*s[:, None]
    rhs = y*s

    # This is what NumPy uses for default from version 1.15 onwards,
    # and what 1.14 uses when rcond=None. Computing it here ensures
    # support for older versions of NumPy.
    rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)

    beta = np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]

    return X0.dot(beta)

def localreg(x, y, x0=None, degree=2, kernel=rbf.epanechnikov, radius=1, frac=None):

    if x0 is None: x0=x

    if len(x.shape) == 1:
        x = x.reshape(-1,1) # No copy. Only makes a view with different shape.

    if len(x0.shape) == 1:
        x0 = x0.reshape(-1,1) # No copy.

    n_samples, n_indeps = x.shape
    n_samples_out, _ = x0.shape

    y0 = np.zeros(n_samples_out, dtype=float)

    if frac is None:

        for i, xi in enumerate(x0):

            weights = kernel(np.linalg.norm(x-xi[None,:], axis=1)/radius)

            # Filter out the datapoints with zero weights.
            # Speeds up regressions with kernels of local support.
            inds = np.where(np.abs(weights)>1e-10)[0]

            y0[i] = polyfit(x[inds], y[inds], xi[None,:],
                            weights[inds], degree=degree)

    else:

        N = int(frac*n_samples)

        for i, xi in enumerate(x0):

            dist = np.linalg.norm(x-xi[None,:], axis=1)
            inds = np.argsort(dist)[:N]
            radius = dist[inds][-1]

            weights = kernel(dist[inds]/radius)

            y0[i] = polyfit(x[inds], y[inds], xi[None,:],
                            weights, degree=degree)

    if np.any(np.isnan(y0)):
        logger.warning("Kernel do not always span any data points")

    return y0
