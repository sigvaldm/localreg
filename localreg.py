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

def polyfit(x, y, x0, weights=None, degree=2):

    s = np.sqrt(weights)

    X = x[:, None]**np.arange(degree + 1)
    X0 = x0[:, None]**np.arange(degree + 1)

    beta = np.linalg.lstsq(X*s[:, None], y*s, rcond=None)[0]

    return X0.dot(beta)

def rectangular(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = 0.5
    return res

def triangular(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = 1-np.abs(t[ind])
    return res

def epanechnikov(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = 0.75*(1-t[ind]**2)
    return res

def biweight(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = (15/16)*(1-t[ind]**2)**2
    return res

def triweight(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = (35/32)*(1-t[ind]**2)**2
    return res

def tricube(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = (1-np.abs(t[ind])**3)**3
    return res

def gaussian(t):
    res = (1/np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    return res

def cosine(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t)<=1)
    res[ind] = (np.pi/4)*np.cos(np.pi*t[ind]/2)
    return res

def logistic(t):
    res = 1/(np.exp(t)+2+np.exp(-t))
    return res

def sigmoid(t):
    res = (2/np.pi)/(np.exp(t)+np.exp(-t))
    return res

def silverman(t):
    res = 0.5*np.exp(-np.abs(t)/np.sqrt(2))*np.sin(np.abs(t)/np.sqrt(2)+np.pi/4)
    return res

def localreg(x, y, x0=None, degree=2, kernel=epanechnikov, width=1, frac=None):

    if x0 == None: x0=x

    y0 = np.zeros_like(x0)

    if frac==None:

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

    return y0
