#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import sys
from frmt import print_table
from time import time

def read_RBF_data(fname):
    """
    Read dataset in Marchand's RBF format

    Parameters
    ----------
    fname: string
        filename

    Returns
    -------
    numpynp.array where data[i,j] is row i, column j
    """
    data = []
    with open(fname) as file:
        file.readline()
        file.readline()
        for line in file:
            data.append(line.split())
    data = np.array(data, dtype=float)
    return data

def kmeans_centers(data, N, random_state=None):
    """
    Get the centers of the radial basis functions (pivots) using K-means
    clustering.

    Parameters
    ----------
    data: numpy.ndarray
        data[i,j] is data point i, coordinate j

    N: int
        Number of clusters/centers for radial bases to compute

    random_state: int or None
        When centers is an integer, random numbers are used to compute the
        best possible centers. This may lead to slightly different results,
        since the seed of the random number generator is different every
        time. For exact reproducibility, set random_state to an integer.
        See also the random_state variable in Scikit-Learn.

    Returns
    -------
    numpy.ndarray where centers[i,j] is center i, coordinate j
    """
    clustering = KMeans(n_clusters=N, random_state=random_state).fit(data)
    centers = clustering.cluster_centers_
    return centers

def plot_data(data, net):
    """
    Plots data points and centers
    """
    for s, (i, j) in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
        plt.subplot(2,3,s+1)
        plt.plot(data[:,i], data[:,j], '.')
        net.plot_centers(plt.gca(), [i,j])
        plt.xlabel('$I_{}\,[\mu A]$'.format(i))
        plt.ylabel('$I_{}\,[\mu A]$'.format(j))
    plt.show()

def gaussian(t):
    return np.exp(-0.5*t**2)

class RBFnet(object):

    def __init__(self):
        self.untrained = True

    def autotrain(self, input, output, centers=10, rbf=gaussian, random_state=None):

        def f(radius):
            self.train(input, output, centers=centers, radius=radius, random_state=5, reuse=True)
            err = net.error(*training_set)
            print(radius[0], err)
            return err

        radius_0 = 1
        res = minimize(f, radius_0, tol=0.00001, options={'maxiter':100, 'disp':True},
                       # method = 'powell')
                       method = 'nelder-mead')
        print(res.x)


    def train(self, input, output, centers=10, rbf=gaussian, radius=1,
              random_state=None, reuse=False):
        """
        Train the RBF net.

        Parameters
        ----------
        input: numpy.ndarray
            input[i,j] is data point i, independent variable j

        output: numpy.ndarray
            output[i] is data point i, dependent variable

        centers: int or numpy.ndarray
            If centers is an array, the centers of the radial basis functions
            are pre-specified, and centers[i,j] is center i, coordinate j. If
            centers is an integer, it is the number of radial basis functions,
            and their centers will be created internally by K-means clustering.

        rbf: function
            Which radial basis function to use. The function should take a
            single scalar argument, and is shifted according to the centers and
            scaled according to the radius internally according to the
            following equation:

                rbf((input-center)/radius)

        radius: float
            The radius to use in the function rbf in terms of standard
            deviations. It is a hyperparameter in order-of-magnitude unity.

        reuse: bool
            Whether or not to reuse the centers and normalizations from
            previous training. This speeds up training, but should only be
            done when training on the same, or at least similar, data.
            Useful for hyperparameter tuning.
        """

        if self.untrained or reuse==False:
            self.input_shift = np.mean(input, axis=0)
            self.input_scale = np.std(input, axis=0)
            self.output_shift = np.mean(output, axis=0)
            self.output_scale = np.std(output, axis=0)

        inp = (input-self.input_shift)/self.input_scale
        outp = (output-self.output_shift)/self.output_scale

        if self.untrained or reuse==False:
            if not isinstance(centers, np.ndarray):
                self.centers = kmeans_centers(inp, centers, random_state=random_state)
            else:
                self.centers = (centers-self.input_shift)/self.input_scale

        assert inp.shape[0]==outp.shape[0]
        assert inp.shape[1]==self.centers.shape[1]

        matrix = np.zeros((len(inp), len(self.centers)), dtype=float)
        for j in range(len(self.centers)):
            distance = np.linalg.norm(inp[:,:]-self.centers[j,:], axis=1)
            matrix[:,j] = rbf(distance/radius)

        coeffs, residual, rank, svalues = np.linalg.lstsq(matrix, outp, rcond=None)
        self.coeffs = coeffs
        self.residual = residual

        self.rbf = rbf
        self.radius = radius
        self.untrained = False

    def plot_centers(self, axis, indeps=(0,1), *args, **kwargs):
        """
        Plot the cluster centers projected onto a plane spanned by the axes
        of two of the independent variables.

        Parameters
        ----------
        axis: matplotlib.axis
            Axis on which to plot the centers

        indeps: list or tuple with two elements
            The independent variables to use as x- and y-axes

        Further accepts the same arguments as matplotlib.axis.plot.
        """
        indeps = list(indeps)
        unnormalized_centers = net.centers[:,indeps]*net.input_scale[indeps]+net.input_shift[indeps]
        axis.plot(*unnormalized_centers.T, *args, 'x', **kwargs)

    def predict(self, input):
        """
        Make one or more predictions using the RBF net.

        Parameters
        ----------
        input: numpy.ndarray
            input[i,j] is prediction point i, independent variable j

        Returns
        -------
        numpy.ndarray where output[i] is the prediction of point i
        """
        inp = (input-self.input_shift)/self.input_scale

        output = np.zeros(inp.shape[0])
        for j in range(len(self.centers)):
            distance = np.linalg.norm(inp-self.centers[j,:], axis=1)
            output += self.coeffs[j]*self.rbf(distance/self.radius)

        output = output*self.output_scale + self.output_shift

        return output

    def error(self, input, output):
        """
        The prediction error of the RBF net for a given set of data points.
        This will be the training error, validation error or test error
        depending on whether the set is the training set, the validation set
        or the test set.

        Parameters
        ----------
        input: numpy.ndarray
            input[i,j] is data point i, independent variable j

        output: numpy.ndarray
            output[i] is data point i, dependent variable
        """
        predicted = self.predict(input)
        error = predicted-output
        relative_error = error/output
        mu = np.mean(relative_error)
        sigma = np.std(relative_error)
        rms = np.sqrt(np.sum(error**2))
        # print(np.linalg.norm(deviations)**2)
        # print_table(zip(output, predicted, deviations, reldev), format='{:g}')
        # print_table(zip(input, output, predicted))
        # print(np.array(list(zip(input, output, predicted))))
        # return rms
        return np.mean(np.abs(relative_error))

N = 100
M = 9000
M2 = 10000
K = 20
data = read_RBF_data(sys.argv[1])
currents = data[:,0:4]
density = data[:,4]
temperature = data[:,5]
voltage = data[:,6]

training_set = (currents[:M], density[:M])
validation_set = (currents[M:M2], density[M:M2])

# training_set = (currents[:M], voltage[:M])
# validation_set = (currents[M:M2], voltage[M:M2])

# training_set = (currents[:M], temperature[:M])
# validation_set = (currents[M:M2], temperature[M:M2])

centers = 50

net = RBFnet()
# net.train(*training_set, centers=centers, radius=1.5, random_state=5)
net.autotrain(*training_set, centers=centers, random_state=5)

pred = net.predict(training_set[0][:K])
error = net.error(*validation_set)
print("Relative validation error: ", error)

# input_shift = np.mean(training_set[0], axis=0)
# input_scale = np.std(training_set[0], axis=0)
# input = (training_set[0]-input_shift)/input_scale

# lower = np.min(input, axis=0)
# upper = np.max(input, axis=0)
# maxdist = max(upper-lower)
# dim = len(upper)
# radius_0 = maxdist/(centers**(1./dim))
# print('guess: ', radius_0)

# centers = kmeans_centers(training_set[0], centers, 5)
# centers_center = np.mean(centers)
# print('centers_center: ', centers_center, np.std(centers))

# i = 0
# def f(radius):
#     net.train(*training_set, centers=centers, radius=radius, random_state=5, reuse=True)
#     global i
#     i = i + 1
#     err = net.error(*training_set)
#     print(i, radius[0], err)
#     return err

# radius_0 = 1
# res = minimize(f, radius_0, tol=0.00001, options={'maxiter':100, 'disp':True},
#                # method = 'powell')
#                method = 'nelder-mead')
# print(res.x)

# rs = np.logspace(-2, 4, 200)
# err = []
# for r in tqdm(rs):
#     net.train(*training_set, centers=centers, radius=r, random_state=5, reuse=True)
#     err.append(net.error(*training_set))

# plt.figure()
# plt.loglog(rs, err)
# plt.show()


# print(currents.shape)
# plot_data(currents, net)


pred = net.predict(training_set[0][:K])
# print_table(zip(pred, density[0:K], (pred-density[0:K])/density[0:K]))
error = net.error(*validation_set)
print("Relative validation error optimized: ", error)

plot = plt.plot
plt.figure()
# plot(density[:M2:10], net.predict(currents[:M2:10]), '+', ms=3)
# plot([1e11,12e11],[1e11,12e11], '-k', lw=0.8)
plot(training_set[1], net.predict(training_set[0]), '+', ms=3)
r = (min(training_set[1]), max(training_set[1]))
plot(r,r,'-k',lw=0.8)
plt.show()
