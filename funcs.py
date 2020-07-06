#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import sys
from frmt import print_table

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

def kmeans_centers(data, N):
    """
    Get the centers of the radial basis functions (pivots) using K-means
    clustering.

    Parameters
    ----------
    data: numpy.ndarray
        data[i,j] is data point i, coordinate j

    N: int
        Number of clusters/centers for radial bases to compute

    Returns
    -------
    numpy.ndarray where centers[i,j] is center i, coordinate j
    """
    clustering = KMeans(n_clusters=N).fit(data)
    centers = clustering.cluster_centers_
    return centers

def plot_data(data, centers):
    """
    Plots data points and centers
    """
    for s, (i, j) in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
        plt.subplot(2,3,s+1)
        plt.plot(data[:,i], data[:,j], '.')
        plt.plot(*(centers[:,[i,j]].T), 'x')
        plt.xlabel('$I_{}\,[\mu A]$'.format(i))
        plt.ylabel('$I_{}\,[\mu A]$'.format(j))
    plt.show()

def Gaussian(distance, radius=0.1):
    return np.exp(-0.5*distance/radius)

class RBFnet(object):

    def train(self, input, output, centers, rbf):
        assert input.shape[0]==output.shape[0]
        assert input.shape[1]==centers.shape[1]
        matrix = np.zeros((len(input), len(centers)), dtype=float)
        for i in tqdm(range(len(input))):
            for j in range(len(centers)):
                distance = np.linalg.norm(input[i,:]-centers[j,:])
                matrix[i,j] = rbf(distance)
        coeffs, residual, rank, svalues = np.linalg.lstsq(matrix, output, rcond=None)
        self.coeffs = coeffs
        self.residual = residual
        self.rbf = rbf
        # print(residual)

    def predict(self, input):
        # print(input)
        output = np.zeros(input.shape[0])
        for j in range(len(centers)):
            output += self.coeffs[j]*self.rbf(np.linalg.norm(input-centers[j,:], axis=1))
        return output

    def validate(self, input, output):
        predicted = self.predict(input)
        deviations = np.abs(predicted-output)
        # print(np.linalg.norm(deviations)**2)
        print_table(zip(output, predicted), format='{:g}')
        # print_table(zip(input, output, predicted))
        # print(np.array(list(zip(input, output, predicted))))

N = 100
M = 10000
K = 2
data = read_RBF_data(sys.argv[1])
currents = data[:,0:4]*1e6 # [uA]
# density = data[:,5]*1e-12
density = data[:,4]*1e-10
centers = kmeans_centers(currents[:M,0:4], N)
# plot_data(currents, centers)

# coeffs, residual = train(currents, centers, density, Gaussian)
# pred = predict(coeffs, currents[0,:], centers, Gaussian)
net = RBFnet()
net.train(currents[:M], density[:M], centers, Gaussian)
pred = net.predict(currents[0:K,:])
print(pred, density[0:K], (pred-density[0:K])/density[0:K])
net.validate(currents[:M], density[:M])
