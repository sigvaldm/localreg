import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

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

def train(indep, centers, dep, rbf):
    assert indep.shape[0]==dep.shape[0]
    assert indep.shape[1]==centers.shape[1]
    matrix = np.zeros((len(indep), len(centers)), dtype=float)
    for i in tqdm(range(len(indep))):
        for j in range(len(centers)):
            distance = np.linalg.norm(indep[i,:]-centers[j,:])
            matrix[i,j] = rbf(distance)
    coeffs, residual, rank, svalues = np.linalg.lstsq(matrix, dep, rcond=None)
    return coeffs, residual

def predict(coeffs, indep, centers, rbf):
    res = 0.
    for j in range(len(centers)):
        res += coeffs[j]*rbf(np.linalg.norm(indep-centers[j,:]))
    return res

N = 100
data = read_RBF_data('langmuir/langmuir_training_data.in')
currents = data[:,0:4]*1e6 # [uA]
# density = data[:,5]*1e-12
density = data[:,4]*1e-10
centers = kmeans_centers(currents[:,0:4], N)
# plot_data(currents, centers)

coeffs, residual = train(currents, centers, density, Gaussian)
pred = predict(coeffs, currents[0,:], centers, Gaussian)
print(pred, density[0], (pred-density[0])/density[0])
