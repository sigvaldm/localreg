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

#!/usr/bin/env python

# TBD:
# - Test that validation error increases
# - Add version check on save/load

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from . import rbf as kernels, metrics

def plot_corr(axis, true, pred, log=False, *args, **kwargs):
    """
    Visualize correlation between true and predicted values. For multivariate
    output, only one variable must be considered at a time.

    Parameters
    ----------
    axis: matplotlib.axis
        Axis to plot on

    true: numpy.array (1D)
        True values

    pred: numpy.array (1D)
        Predicted values (same length as true)

    log: bool
        Whether or not to use logarithmic axes

    Accepts in addition the same arguments as matplotlib.plot
    """
    plot = axis.loglog if log else axis.plot
    plot(true, pred, '+', *args, ms=4, **kwargs)
    xmin = min([min(true), min(pred)])
    xmax = max([max(true), max(pred)])
    plot([xmin, xmax], [xmin, xmax], '--k')
    axis.set_aspect('equal', 'box')
    axis.set_xlabel('True')
    axis.set_ylabel('Predicted')

class RBFnet(object):
    """
    A radial basis function (RBF) machine learning network.

    A function f(x) approximates data y (possibly vector-valued) using a sum of
    radial basis functions g:

        y_i ~ f_i(x) = sum_j w_ij g(|x-c_j|/r)

    x is the input, also possibly a vector. c_j is the center coordinate of
    basis function j, while w_ij is the weight of this basis function for
    output variable i. The center coordinate is the same for all output
    variables y_i (for different centers per ouptut, use different RBFnet
    objects). r is the radius, which is the same for all basis functions. This
    is the inverse of the scaling parameter epsilon sometimes used. c_i, w_ij
    and r are fitted during training, and are later used for predictions.

    The methods train() and predict() are the two most important to start with.
    """

    def __init__(self):
        self.input_shift = 0
        self.input_scale = 1
        self.output_shift = 0
        self.output_scale = 1
        self.radius = 1

    def predict(self, input):
        """
        Make one or more predictions using the RBF net.

        Parameters
        ----------
        input: numpy.ndarray
            input[i,j] is prediction point i, independent variable j (x_j). For
            univariate input, the last index can be omitted.

        Returns
        -------
        numpy.ndarray where output[i,j] is the prediction of point i, dependent
        variable j (y_j). The last index is omitted, if the last index were
        omitted during training.
        """
        inp = self.normalize_input(input)

        # In diff[k,i,l], 
        # k is the index of the data point,
        # i is the index of the basis function,
        # l is the independent (input) variable index
        diff = inp[:,None,:] - self.centers[None,:,:]

        # distance[k,i] is then the euclidian distance between data point k and
        # basis function i
        distance = np.linalg.norm(diff, axis=2)

        rbf_matrix = self.rbf(distance/self.radius)
        output = rbf_matrix.dot(self.coeffs)

        output = self.denormalize_output(output)

        return output

    def train(self, input, output, num=10, radius=None, rbf=kernels.gaussian,
              random_state=None, keep_aspect=False, verbose=True,
              measure=metrics.rms_error, relative=False, normalize=True,
              method='powell', options=None, tol=1e-6):
        """
        Train the RBF net to learn the relation between input and output.

        Training consists of three steps:

            1. adapt_normalization()
            2. compute_centers()
            3. fit_weights() or fit_weights_and_radius() if radius is None

        This method runs all three, but they may alternatively be run
        separately.

        Parameters
        ----------
        input: numpy.ndarray
            input[i,j] is data point i, independent variable j (x_j). For
            univariate input the last index can be omitted.

        output: numpy.ndarray
            output[i,j] is data point i, dependent variable j (y_j). For
            univariate output, the last index can be omitted.

        num: integer
            Number of radial basis functions (learning capacity of network).

        radius: float
            The radius of the basis functions on normalized data. When
            adapt_normalization is used, it is the radius in terms of
            the standard deviation, and is usually near one. This is
            a hyperparameter that can be tuned for minimum error. If None,
            an algorithm will try to find the optimum radius with respect to
            the error measure given by the parameter 'measure'.

        rbf: function
            Which radial basis function to use. The function should take a
            single scalar argument, and is shifted according to the centers and
            scaled according to the radius internally.

        random_state: integer
            State of random number generator used to compute the centers. For
            exactly reproducible results this must be set manually. See
            sklearn.cluster.KMeans in Scikit-Learn.

        keep_aspect: bool
            Whether to scale all independent (input) and dependent (output)
            variables by the same factor during normalization, i.e., to keep
            the aspect ratio, or to scale them independently such that the
            standard deviation is one along each axis. It may make sense to
            keep the aspect ratio if the input variables have the same physical
            dimension.

        verbose: bool
            Print progress information

        measure: function
            The error measure used during optimization of the radius, with
            signature measure(true_values, predicted_values). This does not
            affect the fit of the weights, only the radius.

        relative: bool
            Use relative linear least squares, i.e., least squares modified
            to optimize the root-mean-square of relative residuals instead of
            absolute residuals.

        normalize: bool
            Whether or not to normalize

        See scipy.optimize.minimize for 'method', 'options' and 'tol'. These
        apply to the optimization of the radius.
        """

        if normalize:
            if verbose: print('Adapting normalization')
            self.adapt_normalization(input, output, keep_aspect)
        else:
            self.input_shift = 0
            self.input_scale = 1
            self.output_shift = 0
            self.output_scale = 1

        if verbose: print('Computing centers')
        self.compute_centers(input, num, random_state)

        if verbose: print('Fitting weights')
        if radius is None:
            self.fit_weights_and_radius(input, output, rbf, measure, relative,
                                        verbose, method=method,
                                        options=options, tol=tol)
        else:
            self.fit_weights(input, output, radius, rbf, relative)

    def adapt_normalization(self, input, output, keep_aspect=False):
        """
        Adapt normalization of network to data, such that the normalized data
        inside the network will have zero mean and a standard deviation of one.

        See RBFnet.train() for explanation of parameters.
        """
        self.input_shift = np.mean(input, axis=0)
        self.output_shift = np.mean(output, axis=0)

        if keep_aspect:
            self.input_scale = np.sqrt(np.mean(np.linalg.norm(input-self.input_shift, axis=1)**2))
            output_ = output.reshape(len(output), 1) # Make sure it is always 2D
            self.output_scale = np.sqrt(np.mean(np.linalg.norm(output_-self.output_shift, axis=1)**2))
        else:
            self.input_scale = np.std(input, axis=0)
            self.output_scale = np.std(output, axis=0)

    def compute_centers(self, input, num, random_state=None):
        """
        Compute the centers of the radial basis functions using K-means
        clustering.

        See RBFnet.train() for explanation of parameters.
        """
        inp = self.normalize_input(input)

        if np.iscomplexobj(inp):
            # KMeans does not support complex numbers.
            # Arrange imaginary axes as independent variables.
            n_points, n_indeps = inp.shape
            inp_ = np.zeros((n_points, 2*n_indeps), dtype=np.float64)
            inp_[:,:n_indeps] = np.real(inp)
            inp_[:,n_indeps:] = np.imag(inp)
            was_complex = True
        else:
            inp_ = inp
            was_complex = False

        clustering = KMeans(n_clusters=num, random_state=random_state).fit(inp_)

        centers = clustering.cluster_centers_

        if was_complex:
            centers = centers[:,:n_indeps]+1j*centers[:,n_indeps:]

        self.centers = centers

    def fit_weights(self, input, output, radius=1, rbf=kernels.gaussian, relative=False):
        """
        Fit the weights to the training data using linear least squares.

        See RBFnet.train() for explanation of parameters.
        """

        inp = self.normalize_input(input)
        outp = self.normalize_output(output)

        assert inp.shape[0]==outp.shape[0], \
            "different number of data points (length of axis 0) in input and output"

        assert inp.shape[1]==self.centers.shape[1], \
            "input has {} independent variables (length of axis 1) when fitting "\
            "weights, but had {} when computing centers"\
            .format(inp.shape[1], self.centers.shape[1])

        # In diff[k,i,l], 
        # k is the index of the data point,
        # i is the index of the basis function,
        # l is the independent (input) variable index
        diff = inp[:,None,:] - self.centers[None,:,:]

        # distance[k,i] is then the euclidian distance between data point k and
        # basis function i
        distance = np.linalg.norm(diff, axis=2)

        # The matrix is the same for each output. The lstsq function
        # automatically apply the least squares for each column using the same
        # matrix.
        rbf_matrix = rbf(distance/radius)

        # TBD: This matrix is generated the same way here and in predict().
        # Remove duplicate code.

        if relative:
            precond = (outp+self.output_shift/self.output_scale)**(-1)
            coeffs, _, _, _ = np.linalg.lstsq(precond[:,None]*rbf_matrix,
                                              precond*outp, rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(rbf_matrix, outp, rcond=None)

        self.coeffs = coeffs
        self.rbf = rbf
        self.radius = radius

    def fit_weights_and_radius(self, input, output, rbf=kernels.gaussian,
                               measure=metrics.rms_error, relative=False, verbose=True,
                               method='powell', options=None, tol=1e-6):
        """
        Similarly as the method fit_weights(), but this function uses an
        iterative minimizer to find the optimal radius.

        See RBFnet.train() for explanation of parameters.
        """

        def fun(radius):
            self.fit_weights(input, output, radius, rbf, relative)
            self.error = measure(output, self.predict(input))
            return self.error

        fmt = "  {:<5}  {:<20}  {:<20}"

        if verbose:
            print(fmt.format("it.", "radius", "error"))

        self.fcall = 0
        def callback(params):
            if verbose:
                self.fcall += 1
                print(fmt.format(self.fcall, params[0], self.error))

        radius_0 = 1
        res = minimize(fun, radius_0, tol=tol,
                       options=options,
                       callback=callback,
                       method=method)

    def save(self, fname):
        """
        Save trained model to npz-file.

        Parameters
        ----------
        fname: string
            filename
        """
        import dill as pickle
        np.savez_compressed(fname,
                            input_scale=self.input_scale,
                            input_shift=self.input_shift,
                            output_scale=self.output_scale,
                            output_shift=self.output_shift,
                            centers=self.centers,
                            coeffs=self.coeffs,
                            radius=self.radius,
                            rbf=pickle.dumps(self.rbf))
        # with open(fname, 'wb') as file:
        #     pickle.dump(self, file)

    def load(self, fname):
        """
        Load a previously trained model from npz-file.

        Parameters
        ----------
        fname: string
            filename
        """
        import dill as pickle
        data = np.load(fname, allow_pickle=False)
        self.input_scale = data['input_scale']
        self.input_shift = data['input_shift']
        self.output_scale = data['output_scale']
        self.output_shift = data['output_shift']
        self.centers = data['centers']
        self.coeffs = data['coeffs']
        self.radius = data['radius']
        self.rbf = pickle.loads(data['rbf'])
        # with open(fname, 'rb') as file:
        #     self = pickle.load(file)
        # return self

    def plot_centers(self, axis, indeps=(0,1), normalized=False,
                     radius='disk', **kwargs):
        """
        Plot the centers of the basis functions projected onto a plane spanned
        by the axes of two of the independent variables. The radius can also be
        indicated by a disk.

        Parameters
        ----------
        axis: matplotlib.axis
            Axis on which to plot the centers

        indeps: list or tuple with two elements
            The independent variables to use as x- and y-axes

        normalized: bool
            Whether to plot the centers on normalized axes or not

        radius: 'disk' (default), 'circle' or None
            Whether to indicate the radius with a disk, a circle, or not at all.

        Further accepts the same keyword arguments as matplotlib.axis.plot.
        """
        indeps = list(indeps)
        centers = self.centers

        theta = np.linspace(0, 2*np.pi, 60)

        # Prepending an index for points along the ellipse
        pts = np.zeros((len(theta), *centers.shape))
        pts[:] = centers

        # Although we strictly speaking only need two indeps in the pts array,
        # we keep all so we can use denormalize_input()
        pts[:,:,indeps[0]] += self.radius*np.cos(theta[:,None])
        pts[:,:,indeps[1]] += self.radius*np.sin(theta[:,None])

        if not normalized:
            pts = self.denormalize_input(pts)
            centers = self.denormalize_input(centers)

        centers = centers[:,indeps]

        x = pts[:,:,indeps[0]]
        y = pts[:,:,indeps[1]]

        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = ''

        if 'marker' not in kwargs:
            kwargs['marker'] = 'x'

        line, = axis.plot(*centers.T, **kwargs)

        if radius=='disk':
            axis.fill(x, y, alpha=0.2, color=line.get_color())
        elif radius=='circle':
            axis.plot(x, y, color=line.get_color())

    def eval_bases(self, input):
        """
        Evaluate each basis function separately for a sequence of input points.

        NB: This function is not yet adapted to multivariate output.

        Parameters
        ----------
        input: numpy.ndarray
            input[i,j] is input point i, independent variable j (x_j)

        Returns
        -------
        numpy.ndarray where output[i,j] is the basis function i, input point j
        """
        inp = self.normalize_input(input)

        output = np.zeros((len(self.centers), inp.shape[0]))
        for j in range(len(self.centers)):
            distance = np.linalg.norm(inp-self.centers[j,:], axis=1)
            output[j] = self.coeffs[j]*self.rbf(distance/self.radius)

        output = self.denormalize_output(output)

        return output

    def plot_bases(self, axis, input, *args, **kwargs):
        """
        Plot the basis functions for a sequence of input points.

        NB: This function is not yet adapted to multivariate output.

        Parameters
        ----------
        axis: matplotlib.axis
            Axis on which to plot the bases

        input: numpy.ndarray
            input[i,j] is input point i, independent variable j (x_j)

        Further accepts the same arguments as matplotlib.axis.plot.
        """
        bases = self.eval_bases(input)
        pred  = self.predict(input)
        axis.plot(input, pred, 'k', *args, **kwargs)
        kwargs.pop('label', None)
        for base in bases:
            axis.plot(input, base, '--k', *args, **kwargs)

    def normalize_input(self, input):
        """
        Normalize an array of values in the domain

        Parameters
        ----------
        input: numpy.array
            Array of values in the domain
        """
        norm_input = (np.array(input)-self.input_shift)/self.input_scale
        if len(norm_input.shape)==1:
            norm_input = norm_input.reshape(-1, 1)
        return norm_input

    def normalize_output(self, output):
        """
        Normalize an array of values in the codomain

        Parameters
        ----------
        input: numpy.array
            Array of values in the codomain
        """
        norm_output = (np.array(output)-self.output_shift)/self.output_scale
        return norm_output

    def denormalize_input(self, norm_input):
        """
        Denormalize an array of values in the domain

        Parameters
        ----------
        input: numpy.array
            Array of normalized values in the domain
        """
        input = np.array(norm_input)*self.input_scale + self.input_shift
        return input

    def denormalize_output(self, norm_output):
        """
        Denormalize an array of values in the codomain

        Parameters
        ----------
        input: numpy.array
            Array of normalized values in the codomain
        """
        output = np.array(norm_output)*self.output_scale + self.output_shift
        return output
