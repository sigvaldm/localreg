localreg
=========

.. image:: https://travis-ci.com/sigvaldm/localreg.svg?branch=master
    :target: https://travis-ci.com/sigvaldm/localreg

.. image:: https://coveralls.io/repos/github/sigvaldm/localreg/badge.svg?branch=master
    :target: https://coveralls.io/github/sigvaldm/localreg?branch=master

.. image:: https://img.shields.io/pypi/pyversions/localreg.svg
    :target: https://pypi.org/project/localreg

Smoothing of noisy data series through *local polynomial regression* (including LOESS/LOWESS).

Installation
------------
Install from PyPI using ``pip`` (preferred method)::

    pip install localreg

Or download the GitHub repository https://github.com/sigvaldm/localreg.git and run::

    python setup.py install

Basic Usage
-----------

.. include:: examples/basic.py
.. image:: examples/basic.png



.. [Hastie] T. Hastie, R. Tibshirani and J. Friedman *The Elements of Statistical Learing -- Data Mining, Inference, and Prediction*, Second Edition, Springer, 2017.
.. [Cleveland] W. Cleveland *Robust Locally Weighted Regression and Smoothing Scatterplots*, Journal of the Americal Statistical Associations, 74, 1979.
