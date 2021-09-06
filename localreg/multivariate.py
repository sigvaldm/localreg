import numpy as np
import itertools as it

# x = np.array([[2],[3],[4]])
# x = np.array([[1,2],[1,2],[1,2]])
x = np.array([[1,2,3],[1,2,3],[1,2,3]])

degree = 4
n_indeps = x.shape[1]

# Bases for univariate polynomials (1, x, x^2, ...) represented as exponents
univariate_bases = np.arange(degree+1)

# Multivariate bases (1, x, y, x*y, ...) are represented as tuples of exponents.
multivariate_bases = it.product(*it.repeat(univariate_bases, n_indeps))
multivariate_bases = list(filter(lambda a: sum(a)<=degree, multivariate_bases))
# multivariate_bases = sorted(multivariate_bases, key=sum) # not really necessary

X = np.ones((x.shape[0], len(multivariate_bases)))

B = np.array(multivariate_bases)

for i in range(len(B)):

    # Un-optimized:
    # for j in range(x.shape[1]):
    #     X[:,i] *= x[:,j]**B[i,j]

    # Optimized away for-loop:
    X[:,i] = np.product(x[:,:]**B[i,:], axis=1)

print('GT')
print(X)
print(X.shape)
