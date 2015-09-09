import kernels
import numpy as np
from scipy.linalg import solve_banded

def scipy_solve_banded(a, b, c, rhs):
    '''
    Solve the tridiagonal system described
    by a, b, c, and rhs.
    a: lower off-diagonal array (first element ignored)
    b: diagonal array
    c: upper off-diagonal array (last element ignored)
    rhs: right hand side of the system
    '''
    l_and_u = (1, 1)
    ab = np.vstack([np.append(0, c[:-1]),
                    b,
                    np.append(a[1:], 0)])
    x = solve_banded(l_and_u, ab, rhs)
    return x

a = np.random.rand(5)
b = np.random.rand(5)
c = np.random.rand(5)
d = np.random.rand(10)
x = kernels.solve_many_small_systems(a, b, c, d, 2, 5)

print np.append(scipy_solve_banded(a, b, c, d[:5]), scipy_solve_banded(a, b, c, d[5:10]))
print x
