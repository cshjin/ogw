""" Implementation of ogw and its associated utilities. """
import numpy as np
from scipy.linalg import svd

from ogw.gromov_prox import linear_solver, projection_matrix, quad_solver
from ogw.spg import SPG, default_options
from ogw.utils import padding_v2, squarify_v2
from scipy.linalg import eigvalsh

norm = np.linalg.norm


def Qcal_lb(C, D, return_matrix=False, **kwargs):
    """ Lower bound of Q function, by jointly optimize linear and quadratic term in trace.

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q matrix is not semi-orthogonal.
    """
    m, n = C.shape[0], D.shape[0]

    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()
    q_dim = max(m, n) - 1

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding_v2(C_hat, D_hat)

    E_hat = 2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
    _E_hat = squarify_v2(E_hat)

    def obj_func(Q):
        # obj = tr(_C_hat Q _D_hat Q.T) + tr(E_hat Q.T)
        Q = Q.reshape((q_dim, -1))
        fval = 0
        quad_ = np.trace(_C_hat @ Q @ _D_hat @ Q.T)
        lin_ = np.trace(_E_hat @ Q.T)

        fval = -quad_ - lin_

        g_quad_ = 2 * _C_hat @ Q @ _D_hat
        g_lin_ = _E_hat
        grad = -g_quad_ - g_lin_
        return fval, grad.flatten()

    def proj_func(Q):
        Q = Q.reshape((q_dim, -1))
        # DEBUG: SVD did not converge in the dataset IMDB-BINARY
        try:
            u, s, vh = svd(Q)
        except BaseException:
            u, s, vh = svd(Q + 1e-7)
        return (u @ vh).flatten()

    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
    spg_options.testOpt = False
    spg_options.verbose = 0 if "verbose" not in kwargs else kwargs['verbose']

    # NOTE: init of Q matters
    if "Q_init" not in kwargs:
        Q_init = np.random.randn(q_dim, q_dim).flatten()
        # REVIEW: could stuck at local min
        # Q_init = np.identity(q_dim)
        # Q_init = np.zeros((q_dim, q_dim))
    else:
        Q_init = kwargs['Q_init']
    res = SPG(obj_func, proj_func, Q_init, options=spg_options)
    q_opt = res[0].reshape((q_dim, - 1))
    q_fval = - res[1]

    Q = q_opt[:m - 1, :n - 1]

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_array_almost_equal(np.identity(m - 1), Q @ Q.T)
    else:
        np.testing.assert_array_almost_equal(np.identity(n - 1), Q.T @ Q)

    P = 1 / mn_sqrt * em @ en.T + U @ Q @ V.T

    fval = sC * sD / mn + q_fval
    if return_matrix:
        return fval, P
    else:
        return fval


def Qcal_ub(C, D, return_matrix=False, **kwargs):
    """ Upper bound of Q function, by optimizing Q1, Q2 separately.

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

        \\max_{Q_1} tr(C_hat Q_1 D_hat Q_1^\top) + \\max_{Q_2} tr(E_hat Q_2^\top)
        s.t. Q_1, Q_2 \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q1 matrix in semi-orthogonal domain.
        np.ndarray, optional: Q2 matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q1 and Q2 matrices are not semi-orthogonal.
    """
    # assert C.shape[0] >= D.shape[0]
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding_v2(C_hat, D_hat)

    E_hat = 2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
    _E_hat = squarify_v2(E_hat)
    q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
    l_fval, Q2 = linear_solver(_E_hat, domain="O", return_matrix=True)

    if "verbose" in kwargs:
        print("from ogw_lb", q_fval, l_fval)

    fval = sC * sD / mn + q_fval + l_fval
    Q1, Q2 = Q1[:m - 1, :n - 1], Q2[:m - 1, :n - 1]

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_almost_equal(np.identity(m - 1), Q1 @ Q1.T)
        np.testing.assert_almost_equal(np.identity(m - 1), Q2 @ Q2.T)
    else:
        np.testing.assert_almost_equal(np.identity(n - 1), Q1.T @ Q1)
        np.testing.assert_almost_equal(np.identity(n - 1), Q2.T @ Q2)

    if return_matrix:
        return fval, Q1, Q2
    else:
        return fval


def Qcal_ub_ub(C, D, return_matrix=False, **kwargs):
    """ Upper bound of Qcal_ub function, by optimizing Q1 only (ignore the linear term in trace).

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

        \\max_{Q_1} tr(C_hat Q_1 D_hat Q_1^\top)
        s.t. Q_1 \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q1 matrix in semi-orthogonal domain.
        np.ndarray, optional: Q2 matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q1 and Q2 matrices are not semi-orthogonal.
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding_v2(C_hat, D_hat)

    # NOTE: remove Q2
    # E_hat = 2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
    # _E_hat = squarify_v2(E_hat)
    q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
    # l_fval, Q2 = linear_solver(_E_hat, domain="O", return_matrix=True)

    if kwargs.get("verbose"):
        print("from Qcal_ub_ub", q_fval)

    fval = sC * sD / mn + q_fval
    Q1 = Q1[:m - 1, :n - 1]

    # Recover the P matrix from Q1. This is similar to ogw_lb.
    P = 1 / mn_sqrt * em @ en.T + U @ Q1 @ V.T

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_almost_equal(np.identity(m - 1), Q1 @ Q1.T)
    else:
        np.testing.assert_almost_equal(np.identity(n - 1), Q1.T @ Q1)

    if return_matrix:
        return fval, P
    else:
        return fval


def ogw_ub(C, D, return_matrix=False, **kwargs):
    """ Upper bound of ogw by soling `Qcal_lb`.

    .. math: :
        ogw_{ub} = ||C||_F^2 / m^2 + ||D||_F^2 / n^2
                - 2 / mn * \\max_{P \\in \\Ocal \\cap \\Ecal} tr(C P D P^\top)

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix(bool, optional): Return P matrix if `True`. Defaults to False.

    Returns:
        float: ogw distance
        np.ndarray, optional: P matrix

    See also:
        `Qcal_lb`
    """
    m, n = C.shape[0], D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    emn = np.ones((m, n))

    C_norm = norm(C)
    D_norm = norm(D)
    U = projection_matrix(m)
    V = projection_matrix(n)

    # recal Q_init from P_init
    if kwargs.get("P_init") is not None:
        kwargs['Q_init'] = squarify_v2(U.T @ (kwargs['P_init'] - 1 / mn_sqrt * emn) @ V)

    quad_val, P = Qcal_lb(C, D, return_matrix=True, **kwargs)

    fval = C_norm**2 / m**2 + D_norm**2 / n**2 - 2 * quad_val / mn

    if return_matrix:
        return fval, P
    else:
        return fval


def ogw_o(C, D, return_matrix=False, **kwargs):
    """ Optimize over the orthogonal domain direct.

    .. math: :
        ogw_{o} = ||C||_F^2 / m^2 + ||D||_F^2 / n^2
                - 2 / mn * \\max_{P \\in \\Ocal} tr(C P D P^\top)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim(m, m).
        D (np.ndarray): Geodesic distance in source domain with dim(n, n).
        return_matrix (bool, optional): Return the optimal P matrix if `True`. Defaults to False.

    Returns:
        float: ogw distance
        np.ndarray, optional: P matrix

    See Also:
    ogw.ogw_ub
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    C_norm = norm(C)
    D_norm = norm(D)

    C_, D_ = padding_v2(C, D)
    fval_Qcal, P = quad_solver(C_, D_, return_matrix=True)
    ogw_val = C_norm**2 / m**2 + D_norm**2 / n**2 - 2 / mn * fval_Qcal
    if return_matrix:
        return ogw_val, P[:m, :n]
    else:
        return ogw_val


def ogw_lb_v2(C, D, V=None):
    """ Faster implementation of computing OGW_lb for graph with same orders.

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(n, n).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        V(np.ndarray, optional): Projection matrix (n-1, n-1).  Not used.
    Returns:
        float: OGW_lb distance.
    """

    n = C.shape[0]
    C_hat = VXV(C)
    D_hat = VXV(D)
    Cone = np.sum(C, axis=1)
    Done = np.sum(D, axis=1)

    evals_C = eigvalsh(C_hat)
    evals_D = eigvalsh(D_hat)

    fval_q1 = np.dot(evals_C, evals_D)

    VCone = Vx(Cone)
    VDone = Vx(Done)
    fval_q2 = norm(VCone) * norm(VDone) * 2 / n
    sC = Cone.sum()
    sD = Done.sum()
    fval_c = sC * sD / n**2

    fval_qcal = fval_c + fval_q1 + fval_q2
    norms = norm(evals_C)**2 \
        + norm(evals_D)**2 \
        + (sC**2 + sD**2) / n**2 \
        + norm(VCone)**2 / n * 2 + norm(VDone)**2 / n * 2
    fval = (norms - 2 * fval_qcal) / n**2

    return fval


def VXV(X):
    """ Efficient implementation for computing `V^T @ X @ V`

    Args:
        X (np.ndarray): Input matrix with dim (n, n).

    Returns:
        np.ndarray: Projected matrix with dim (n-1, n-1).
    """
    n = X.shape[0]
    x = -1 / (n + np.sqrt(n))
    y = -1 / np.sqrt(n)
    a = X[0, 0]
    b = X[1:, 0]
    B = X[1:, 1:]
    c = y * b + x * np.sum(B, axis=1)
    return a * y**2 + 2 * x * y * np.sum(b) + x**2 * np.sum(B) + B + c + c.reshape(n - 1, 1)


def Vx(c):
    """ Efficient implementation for computing `V^T @ x`

    Args:
        c (np.array): Array with dim (n, ).

    Returns:
        np.array: Projected array with dim (n-1, ).
    """
    n = len(c)
    x = -1 / (n + np.sqrt(n))
    y = -1 / np.sqrt(n)
    return c[1:] + ((y - x) * c[0] + x * sum(c))


def ogw_lb(C, D, return_matrix=False, **kwargs):
    """ Lower bound of ogw by solving `Qcal_ub`.

    .. math: :
        ogw = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * \\max_{Q1, Q2} \\Qcal(Q1, Q2)

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix(bool, optional): Return P matrix if `True`. Defaults to False.

    Returns:
        float: ogw distance
        np.ndarray, optional: P matrix

    See also:
        `Qcal_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n

    C_norm = norm(C)
    D_norm = norm(D)

    qfunc_val, Q1, Q2 = Qcal_ub(C, D, return_matrix=True, **kwargs)
    ogw_val = C_norm ** 2 / m**2 + D_norm**2 / n**2 - 2 / mn * qfunc_val
    if return_matrix:
        return ogw_val, Q1, Q2
    else:
        return ogw_val


def ogw_lb_lb(C, D, return_matrix=False, **kwargs):
    """ Lower bound of ogw by solving `Qcal_ub_ub`.

    .. math: :
        ogw = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * \\max_{Q1, Q2} \\Qcal(Q1, Q2)

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix(bool, optional): Return P matrix if `True`. Defaults to False.

    Returns:
        float: ogw distance
        np.ndarray, optional: P matrix

    See also:
        `Qcal_ub_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n

    C_norm = norm(C)
    D_norm = norm(D)

    qfunc_val, P = Qcal_ub_ub(C, D, return_matrix=True, **kwargs)
    ogw_val = C_norm ** 2 / m**2 + D_norm**2 / n**2 - 2 / mn * qfunc_val
    if return_matrix:
        return ogw_val, P
    else:
        return ogw_val


def eval_ogw_ub(C, D, P):
    """ Evaluate the ogw given C, D, and optimzed P.

    .. math::
        ogw = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * tr (C P D P^\top)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        P (np.ndarray): Optimized P matrix with dim (m, n).

    Returns:
        float: ogw function value
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    C_norm = norm(C)
    D_norm = norm(D)
    ogw_fval = C_norm**2 / m**2 + D_norm**2 / n**2 - 2 / mn * np.trace(C @ P @ D @ P.T)
    return ogw_fval


def eval_ogw_lb(C, D, Q1, Q2):
    """ Evaluate the ogw given C, D, and optimzed P.

    .. math::
        ogw = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * Qcal(Q1, Q2)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        Q1 (np.ndarray): Optimized Q1 matrix with dim (m-1, n-1).
        Q2 (np.ndarray): Optimized Q2 matrix with dim (m-1, n-1).

    Returns:
        float: ogw function value

    See Also:
        `Qcal_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = mn ** 0.5
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)
    C_norm = norm(C)
    D_norm = norm(D)

    _const = 1 / mn * sC * sD
    _linear = np.trace(V.T @ D @ en @ em.T @ C @ U @ Q2)
    _quad = np.trace(U.T @ C @ U @ Q1 @ V.T @ D @ V @ Q1.T)
    ogw_fval = C_norm ** 2 / m**2 + D_norm ** 2 / n**2 \
        - 2 / mn * (_const + 2 / mn_sqrt * _linear + _quad)

    return ogw_fval
