import numpy as np


def laplacian(A):
    """ Get graph Laplacian matrix

    Args:
        A (np.ndarray): Adjacency matrix

    Returns:
        (np.ndarray): Laplacian matrix
    """
    return np.diag(A.sum(1)) - A


# def barycenter_optimizer(A_init, lambdas, Ds, Ps):
#     N = A_init.shape[0]

#     def J(C):
#         C = np.reshape(C, (N, N))
#         obj = 0
#         for i in range(len(Ds)):
#             obj += lambdas[i] * (np.linalg.norm(C)**2 / C.shape[0]**2 + np.linalg.norm(Ds[i]) **
#                                  2 / Ds[i].shape[0]**2 - 2 * np.trace(C @ Ps[i] @ Ds[i] @ Ps[i].T))
#         return obj

#     def grad_C(C):
#         grad = 0
#         for i in range(len(Ds)):
#             grad += lambdas[i] * Ps[i] @ Ds[i].T @ Ps[i].T
#         return 2 * C / C.shape[0]**2 - 2 * grad

#     def Z(L):
#         e = np.ones([N, 1])
#         eet = e @ e.T / N
#         return np.linalg.inv(L + eet)

#     def vol(X):
#         return X.sum()

#     def grad_vol():
#         return np.ones([N, N])

#     def vol_Z(Z):
#         Z = np.reshape(Z, [N, N])
#         e = np.ones([N, 1])
#         eet = e @ e.T
#         L = np.linalg.inv(Z) - eet / N
#         np.fill_diagonal(L, 0)
#         return -L.sum()

#     def grad_vol_Z(Z):
#         Z = np.reshape(Z, [N, N])
#         Z_inv = np.linalg.inv(Z)
#         offdiag = np.ones([N, N])
#         # np.fill_diagonal(offdiag, 0)
#         return - Z_inv @ Z_inv * offdiag + offdiag

#     def calc_T(Z):
#         n = Z.shape[0]
#         T = np.zeros([n, n])
#         for i in range(n):
#             for j in range(n):
#                 T[i, j] = (Z[i, i] + Z[j, j] - Z[i, j] - Z[j, i]) if i != j else (Z[i, i] + Z[j, j]) / 2
#         return T

#     def calc_Z(T):
#         n = T.shape[0]
#         Z = np.zeros([n, n])
#         for i in range(n):
#             for j in range(n):
#                 Z[i, j] = 0.5 * (T[i, i] + T[j, j] - T[i, j]) if i != j else (T[i, i] + T[j, j]) / 2
#         return Z

#     def vol_T(T):
#         return vol_Z(calc_Z(T))

#     def vol_X(X):
#         return vol_Z()

#     # def T_star(T):
#     #     n = T.shape[0]
#     #     Z = np.zeros([n, n])
#     #     for i in range(n):
#     #         for j in range(n):
#     #             Z[i, j] = 0.5*(T[i, i] + T[j, j] - T[i, j]) if i!=j else (T[i, i]+T[j, j])/2
#     #     return Z

#     def T_star(X):
#         Z = np.zeros([N, N])
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     Z[i, j] = - X[i, j] - X[j, i]
#                 else:
#                     Z[i, j] = -X[i, i] + 2 * sum([X[i, k] for k in range(N)])
#         return Z

#     def CT(Z):
#         Z = np.reshape(Z, [N, N])
#         T = calc_T(Z)
#         e = np.ones([N, 1])
#         ct = vol_Z(Z) * (T * (e @ e.T - np.identity(N)))
#         np.fill_diagonal(ct, 0)
#         return ct

#     def J_Z(Z):
#         return J(CT(Z))

#     # def grad_Z(Z, X):
#     #     return vol_Z(Z)*T_star(X) + grad_vol_Z(Z)*CT(Z)/vol_Z(Z)

#     def grad_XZ(Z):
#         XZ = np.zeros([N, N, N, N])
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     XZ[i, j, i, j] = -1
#                     XZ[j, i, i, j] = -1
#                     XZ[i, i, i, j] = 1
#                     XZ[j, j, i, j] = 1
#         for i in range(N):
#             XZ[:, : i, i] *= 0
#         # wried observation: only the upper triangular of the last two dims are correct.
#         # symmetrize over last two dims.
#         for i in range(N):
#             for j in range(i + 1, N):
#                 XZ[:, :, j, i] = XZ[:, :, i, j]
#         return XZ

#     def vec(X):
#         return X.reshape([-1, 1])

#     def grad_CZ(Z):
#         gXZ = grad_XZ(Z)
#         for i in range(N):
#             gXZ[:, :, i, i] *= 0
#         grad_CZ = vol_Z(Z) * gXZ.reshape([N**2, N**2]) + vec(grad_vol_Z(Z)) @ vec(CT(Z)).T / vol_Z(Z)
#         return grad_CZ.reshape([N, N, N, N])

#     def grad_Z(Z):
#         return np.einsum('pq, ijpq -> ij', grad_C(CT(Z)), grad_CZ(Z))

#     def C(L):
#         e = np.ones([N, 1])
#         eet = e @ e.T / N
#         return np.linalg.inv(L + eet) - eet

#     def J_L(L):
#         L = np.reshape(L, (N, N))
#         e = np.ones([N, 1])
#         eet = e @ e.T / N
#         return J_Z(Z(L))

#     def grad_L(X, L):
#         return -Z(L) @ X @ Z(L)

#     def J_A(X):
#         X = np.reshape(X, (N, N))
#         return J_L(laplacian(X))

#     def A_star(X):
#         N = X.shape[0]
#         _ = np.zeros([N, N])
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     _[i, j] = -X[i, j] + 0.5 * X[i, i] + 0.5 * X[j, j]
#         return _

#     def obj(A):
#         return J_A(A)

#     def grad(A):
#         return A_star(grad_L(grad_Z(Z(laplacian(A))), laplacian(A)))

#     return obj, grad


# def barycenter_optimizer2(func_handler, N):
#     J, grad_C = func_handler()

#     def Z(L):
#         e = np.ones([N, 1])
#         eet = e @ e.T / N
#         return np.linalg.inv(L + eet)

#     def vol(X):
#         return X.sum()

#     def grad_vol():
#         return np.ones([N, N])

#     def vol_Z(Z):
#         Z = np.reshape(Z, [N, N])
#         e = np.ones([N, 1])
#         eet = e @ e.T
#         L = np.linalg.inv(Z) - eet / N
#         np.fill_diagonal(L, 0)
#         return -L.sum()

#     def grad_vol_Z(Z):
#         Z = np.reshape(Z, [N, N])
#         Z_inv = np.linalg.inv(Z)
#         offdiag = np.ones([N, N])
#         # np.fill_diagonal(offdiag, 0)
#         return - Z_inv @ Z_inv * offdiag + offdiag

#     def calc_T(Z):
#         n = Z.shape[0]
#         T = np.zeros([n, n])
#         for i in range(n):
#             for j in range(n):
#                 T[i, j] = (Z[i, i] + Z[j, j] - Z[i, j] - Z[j, i]) if i != j else (Z[i, i] + Z[j, j]) / 2
#         return T

#     def calc_Z(T):
#         n = T.shape[0]
#         Z = np.zeros([n, n])
#         for i in range(n):
#             for j in range(n):
#                 Z[i, j] = 0.5 * (T[i, i] + T[j, j] - T[i, j]) if i != j else (T[i, i] + T[j, j]) / 2
#         return Z

#     def vol_T(T):
#         return vol_Z(calc_Z(T))

#     def vol_X(X):
#         return vol_Z()

#     # def T_star(T):
#     #     n = T.shape[0]
#     #     Z = np.zeros([n, n])
#     #     for i in range(n):
#     #         for j in range(n):
#     #             Z[i, j] = 0.5*(T[i, i] + T[j, j] - T[i, j]) if i!=j else (T[i, i]+T[j, j])/2
#     #     return Z

#     def T_star(X):
#         Z = np.zeros([N, N])
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     Z[i, j] = - X[i, j] - X[j, i]
#                 else:
#                     Z[i, j] = -X[i, i] + 2 * sum([X[i, k] for k in range(N)])
#         return Z

#     def CT(Z):
#         Z = np.reshape(Z, [N, N])
#         T = calc_T(Z)
#         e = np.ones([N, 1])
#         ct = vol_Z(Z) * (T * (e @ e.T - np.identity(N)))
#         np.fill_diagonal(ct, 0)
#         return ct

#     def J_Z(Z):
#         return J(CT(Z))

#     # def grad_Z(Z, X):
#     #     return vol_Z(Z)*T_star(X) + grad_vol_Z(Z)*CT(Z)/vol_Z(Z)

#     def grad_XZ(Z):
#         XZ = np.zeros([N, N, N, N])
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     XZ[i, j, i, j] = -1
#                     XZ[j, i, i, j] = -1
#                     XZ[i, i, i, j] = 1
#                     XZ[j, j, i, j] = 1
#         for i in range(N):
#             XZ[:, : i, i] *= 0
#         # wried observation: only the upper triangular of the last two dims are correct.
#         # symmetrize over last two dims.
#         for i in range(N):
#             for j in range(i + 1, N):
#                 XZ[:, :, j, i] = XZ[:, :, i, j]
#         return XZ

#     def vec(X):
#         return X.reshape([-1, 1])

#     def grad_CZ(Z):
#         gXZ = grad_XZ(Z)
#         for i in range(N):
#             gXZ[:, :, i, i] *= 0
#         grad_CZ = vol_Z(Z) * gXZ.reshape([N**2, N**2]) + vec(grad_vol_Z(Z)) @ vec(CT(Z)).T / vol_Z(Z)
#         return grad_CZ.reshape([N, N, N, N])

#     def grad_Z(Z):
#         return np.einsum('pq, ijpq -> ij', grad_C(CT(Z)), grad_CZ(Z))

#     def C(L):
#         e = np.ones([N, 1])
#         eet = e @ e.T / N
#         return np.linalg.inv(L + eet) - eet

#     def J_L(L):
#         L = np.reshape(L, (N, N))
#         e = np.ones([N, 1])
#         eet = e @ e.T / N
#         return J_Z(Z(L))

#     def grad_L(X, L):
#         return -Z(L) @ X @ Z(L)

#     def J_X(X):
#         X = np.reshape(X, (N, N))
#         return J_L(laplacian(X))

#     def A_star(X):
#         N = X.shape[0]
#         _ = np.zeros([N, N])
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     _[i, j] = -X[i, j] + 0.5 * X[i, i] + 0.5 * X[j, j]
#         return _

#     def obj(A):
#         return J_X(A)

#     def grad(A):
#         return A_star(grad_L(grad_Z(Z(laplacian(A))), laplacian(A)))

#     return obj, grad


def barycenter_optimizer3(func_handler, N):
    bary_C, grad_C = func_handler()

    eet = np.ones((N, N))
    # e = np.ones((N, 1))
    # @PendingDeprecationWarning
    # def vol(X):
    #     """ vol(X) = 1 X 1.T """
    #     return X.sum()

    # @PendingDeprecationWarning
    # def grad_vol():
    #     r""" \grad vol(X) = 11.T """
    #     return np.ones([N, N])

    def L_pinv(A):
        """ pseudo inverse
        .. math::

        L_pinv = (L + 11.T / N)^{-1} - 11.T / N
        """
        N = A.shape[0]
        L = laplacian(A)
        # eet = np.ones([N, N])
        Lpinv = np.linalg.inv(L + eet / N) - eet / N
        return Lpinv

    def vol_Z(Z):
        """ calculate the vol(A) in terms of L -> vol(L^{-1}) """
        Z = np.reshape(Z, [N, N])
        # e = np.ones([N, 1])
        # eet = e @ e.T
        L = np.linalg.pinv(Z) - eet / N
        # np.fill_diagonal(L, 0)
        # return -L.sum()
        # REVIEW: simplied
        return np.trace(L)

    def grad_vol_Z(Z):
        """ grad in vol(L^{-1}) """
        Z = np.reshape(Z, [N, N])
        Z_inv = np.linalg.pinv(Z)
        # eet = np.ones([N, N])
        # np.fill_diagonal(eet, 0)
        # REVIEW: simplied
        # return - Z_inv @ Z_inv * eet + eet
        return - Z_inv @ Z_inv

    def calc_T(Z):
        """ Given Z, cal T """
        n = Z.shape[0]
        T = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                T[i, j] = (Z[i, i] + Z[j, j] - Z[i, j] - Z[j, i]) if i != j else (Z[i, i] + Z[j, j]) / 2
        return T

    @DeprecationWarning
    def calc_Z(T):
        """ Given T, calculate Z """
        n = T.shape[0]
        Z = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                Z[i, j] = 0.5 * (T[i, i] + T[j, j] - T[i, j]) if i != j else (T[i, i] + T[j, j]) / 2
        return Z

    def grad_T_Z(Z):
        N = Z.shape[0]
        grad_ = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        if i == j and i == k:
                            grad_[i, j, k, k] = 1
                        if i != j and i == k:
                            grad_[i, j, k, k] = 1
                        if i != j and j == l:
                            grad_[i, j, l, l] = 1
                        if i != j and i == k and j == l:
                            grad_[i, j, k, l] = -2
        np.fill_diagonal(grad_, 0)
        return grad_

    def grad_D_A(A):
        N = A.shape[0]
        grad_ = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        if i == j and k == i:
                            grad_[i, j, k, l] = 1
        return grad_

    def grad_L_A(A):
        """
        .. math::
        \\partial L / \\parital A =  \\partial D / \\parital A - \\partial A / \\partial A
        \\partial A / \\parital A =
        """
        N = A.shape[0]
        grad_ = grad_D_A(A) - np.kron(np.eye(N), np.eye(N)).reshape((N, N, N, N))
        return grad_
        # def vol_T(T):
        #     return vol_Z(calc_Z(T))

        # def vol_X(X):
        #     return vol_Z()

        # def T_star(T):
        #     n = T.shape[0]
        #     Z = np.zeros([n, n])
        #     for i in range(n):
        #         for j in range(n):
        #             Z[i, j] = 0.5*(T[i, i] + T[j, j] - T[i, j]) if i!=j else (T[i, i]+T[j, j])/2
        #     return Z

        # def T_star(X):
        #     Z = np.zeros([N, N])
        #     for i in range(N):
        #         for j in range(N):
        #             if i != j:
        #                 Z[i, j] = - X[i, j] - X[j, i]
        #             else:
        #                 Z[i, j] = -X[i, i] + 2 * sum([X[i, k] for k in range(N)])
        #     return Z

    def CT(Z):
        """ CT(Z) = vol(Z) (T 。(11.T - I))  """
        Z = np.reshape(Z, [N, N])
        T = calc_T(Z)
        e = np.ones([N, 1])
        ct = vol_Z(Z) * (T * (e @ e.T - np.identity(N)))
        np.fill_diagonal(ct, 0)
        return ct

    # def grad_Z(Z, X):
    #     return vol_Z(Z)*T_star(X) + grad_vol_Z(Z)*CT(Z)/vol_Z(Z)

    def grad_XZ(Z):
        XZ = np.zeros([N, N, N, N])
        for i in range(N):
            for j in range(N):
                if i != j:
                    XZ[i, j, i, j] = -1
                    XZ[j, i, i, j] = -1
                    XZ[i, i, i, j] = 1
                    XZ[j, j, i, j] = 1
        for i in range(N):
            XZ[:, : i, i] *= 0
        # wried observation: only the upper triangular of the last two dims are correct.
        # symmetrize over last two dims.
        for i in range(N):
            for j in range(i + 1, N):
                XZ[:, :, j, i] = XZ[:, :, i, j]
        return XZ

    def vec(X):
        return X.reshape([-1, 1])

    def grad_CZ(Z):
        gXZ = grad_XZ(Z)
        for i in range(N):
            gXZ[:, :, i, i] *= 0
        grad_CZ = vol_Z(Z) * gXZ.reshape([N**2, N**2]) + vec(grad_vol_Z(Z)) @ vec(CT(Z)).T / vol_Z(Z)
        return grad_CZ.reshape([N, N, N, N])

    def grad_Z(Z):
        return np.einsum('pq, ijpq -> ij', grad_C(CT(Z)), grad_CZ(Z))

    def C(L):
        # e = np.ones([N, 1])
        # eet = e @ e.T / N
        return np.linalg.pinv(L + eet) - eet

    def Z(L):
        """ Z = (L + 11.T)^{-1} """
        # e = np.ones([N, 1])
        # eet = e @ e.T / N
        return np.linalg.pinv(L + eet)

    def J_Z(Z):
        """ calculate the barycenter loss """
        return bary_C(CT(Z))

    def J_L(L):
        L = np.reshape(L, (N, N))
        # e = np.ones([N, 1])
        # eet = e @ e.T / N
        return J_Z(Z(L))

    def grad_L(X, L):
        return -Z(L) @ X @ Z(L)

    def J_A(A):
        A = np.reshape(A, (N, N))
        return J_L(laplacian(A))

    def A_star(X):
        N = X.shape[0]
        _ = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                if i != j:
                    _[i, j] = -X[i, j] + 0.5 * X[i, i] + 0.5 * X[j, j]
        return _

    def obj(A):
        return J_A(A)

    def grad(A):
        L = laplacian(A)
        return A_star(grad_L(grad_Z(Z(L)), L))

    return obj, grad
