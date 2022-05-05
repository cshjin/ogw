import numpy as np
from ot.gromov import gromov_wasserstein
from scipy.optimize import minimize

from ogw.graph import find_thresh, sp_to_adjacency
from ogw.gw_bary import grad_gw_C, gromov_wasserstein_v2, optim_C_gw, optim_C_gw_v2
from ogw.ogw_dist import ogw_lb, ogw_ub
from ogw.ogw_bary import (grad_ogw_lb_C, grad_ogw_ub_C, optim_C_ogw_lb,
                          optim_C_ogw_lb_lb, optim_C_ogw_lb_lb_v2,
                          optim_C_ogw_lb_v2, optim_C_ogw_ub,
                          optim_C_ogw_ub_v2)
from scipy.spatial.distance import cdist


class MethodError(Exception):
    pass


class Barycenter(object):
    def __init__(self, topo_metric="gw", dist_func="sp") -> None:
        super().__init__()
        self.topo_metric = topo_metric
        self.dist_func = dist_func

    def optim_C(self, N, Ds, ps, p, lambdas, method="BFGS", **kwargs):
        """ Optimize barycenter regardless of dist_func """
        assert len(Ds) == len(ps) and len(Ds) == len(lambdas)
        assert p.shape[0] == N

        if method == "BFGS":
            if self.topo_metric == "gw":
                res = optim_C_gw(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "ogw_ub":
                res = optim_C_ogw_ub(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "ogw_lb":
                res = optim_C_ogw_lb(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "ogw_lb_lb":
                res = optim_C_ogw_lb_lb(N, Ds, ps, p, lambdas, **kwargs)

        elif method == "closed-form":
            if self.topo_metric == "gw":
                res = optim_C_gw_v2(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "ogw_ub":
                res = optim_C_ogw_ub_v2(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "ogw_lb":
                res = optim_C_ogw_lb_v2(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "ogw_lb_lb":
                res = optim_C_ogw_lb_lb_v2(N, Ds, ps, p, lambdas, **kwargs)
        else:
            raise MethodError

        if isinstance(res, tuple):
            self.C = res[0]
        else:
            self.C = res
        return res

    def optim_A_from_C(self, C):
        """ Solve A from barycenter using heuristic method """
        if self.dist_func == "sp":
            _up_bound = find_thresh(C, sup=C.max(), step=100, metric=self.dist_func)[0]
            A = sp_to_adjacency(C, threshinf=0, threshsup=_up_bound)
            return A
