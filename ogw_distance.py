import argparse
import logging
import os
import os.path as osp
import pickle

import networkx as nx
import numpy as np
from joblib.parallel import Parallel, delayed
from ot.gromov import gromov_wasserstein
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from yaml import parse
from time import time

from ogw.gw_lb import flb, slb, tlb
from ogw.ogw_dist import ogw_lb, ogw_o, ogw_ub
from ogw.utils import load_pyg_data

logging.basicConfig(format='%(asctime)s - %(message)s ', level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="MUTAG", type=str, help="dataset")
    args = vars(parser.parse_args())

    dsname = args["dataset"]

    Gs, ys = load_pyg_data(dsname)
    Cs = [np.array(nx.floyd_warshall_numpy(G)) for G in Gs]
    Ns = [C.shape[0] for C in Cs]
    ps = [np.ones(n) / n for n in Ns]
    N = len(Gs)

    ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "TUDataset")
    SAVED_PATH = osp.join(ROOT, dsname, "saved")
    if not osp.isdir(SAVED_PATH):
        logging.info("creating folder")
        os.makedirs(SAVED_PATH)


    def calc_D_GW(i, j, D):
        T, gw_log = gromov_wasserstein(Cs[i], Cs[j], ps[i], ps[j], loss_fun="square_loss", log=True)
        D[i, j] = gw_log['gw_dist']

    for _ in range(10):
        path_to_file = osp.join(SAVED_PATH, "D_GW.pkl")
        if osp.exists(path_to_file) and False:
            D_GW = pickle.load(open(path_to_file, "rb"))
        else:
            tic = time()
            fn_mm = osp.join(ROOT, dsname, "D_GW")
            D_GW = np.memmap(fn_mm, mode="w+", shape=(N, N), dtype=float)

            logging.info(f"calcualte GW")
            Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(calc_D_GW)(i, j, D_GW) for i in range(N) for j in range(i + 1, N))
            D_GW += D_GW.T

            pickle.dump(D_GW, open(osp.join(SAVED_PATH, "D_GW.pkl"), "wb"))
            toc = time()
            logging.info(f"calculate GW done: time {toc-tic:.4f} s")
