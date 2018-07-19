import sys
import os
join = os.path.join

import numpy as np
import pandas as pd
import datetime

from bayes_opt import BayesianOptimization

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

sys.path.append("..")
from make_candidates import UnrollingHelices
from merge import LengthMerge
import score_track
from extension import extend

from common import calc_steps

def run():
    print(datetime.datetime.now(), sys.argv[0], " begin")
    
    path_to_input = os.path.join(path_to_trackml, "train_1")
    path_to_out = "out_" + sys.argv[0].split(".")[0]

    niter = 150
    num_step = 5

    def Fun4BO(eps1, eps2, eps3, eps4, eps5):
        eps0s   = [eps1,   eps2,   eps3,   eps4,   eps5]
        th_lens = [13,     10,     6,      3,      3]
        num_exts= [0,      0,      0,      0,      0]
        score = calc_steps(niter, eps0s, th_lens, num_exts, path_to_out)
        return score

    opt = BayesianOptimization(Fun4BO,
                               {"eps1": (0.0001, 0.01),
                                "eps2": (0.001,  0.01),
                                "eps3": (0.001,  0.01),
                                "eps4": (0.001,  0.01),
                                "eps5": (0.01,   0.1)},
                               verbose = True)
    opt.maximize(init_points = 3,
                 n_iter = 20,
                 acq = "ucb",
                 kappa = 2.576)

    print(datetime.datetime.now(), sys.argv[0], " end")
            
if __name__=="__main__":
    run()
