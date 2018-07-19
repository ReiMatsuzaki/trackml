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
    th_len_1 = 13
    th_len_N = 3

    eps0s   = list(np.logspace(np.log10(0.001), np.log10(0.008), num_step-1)) + [0.0400]
    th_lens = [int(x) for x in np.linspace(th_len_1, th_len_N, num_step-1)]
    th_lens.append(th_lens[-1])
    num_exts= [0 for _ in range(num_step)]

    calc_steps(niter, eps0s, th_lens, num_exts, path_to_out)

    print(datetime.datetime.now(), sys.argv[0], " end")
    
if __name__=="__main__":
    run()
