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
from models import UnrollingHelicesA1Z1Z2, MultiStageByLen

from common import calc_steps

def run():
    print(datetime.datetime.now(), sys.argv[0], " begin")

    model = UnrollingHelicesA1Z1Z2()
    path_to_input = os.path.join(path_to_trackml, "train_1")

    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                       skip=0, nevents=1):
        print("len(hits): ", len(hits))
        labels = model.fit_predict(hits)

        
    
    path_to_out = "out_" + sys.argv[0].split(".")[0]

    niter = 150
    num_step = 5

    eps0s   = [0.0010, 0.0035, 0.0050, 0.0080, 0.0400]
    th_lens = [13,     10,     6,      3,      3]
    num_exts= [0,      0,      0,      0,      0]

    calc_steps(niter, eps0s, th_lens, num_exts, path_to_out)

    print(datetime.datetime.now(), sys.argv[0], " end")
            
if __name__=="__main__":
    run()
