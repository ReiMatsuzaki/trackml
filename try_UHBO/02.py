import sys
import os
join = os.path.join

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

sys.path.append("..")
import models
from utils import create_one_event_submission



def run(f):
    model = models.UnrollingHelicesBayessianOpt()
    path_to_input = os.path.join(path_to_trackml, "train_1")
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              skip=0, nevents=1):

        def Fun4BO(w1, w2, w3, niter):
            labels = model.predict(hits, w1, w2, w3, niter)
            one_submission = create_one_event_submission(event_id, hits, labels)
            score = score_event(truth, one_submission)
            return score

        print("Bayesian Optimization")
        opt = BayesianOptimization(Fun4BO,
                                   {"w1": (0.9, 1.2),
                                    "w2": (0.3, 0.7),
                                    "w3": (0.1, 0.4),
                                    "niter": (140, 190)},  #(140, 190)
                                   verbose = True)
        opt.maximize(init_points = 3,
                     n_iter = 20,
                     acq = "ucb",
                     kappa = 2.576)
        f.write(str(opt.res["max"]))
        
        

if __name__=="__main__":
    with open(sys.argv[0].split(".")[0]+".log", "w") as f:
        run(f)
