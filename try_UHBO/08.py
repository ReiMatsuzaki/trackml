# copied from 06.py

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

def run(filename):
    model = models.UnrollingHelicesRt2(
        dbscan_features=["sina1", "cosa1", "z1", "z2", "x_rt", "y_rt"],
        dbscan_weight  =[1.0,     1.0,     0.75, 0.2,  0.05,   0.05])
    model.niter = 150
    nevents = 10
    niter_bo = 40
    path_to_input = os.path.join(path_to_trackml, "train_1")

    event_id_list = []
    hits_list = []
    truth_list = []
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              nevents=nevents):
        event_id_list.append(event_id)
        hits_list.append(hits)
        truth_list.append(truth)

    def Fun4BO(w_a1, w_z1, w_z2, w_xy_rt, c_rt1, c_rt2):
        model.dbscan_weight[0] = w_a1
        model.dbscan_weight[1] = w_a1
        model.dbscan_weight[2] = w_z1
        model.dbscan_weight[3] = w_z2
        model.dbscan_weight[4] = w_xy_rt
        model.dbscan_weight[5] = w_xy_rt
        model.coef_rt1 = c_rt1
        model.coef_rt2 = c_rt2
        score_list = []
        for (hits, truth) in zip(hits_list, truth_list):
            labels = model.predict(hits)
            one_submission = create_one_event_submission(event_id, hits, labels)
            score = score_event(truth, one_submission)
            score_list.append(score)
        return np.sum(score_list)/len(score_list)

    print("Bayesian Optimization")
    opt = BayesianOptimization(Fun4BO,
                               {"w_a1"    : (0.9, 1.2),
                                "w_z1"    : (0.3, 0.7),
                                "w_z2"    : (0.1, 0.4),
                                "w_xy_rt" : (0.01, 0.2),
                                "c_rt1"   : (0.5, 1.5),
                                "c_rt2"   : (0.1, 5.0)
                               },
                               verbose = True)
    opt.maximize(init_points = 3,
                 n_iter = niter_bo, #
                 acq = "ucb",
                 kappa = 2.576)

    # [string]
    labels = opt.res["max"]["max_params"].keys()
    # [dict(string, [float])]
    params = opt.res["all"]["params"]
    len_params = len(params)
    
    data_dic = {}
    
    for label in labels:
        val = [opt.res["max"]["max_params"][label]]
        for i in range(len_params):
            val.append(params[i][label])
            data_dic[label] = val
    data_dic["value"] = [opt.res["max"]["max_val"]] + opt.res["all"]["values"]
    data_dic["label"] = ["max"] + [str(x) for x in range(len_params)]
    df = pd.DataFrame(data_dic)
    df.to_csv(filename, index=None)

if __name__=="__main__":
    run(sys.argv[0].split(".")[0]+".log")
