from bayes_opt import BayesianOptimization
import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
from tqdm import tqdm
import datetime

import numpy as np
tr = np.transpose
import pandas as pd

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

sys.path.append("../..")
import models

def run(filename):
    print("script begin", datetime.datetime.now())
    EPS = 1e-12
    model = models.UnrollingHelicesShiftingZ(
        djs=np.arange(-20, 20+EPS, 10),
        dbscan_features=["sina1", "cosa1", "z1", "z2"],
        dbscan_weight  =[1.0,     1.0,    0.75, 0.2],
        niter=150)
                                             
    nevents = 1
    path_to_input = os.path.join(path_to_trackml, "train_1")
    path_to_out   = "out_{0}".format(sys.argv[0].split(".")[0])

    event_id_list = []
    hits_list = []
    truth_list = []
    sys.stderr.write("load data\n")
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              skip=0, nevents=nevents):
        event_id_list.append(event_id)
        hits_list.append(hits)
        truth_list.append(truth)

    def Fun4BO(w_a1, w_z1, w_z2):
        model.dbscan_weight[0] = w_a1
        model.dbscan_weight[1] = w_a1
        model.dbscan_weight[2] = w_z1
        model.dbscan_weight[3] = w_z2
        
        sys.stderr.write("scan\n")
        score_list = []
        for (event_id, hits, truth) in zip(event_id_list, hits_list, truth_list):
            label = model.predict(hits)
            submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
                                      data=np.column_stack(([int(event_id),]*len(hits),
                                                            hits.hit_id.values,
                                                            label))).astype(int)
            score = score_event(truth, submission)
            score_list.append(score)
        return np.sum(score_list) / len(score_list)
            
    opt = BayesianOptimization(Fun4BO,
                               {"w_a1": (0.9, 1.2),
                                "w_z1": (0.3, 0.7),
                                "w_z2": (0.1, 0.4)
                                }, verbose = True)
    opt.maximize(init_points = 3,
                 n_iter = 30, # 30
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
    run(sys.argv[0].split(".")[0] + ".csv")
    
