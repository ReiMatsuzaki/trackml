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

sys.path.append("..")
import models

def run_candidate():
    print("script begin", datetime.datetime.now())
    EPS = 1e-12
    model = models.UnrollingHelicesShiftingZ(
        dbscan_features = ["sina1", "cosa1", "z1", "z2", "x_y", "x_r", "y_r", "rt_r"],
        dbscan_weight   = [2.7474448671796874, 2.7474448671796874,
                           1.3649721713529086, 0.7034918842926337,
                           0.0005549122352940002, 0.023096034747190672,0.04619756315527515,0.2437077420144654],
        djs = [-20, -10, 0, 10, 20],
        niter = 150,
        eps0 = 0.00975)

    nevents = 1
    path_to_input = os.path.join(path_to_trackml, "train_1")
    path_to_out   = "out_{0}".format(sys.argv[0].split(".")[0])

    event_id_list = []
    hits_list = []
    truth_list = []
    sys.stderr.write("load data\n")
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              skip=0, nevents=nevents):
        print("size(hits) : ", len(hits))
        event_id_list.append(event_id)
        hits_list.append(hits)
        truth_list.append(truth)

    sys.stderr.write("scan\n")
    for (event_id, hits, truth) in zip(event_id_list, hits_list, truth_list):

        truth = truth.merge(hits,       on=['hit_id'],      how='left')
        dfh = truth.copy()
        label = model.predict(dfh)

        submission = pd.DataFrame(columns=['event_id', 'hit_id', 'track_id'],
                                  data=np.column_stack(([int(event_id),]*len(dfh),
                                                        dfh.hit_id.values,
                                                        label))).astype(int)
        submission.to_csv("05.csv", index=None)
        score = score_event(dfh, submission)
        max_score = dfh.weight.sum()        
        print("score: %0.5f  (%0.5f)" % (score*max_score, score))

    print("script end", datetime.datetime.now())

if __name__=="__main__":
    run_candidate()
    
