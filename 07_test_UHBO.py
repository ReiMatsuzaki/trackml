# parameters are determined in try_UHBO/06.log

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

import models
from utils import create_one_event_submission

def run(filename):
    w_a1 = 0.90291
    w_xy_rt = 0.010809
    w_z1 = 0.357996
    w_z2 = 0.229602
    c_rt1 = 1.330075
    c_rt2 = 1.92522
    
    model = models.UnrollingHelicesRt2(
        dbscan_features=["sina1", "cosa1", "z1", "z2", "x_rt", "y_rt"],
        dbscan_weight  =[w_a1,     w_a1,   w_z1, w_z2, w_xy_rt, w_xy_rt])
    model.coef_rt1 = c_rt1
    model.coef_rt2 = c_rt2
    model.niter = 150
    path_to_input = os.path.join(path_to_trackml, "test")
    dataset_submission = []
    for event_id, hits in load_dataset(path_to_input, parts=["hits"]):

        labels = model.predict(hits)
        
        one_submission = create_one_event_submission(event_id, hits, labels)

        dataset_submission.append(one_submission)
        
    submission = pd.concat(dataset_submission)
    submission.to_csv(filename)

if __name__=="__main__":
    run(sys.argv[0].split(".")[0]+"_submission.csv")

