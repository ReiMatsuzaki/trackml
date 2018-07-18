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
import merge
import utils

def run():
    model = merge.LengthMerge()

    path_to_input = os.path.join(path_to_trackml, "train_1")
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                       skip=0, nevents=1):
        submission = model.run(event_id, hits)
        submission.to_csv("01_merge.submission.csv" ,index=None)
        score = score_event(truth, submission)
        print("score: %0.5f" % (score))

if __name__=="__main__":
    run()
    
    
