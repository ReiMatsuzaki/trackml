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
import scan

def run():
    model = scan.UnrollingHelices(niter=150)
    
    path_to_input = os.path.join(path_to_trackml, "train_1")
    for event_id, hits in load_dataset(path_to_input, parts=["hits"],
                                       skip=0, nevents=1):
        model.run(event_id, hits)

if __name__=="__main__":
    run()
