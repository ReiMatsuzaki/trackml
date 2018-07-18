import sys
import os
import datetime

import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

sys.path.append("..")
import extension
from utils import create_one_event_submission

def run():
    print(datetime.datetime.now(), sys.argv[0], " begin")
    submission = pd.read_csv("../09_UH_len/01_merge.submission.csv")
    model = extension.RemoveOutliersByQuadric()
    
    path_to_input = os.path.join(path_to_trackml, "train_1")
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                       skip=0, nevents=1):
        submission = model.run(submission, hits)
        score = score_event(truth, submission)
        print("step1, score: %0.5f" % (score))

    print(datetime.datetime.now(), sys.argv[0], " end")
    
if __name__=="__main__":
    run()
    
    
