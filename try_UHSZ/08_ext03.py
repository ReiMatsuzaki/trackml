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
import extension


def run():
    path_to_input = os.path.join(path_to_trackml, "train_1")
    nevents = 1
    old_submission = pd.read_csv("03.csv")
    sys.stderr.write("load data\n")
    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              skip=0, nevents=nevents):
        submission = extension.extend(old_submission, hits)
        submission.to_csv("08.csv", index=None)
        score = score_event(truth, submission)
        print("")
        print("score: %0.5f" % (score))

if __name__=="__main__":
    run()
    
        
