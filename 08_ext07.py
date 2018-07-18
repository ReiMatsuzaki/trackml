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

import models
import extension


def run():
    f = open("08.log", "w")
    f.write("extention of 07 results.\n")
    path_to_input = os.path.join(path_to_trackml, "test")
    old_submission = pd.read_csv("07_test_UHBO_submission.csv")
    sys.stderr.write("load data\n")
    for event_id, hits in load_dataset(path_to_input, parts=["hits"]):
        submission = extension.extend(old_submission, hits)
        submission.to_csv("08_ext07_submission.csv", index=None)        
    f.close()

if __name__=="__main__":
    run()
    
        
