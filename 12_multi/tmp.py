import sys
import os
join = os.path.join

import numpy as np
import pandas as pd
import datetime

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

sys.path.append("..")
import make_candidates
import merge
import score_track

def run():
    print(datetime.datetime.now(), sys.argv[0], " begin")
    
    path_to_input = os.path.join(path_to_trackml, "train_1")
    th1_len = 13
    th2_len = 10
    
    for event_id, hits in load_dataset(path_to_input, parts=["hits"],
                                       skip=0, nevents=1):
        submission0 = pd.read_csv("step0.submission.csv")
        tscore = score_track.score_by_length(submission0, hits)
        outlier_mask = tscore < th1_len
        
        df = submission0[outlier_mask]
        print(submission0.head(10))
        print(df)
        df.to_csv("step{0}_outlier.submission.csv".format(istep-1), index=None)
        df = submission0[~outlier_mask]
        df.to_csv("step{0}_good.submission.csv".format(istep-1), index=None)

    print(datetime.datetime.now(), sys.argv[0], " end")

if __name__=="__main__":
    run()
