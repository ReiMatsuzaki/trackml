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
    candidates_dir = "candidates1"
    th_len = 6
    istep = 3
    candidates_maker = make_candidates.UnrollingHelices(niter=150,
                                                        output_dir=candidates_dir,
                                                        eps0=0.0080)
    merger = merge.LengthMerge(candidates_output_dir = candidates_dir)
    
    for event_id, hits in load_dataset(path_to_input, parts=["hits"],
                                       skip=0, nevents=1):
        
        print("len(hits): ", len(hits))
        path_submission0 = "step{0}.submission.csv".format(istep-1)
        print("read submission file from {0}".format(path_submission0))
        submission0 = pd.read_csv(path_submission0)

        print("calculate track score")
        tscore = score_track.score_by_length(submission0, hits)
        outlier_mask = tscore < th_len

        hits1 = submission0[outlier_mask].merge(hits, on="hit_id")
        print("len(hits1): ", len(hits1))
        
        print("# of outlier: ", len(submission0[outlier_mask]))
        print("# of good: ",    len(submission0[~outlier_mask]))
        df = submission0[outlier_mask]
        df.to_csv("step{0}_outlier.submission.csv".format(istep-1), index=None)
        df = submission0[~outlier_mask]
        df.to_csv("step{0}_good.submission.csv".format(istep-1), index=None)

        print("make candidates")
        candidates_maker.run(event_id, hits1)

        print("merge")        
        submission1 = merger.run(event_id, hits1)
        submission1.to_csv("step{0}.submission.csv".format(istep), index=None)

    print(datetime.datetime.now(), sys.argv[0], " end")

if __name__=="__main__":
    run()
