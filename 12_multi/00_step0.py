import sys
import os
join = os.path.join

import numpy as np
import pandas as pd
import datetime

from bayes_opt import BayesianOptimization

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
    candidates_dir = "candidates0"
    istep = 0
    candidates_maker = make_candidates.UnrollingHelices(niter=150,
                                                        output_dir=candidates_dir,
                                                        eps0 = 0.0010)
    merger = merge.LengthMerge(candidates_output_dir =candidates_dir)
    
    for event_id, hits in load_dataset(path_to_input, parts=["hits"],
                                       skip=0, nevents=1):
        print("len(hits): ", len(hits))
        
        print("make candidates")        
        candidates_maker.run(event_id, hits)
        
        print("merge")
        submission = merger.run(event_id, hits)

        csvfilename = "step{0}.submission.csv".format(istep)
        print("save subimission as {0}".format(csvfilename))
        submission.to_csv(csvfilename, index=None)

if __name__=="__main__":
    run()
