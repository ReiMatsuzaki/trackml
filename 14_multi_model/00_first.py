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
from models import UnrollingHelicesModel

def run():
    print(datetime.datetime.now(), sys.argv[0], " begin")

    path_to_input = os.path.join(path_to_trackml, "train_1")
    model = UnrollingHelicesModel()

    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                       skip=0, nevents=1):
        print("len(hits): ", len(hits))
        labels = model.fit_predict(hits)
        score = model.score(hits, truth)
        print("score: ", score)

    print(datetime.datetime.now(), sys.argv[0], " end")
            
if __name__=="__main__":
    run()
