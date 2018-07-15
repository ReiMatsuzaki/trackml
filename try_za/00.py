import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
from tqdm import tqdm

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

def run_candidate():
    model = models.ZAScale(djs=np.array([0]),
                           dis=np.array([0.0]))
    nevents = 1
    path_to_input = os.path.join(path_to_trackml, "train_1")
    path_to_out   = "out_{0}".format(sys.argv[0].split(".")[0])

    os.makedirs(path_to_out, exist_ok=True)

    event_id_list = []
    hits_list = []
    print("load data")
    for event_id, hits in load_dataset(path_to_input, parts=["hits"],
                                              skip=0, nevents=nevents):                
        event_id_list.append(event_id)
        hits_list.append(hits)

    print("scan")
    for (event_id, hits) in zip(event_id_list, hits_list):

        print("# of hits : {0}".format(len(hits)))
        labels = model.predict(hits)

        """
        for (i, s) in tqdm(enumerate(labels)):
            df = pd.DataFrame(columns=["label"], data=s)
            df.to_csv(join(path_to_out, "{0}_scan{1}.csv".format(event_id, i)), index=None)
        """

if __name__=="__main__":
    run_candidate()
    
