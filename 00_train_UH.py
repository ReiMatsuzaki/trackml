import os
import sys
import pandas as pd
import numpy as np
import datetime
import models
from utils import create_one_event_submission

if __name__=="__main__":
    path_to_trackml = os.path.expanduser("~/trackml")
    path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
    path_to_input = os.path.join(path_to_trackml, "train_1")
    nevents = 2
    path_to_out = os.path.join("out", sys.argv[0].split(".")[0])
    path_to_log = os.path.join(path_to_out, "calc.log")
    model = models.UnrollingHelices(use_outlier=False, iter_size_helix=100)

if __name__=="__main__":
    sys.path.append(path_to_trackmllib)
    from trackml.dataset import load_dataset
    from trackml.score import score_event
    
    os.makedirs(path_to_out, exist_ok=True)
    f_log = open(path_to_log, "w")
    f_log.write("calculation begin\n")
    f_log.write(str(datetime.datetime.today())+"\n")

    dataset_submission = []
    dataset_score = []
    for event_id, hits, cells, particles, truth in load_dataset(path_to_input, skip=0, nevents=nevents):        
        labels = model.predict(hits)

        one_submission = create_one_event_submission(event_id, hits, labels)
        dataset_submission.append(one_submission)

        score = score_event(truth, one_submission)
        dataset_score.append(score)

        f_log.write("Score for event %d:%.8f\n" % (event_id, score))
    
    submission = pd.concat(dataset_submission)
    submission.to_csv(os.path.join(path_to_out, "submission.csv"), index=None)
    f_log.write("Mean Score : %.8f\n" % (np.sum(dataset_score)/len(dataset_score)))
    f_log.write("calculation end\n")
    f_log.write(str(datetime.datetime.today()))

    
