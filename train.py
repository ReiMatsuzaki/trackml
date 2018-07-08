import os
join = os.path.join
import sys
import numpy as np
import pandas as pd
import json

import pandas as pd
import numpy as np
import datetime
import models
from utils import create_one_event_submission

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event


def run(model, test_or_train, path_to_out, nevents=None):
    if(test_or_train not in ["test", "train_1"]):
        sys.stderr.write("Error. test_or_train must be \"test\" or \"train_1\"\n")
        sys.exit()
    if(test_or_train=="test" and (nevents is not None)):
        sys.strerr.write("Error")
        sys.exit()

        path_to_input = os.path.join(path_to_trackml, test_or_train)

    os.makedirs(path_to_out, exist_ok=True)
    print("calculation begin : {0}".format(datetime.datetime.today()))

    dataset_submission = []    
    if(test_or_train == "test"):
        for event_id, hits in load_dataset(path_to_input, parts=["hits"]):
            sys.stderr.write("processing event_id : {0}".format(event_id))
            labels = model.predict(hits)

            one_submission = create_one_event_submission(event_id, hits, labels)
            dataset_submission.append(one_submission)
    else:
        dataset_score = []
        for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                                  skip=0, nevents=nevents):
            sys.stderr.write("processing event_id : {0}".format(event_id))
            labels = model.predict(hits)

            one_submission = create_one_event_submission(event_id, hits, labels)
            dataset_submission.append(one_submission)
            
            score = score_event(truth, one_submission)
            dataset_score.append(score)

            print("Score for event %d:%.8f" % (event_id, score))
        print("Mean Score : %.8f" % (np.sum(dataset_score)/len(dataset_score)))
        
    submission = pd.concat(dataset_submission)
    submission.to_csv(os.path.join(path_to_out, "submission.csv"), index=None)
    print("calculation end : {0}".format(datetime.datetime.today()))


