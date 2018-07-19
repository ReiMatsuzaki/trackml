import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
import numpy as np
tr = np.transpose
import pandas as pd

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

sys.path.append("..")
import extension

def run():

    max_istep = 4
    
    full_submission_list = [ pd.read_csv("step{0}.submission.csv".format(istep))
                             for istep in range(max_istep+1)]
    good_submission_list = [ pd.read_csv("step{0}_good.submission.csv".format(istep))
                             for istep in range(max_istep)]
    path_to_input = os.path.join(path_to_trackml, "train_1")


    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              skip=0, nevents=1):
        for mi in range(max_istep+1):
            submission_list = [full_submission_list[mi]] + [good_submission_list[i] for i in range(mi)]
            submission = pd.concat(submission_list)
            score0 = score_event(truth, submission)
            print("step = {0}, score = {1}".format(mi, score0))

            if(mi==max_istep):
                for i in range(5):
                    submission = extension.extend(submission, hits)
                    score0 = score_event(truth, submission)
                    print("with extension = {0}, score = {1}".format(i+1, score0))
        
if __name__=="__main__":
    run()

