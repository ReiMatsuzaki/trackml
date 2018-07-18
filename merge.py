import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from score_track import score_by_quadric, score_by_length

# reading candidates file created by run method in scan.py, megin these datas as one submission file.


class LengthMerge(object):
    def __init__(self, candidates_output_dir = "candidates"):
        self.candidates_output_dir = candidates_output_dir
    
    def run(self, event_id, dfh):
        candidates_dir = os.path.join(self.candidates_output_dir, "{:0=9}".format(event_id))
        candidates = os.listdir(candidates_dir)        
        sys.stderr.write("merge begin\n")
        
        for (i, csvfile) in tqdm(enumerate(candidates), total=len(candidates)):
            candidate = pd.read_csv(os.path.join(candidates_dir, csvfile))
            if(i==0):
                best_candidate = candidate.copy()
                best_candidate_num = score_by_length(best_candidate, dfh)
            else:
                candidate_num = score_by_length(candidate, dfh)
                max_best = np.max(best_candidate["track_id"].values)
                best_candidate["track_id"] = np.where(
                    (candidate_num > best_candidate_num) & (candidate_num < 20),
                    candidate["track_id"].values + max_best,
                    best_candidate["track_id"].values)
                best_candidate_num = score_by_length(best_candidate, dfh)

        return best_candidate

class QuadricMerge(object):
    def __init__(self, candidates_output_dir = "candidates"):
        self.candidates_output_dir = candidates_output_dir
    
    def run(self, event_id, dfh):
        candidates_dir = os.path.join(self.candidates_output_dir, "{:0=9}".format(event_id))
        candidates = os.listdir(candidates_dir)        
        sys.stderr.write("merge begin\n")
        
        for (i, csvfile) in tqdm(enumerate(candidates), total=len(candidates)):
            candidate = pd.read_csv(os.path.join(candidates_dir, csvfile))
            if(i==0):
                best_candidate = candidate.copy()
                best_candidate_score = score_by_quadric(best_candidate, dfh)                
            else:
                candidate_score = score_by_quadric(candidate, dfh)
                candidate_num   = score_by_length(candidate, dfh)
                maxid_best_candidate = np.max(best_candidate["track_id"].values)
                best_candidate["track_id"] = np.where(
                    (candidate_score > best_candidate_score) & (candidate_num < 20),
                    candidate["track_id"].values + maxid_best_candidate,
                    best_candidate["track_id"].values)
                best_candidate_score = score_by_quadric(best_candidate, dfh)

        return best_candidate


    
