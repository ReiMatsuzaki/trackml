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
from make_candidates import UnrollingHelices
from merge import LengthMerge
import score_track
from extension import extend

def calc_one(istep, submission0, event_id, hits, output_dir, mk_cand, merger, th_len, num_extend):
    
    candidates_dir = os.path.join(output_dir, "candidates{0}".format(istep))
    mk_cand.output_dir = candidates_dir
    merger.candidates_output_dir = candidates_dir
    
    if(submission0 is None):
        hits0 = hits
    else:
        hits0 = submission0.merge(hits, on="hit_id")[hits.columns]

    # -- make candidate --
    print(datetime.datetime.now(), "step{0} make_candidates".format(istep))
    mk_cand.run(event_id, hits0)
    
    # -- merge --
    print(datetime.datetime.now(), "step{0} merge".format(istep))
    submission = merger.run(event_id, hits0)

    # -- extend --
    print(datetime.datetime.now(), "step{0} extend {1} times".format(istep, num_extend))
    for i in range(num_extend):
        submission = extend(submission, hits0)

    # -- separate outlier --
    print(datetime.datetime.now(), "step{0} compute outlier".format(istep, num_extend))
    tscore = score_track.score_by_length(submission, hits)
    outlier_mask = tscore < th_len
    submission_outlier = submission[outlier_mask]
    submission_good    = submission[~outlier_mask]
    print("# of outlier: ", len(submission_outlier))
    print("# of good: ",    len(submission_good))
    

    # -- save --
    submission_outlier.to_csv(
        os.path.join(output_dir,
                     "step{0}_outlier.submission.csv".format(istep)), index=None)
    submission_good.to_csv(
        os.path.join(output_dir,
                     "step{0}_good.submission.csv".format(istep)), index=None)
    submission.to_csv(
        os.path.join(output_dir,
                     "step{0}.submission.csv".format(istep)), index=None)
        
    return (submission_good, submission_outlier)

def calc_steps(niter, eps0s, th_lens, num_exts, output_dir):

    if(type(niter)!=list):
        niter = [niter] * len(eps0s)
    
    print("eps list: ", eps0s)
    print("th_len list: ", th_lens)
    print("num ext list: ", num_exts)
    
    path_to_input = os.path.join(path_to_trackml, "train_1")
    path_to_out = "out_" + sys.argv[0].split(".")[0]

    for event_id, hits, truth in load_dataset(path_to_input, parts=["hits", "truth"],
                                              skip=0, nevents=1):
        print("len(hits): ", len(hits))

        sub_out = None
        subs_good = []
        subs_out  = []        
        
        for i in range(len(eps0s)):
            sub_good, sub_out = calc_one(i+1, sub_out, event_id, hits, path_to_out,
                                         UnrollingHelices(niter=niter[i],
                                                          eps0=eps0s[i]),
                                         LengthMerge(),
                                         th_lens[i], num_exts[i] )
            subs_good.append(sub_good)
            subs_out.append( sub_out)

            submission = pd.concat(subs_good + [subs_out[-1]])
            total_score = score_event(truth, submission)
            print("step {0}, total_score:{1}".format(i+1, total_score))            

            truth_good = sub_good.merge(truth, on="hit_id")[truth.columns]
            score_good = score_event(truth_good, sub_good)
            print("step {0}, good_score: {1}".format(i+1, score_good))
    return total_score
    
