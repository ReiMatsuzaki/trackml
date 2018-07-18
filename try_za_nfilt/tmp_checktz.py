import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
from tqdm import tqdm
import datetime

import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pylab as plt

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

def run():
    print("script begin", datetime.datetime.now())
    path_to_input = os.path.join(path_to_trackml, "train_1")
    nevents = 1
    sys.stderr.write("load data\n")
    for event_id, hits, truth, particle in load_dataset(path_to_input, parts=["hits", "truth", "particles"],
                                                        skip=0, nevents=nevents):
        z = particle["vz"].values
        for zz in [10, 20, 30]:
            print(zz, len(z[np.where(abs(z)<zz)])/len(z))
        z0 = z[np.where(abs(z)<100.0)]
        plt.hist(z, bins=100)

    plt.savefig("tmp.pdf")

if __name__=="__main__":
    run()
    
