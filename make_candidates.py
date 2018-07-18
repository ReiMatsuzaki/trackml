import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.cluster.dbscan_ import dbscan
from sklearn.preprocessing import StandardScaler

from utils import create_one_event_submission
#
# Reading event file and Calculate track candidates as csv files.
# Each result csv file name is [out_dir]/[event_id]/[caondidate_id].csv and
# its header are hit_id, track_id, track_num.
#

class UnrollingHelices(object):
    def __init__(self,
                 features= ["sina1", "cosa1", "z1", "z2"],
                 weights  = [1.0,     1.0,     0.75, 0.2],
                 niter = 100,
                 coef_rt1 = 1.0,
                 coef_rt2 = 0.0,
                 eps0 = 0.0035,
                 step_eps = 0.0,
                 output_dir = "candidates"):
        if(len(features) != len(weights)):
            raise InputError("len(features) != len(weights)")

        self.features = features
        self.weights  = np.array(weights)
        self.niter = niter
        self.coef_rt1 = coef_rt1
        self.coef_rt2 = coef_rt2
        self.eps0 = eps0
        self.step_eps = step_eps
        self.output_dir = output_dir

    def run(self, event_id, dfh):
        dfh["s1"] = dfh.hit_id
        dfh["N1"] = 1
        dfh['r']  = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        dfh['z1'] = dfh['z'].values/dfh['rt'].values
        dfh['z2'] = dfh['z'].values/dfh['r'].values

        mm = 1
        for ii in tqdm(range(self.niter)):
            # unroll helices
            mm = mm*(-1)
            dfh["a1"] = dfh.a0 + mm*(self.coef_rt1 * dfh.rt.values +
                                     self.coef_rt2 * 0.000005 * dfh.rt.values**2) / 1000.0 * (ii/2)/180.0*np.pi
            dfh["sina1"] = np.sin(dfh["a1"].values)
            dfh["cosa1"] = np.cos(dfh["a1"].values)
            
            dfh['x_y'] = dfh['x'].values/dfh['y'].values
            dfh['x_rt'] = dfh['x'].values/dfh['rt'].values
            dfh['y_rt'] = dfh['y'].values/dfh['rt'].values

            # scan using DBSCAN
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[self.features].values)
            dfs[:,:] = dfs[:,:] * self.weights[np.newaxis,:]
            eps = self.eps0 + ii*self.step_eps
            labels=DBSCAN(eps=eps,min_samples=1,metric='euclidean',n_jobs=4).fit(dfs).labels_

            # save as csv file
            submission = create_one_event_submission(event_id, dfh, labels)
            output_dir = os.path.join(self.output_dir, "{:0=9}".format(event_id))
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, "{:0=4}.csv".format(ii))
            submission.to_csv(filename, index=None)
            
