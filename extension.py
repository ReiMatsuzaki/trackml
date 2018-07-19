import sys
import numpy as np
from sklearn.neighbors import KDTree
from utils import create_one_event_submission
from sklearn.preprocessing import StandardScaler
import hdbscan
from tqdm import tqdm
import datetime

from score_track import score_by_quadric

class RemoveOutliers(object):
    def __init__(self, rz_scales=[0.65, 0.965, 1.528]):
        self.rz_scales = np.array(rz_scales)

    def test_quadric(self,x):
        if x.size == 0 or len(x.shape)<2:
            return 0
        Z = np.zeros((x.shape[0],10), np.float32)
        Z[:,0] = x[:,0]**2
        Z[:,1] = 2*x[:,0]*x[:,1]
        Z[:,2] = 2*x[:,0]*x[:,2]
        Z[:,3] = 2*x[:,0]
        Z[:,4] = x[:,1]**2
        Z[:,5] = 2*x[:,1]*x[:,2]
        Z[:,6] = 2*x[:,1]
        Z[:,7] = x[:,2]**2
        Z[:,8] = 2*x[:,2]
        Z[:,9] = 1
        v, s, t = np.linalg.svd(Z,full_matrices=False)
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index,:]
        norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
        return norm

    def preprocess(self, hits):
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values
        
        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r
        
        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r
        
        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        X = X * self.rz_scales[np.newaxis,:]
        return X

    def eliminate_outliers(self, labels, X):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            x = X[index]
            norms[i] = self.test_quadric(x)
            
        threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 6
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters==cluster]=0
    
    def run(self, submission, hits):
        print(datetime.datetime.now(), "RemoveOutliers begin")
        X = self.preprocess(hits)
        self.clusters = submission["track_id"].values
        labels = np.unique(self.clusters)

        print(datetime.datetime.now(), "eliminate_outliers")

        self.eliminate_outliers(labels, X)

        print(datetime.datetime.now(), "HDSCAN")
        max_len = np.max(self.clusters)
        mask = (self.clusters==0)
        print("# of outliner : ", len(self.clusters[mask]))
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',
                             cluster_selection_method='leaf',
                             algorithm='best',
                             leaf_size=50)
        self.clusters[mask] = cl.fit_predict(X[mask]) + max_len

        df = submission.copy()
        df["track_id"] = self.clusters
        print(datetime.datetime.now(), "RemoveOutliers end")
        return df

class RemoveOutliersByQuadric(object):
    def __init__(self, rz_scales=[0.65, 0.965, 1.528], threshold=-1e-5):
        self.rz_scales = np.array(rz_scales)
        self.threshold = threshold
        
    def preprocess(self, hits):
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values
        
        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r
        
        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r
        
        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        X = X * self.rz_scales[np.newaxis,:]
        return X

    def run(self, submission, hits):
        print(datetime.datetime.now(), "RemoveOutliersByQuadric begin")
        print("len(hits) : ", len(hits))
        clusters = submission["track_id"].values
        print("len(clusters) : ", len(clusters))

        print(datetime.datetime.now(), "evaluate score")
        score = score_by_quadric(submission, hits, verbose=True)
        print("max[score] : ", np.max(score))        
        print("scores[:10]", score[:10])

        print(datetime.datetime.now(), "HDSCAN")
        threshold = self.threshold
        print("threashfold : ", threshold)
        
        mask = (score<threshold)
        
        print("# of outliers : ", len(clusters[mask]))
        X = self.preprocess(hits)
        max_len = np.max(clusters)
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',
                             cluster_selection_method='leaf',
                             algorithm='best',
                             leaf_size=50)
        clusters[mask] = cl.fit_predict(X[mask]) + max_len

        df = submission.copy()
        df["track_id"] = clusters
        print(datetime.datetime.now(), "RemoveOutliersByQuadric end")
        return df
    
def extend(submission,hits,limit=0.04, num_neighbours=18, verbose=False):

    if(verbose):
        print(datetime.datetime.now(), "extend begin")
    
    df = submission.merge(hits,  on=['hit_id'], how='left')
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))

    angles = range(-90,90,1)
    for angle in tqdm(angles, total=len(angles)):
        
        #df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)]
        df1 = df.loc[(df.arctan2>(angle-1.5)/180*np.pi) & (df.arctan2<(angle+1.5)/180*np.pi)]
        
        min_num_neighbours = len(df1)
        if min_num_neighbours<3: continue
        
        hit_ids = df1.hit_id.values
        x,y,z = df1[['x', 'y', 'z']].values.T
        r  = (x**2 + y**2)**0.5
        r  = r/1000
        a  = np.arctan2(y,x)
        c = np.cos(a)
        s = np.sin(a)
        #tree = KDTree(np.column_stack([a,r]), metric='euclidean')
        tree = KDTree(np.column_stack([c, s, r]), metric='euclidean')
        
        
        track_ids = list(df1.track_id.unique())
        num_track_ids = len(track_ids)
        min_length=3
        
        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue
            
            idx = np.where(df1.track_id==p)[0]
            if len(idx)<min_length: continue
            
            if angle>0:
                idx = idx[np.argsort( z[idx])]
            else:
                idx = idx[np.argsort(-z[idx])]
                
                
            ## start and end points  ##
            idx0,idx1 = idx[0],idx[-1]
            a0 = a[idx0]
            a1 = a[idx1]
            r0 = r[idx0]
            r1 = r[idx1]
            c0 = c[idx0]
            c1 = c[idx1]
            s0 = s[idx0]
            s1 = s[idx1]
            
            da0 = a[idx[1]] - a[idx[0]]  #direction
            dr0 = r[idx[1]] - r[idx[0]]
            direction0 = np.arctan2(dr0,da0)
            
            da1 = a[idx[-1]] - a[idx[-2]]
            dr1 = r[idx[-1]] - r[idx[-2]]
            direction1 = np.arctan2(dr1,da1)
            
            
            
            ## extend start point
            ns = tree.query([[c0, s0, r0]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
            ns = np.concatenate(ns)
            
            direction = np.arctan2(r0 - r[ns], a0 - a[ns])
            diff = 1 - np.cos(direction - direction0)
            ns = ns[(r0 - r[ns] > 0.01) & (diff < (1 - np.cos(limit)))]
            for n in ns: df.loc[df.hit_id == hit_ids[n], 'track_id'] = p
            
            ## extend end point
            ns = tree.query([[c1, s1, r1]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
            ns = np.concatenate(ns)
            
            direction = np.arctan2(r[ns] - r1, a[ns] - a1)
            diff = 1 - np.cos(direction - direction1)
            ns = ns[(r[ns] - r1 > 0.01) & (diff < (1 - np.cos(limit)))]
            for n in ns:  df.loc[df.hit_id == hit_ids[n], 'track_id'] = p
            
    #print ('\r')
    df = df[['event_id', 'hit_id', 'track_id']]
    if(verbose):
        print(datetime.datetime.now(), "extend end")
    return df




