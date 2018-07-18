import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.cluster.dbscan_ import dbscan
from sklearn.linear_model import LinearRegression
from itertools import product

class UnrollingHelices(object):
    def __init__(self,rz_scales=[0.65, 0.965, 1.528], use_outlier=True, iter_size_helix=100,
                 dbscan_features = ["sina1", "cosa1", "z1", "x1", "x2"],
                 dbscan_weight   = [1.0,     1.0,     0.75, 0.5, 0.5],
                 dz0    = -0.00070,
                 stepdz = +0.00001):
        self.rz_scales=rz_scales
        self.use_outlier=use_outlier
        self.iter_size_helix=iter_size_helix
        self.dbscan_features = dbscan_features
        self.dbscan_weight   = np.array(dbscan_weight)
        self.dz0 = dz0 
        self.stepdz = stepdz
        
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            x = M[index]
            norms[i] = self._test_quadric(x)
            
        threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 6
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters==cluster]=0
                    
    def _test_quadric(self,x):
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
    
    def _preprocess(self, hits):
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
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
        return X
        
    def _init(self,dfh):
        dfh['r'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        dfh['z1'] = dfh['z'].values/dfh['rt'].values
        dfh['x2'] = 1/dfh['z1'].values
        
        stepeps = 0.000005
        mm = 1
        for ii in tqdm(range(self.iter_size_helix)):
            mm = mm*(-1)
            dz = mm*(self.dz0+ii*self.stepdz)

            # calculate angles for ii th rotation
            dfh['a1'] = dfh['a0'].values + dz * abs(dfh['z'].values)
            dfh['sina1'] = np.sin(dfh['a1'].values)
            dfh['cosa1'] = np.cos(dfh['a1'].values)
            dfh['x1'] = dfh['a1'].values/dfh['z1'].values
            dfh['x_y'] = dfh['x'].values/dfh['y'].values
            dfh['x_rt'] = dfh['x'].values/dfh['rt'].values
            dfh['y_rt'] = dfh['y'].values/dfh['rt'].values

            ss = StandardScaler()
            df2 = ss.fit_transform(dfh[self.dbscan_features].values)
            df2[:,:] = df2[:,:] * self.dbscan_weight[np.newaxis,:]
            clusters=DBSCAN(eps=0.0035+ii*stepeps,min_samples=1,metric='euclidean',n_jobs=4).fit(df2).labels_
            if ii==0:
                dfh['s1'] = clusters
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            else:
                dfh['s2'] = clusters
                dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                maxs1 = dfh['s1'].max()
                cond = np.where((dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values<20))
                s1 = dfh['s1'].values
                s1[cond] = dfh['s2'].values[cond]+maxs1
                dfh['s1'] = s1
                dfh['s1'] = dfh['s1'].astype('int64')
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values
    
    def predict(self, hits):

        # initial clustering by DBSCAN + length merge
        self.clusters = self._init(hits)
        
        if not self.use_outlier:
            return self.clusters

        # remove outlinlier
        X = self._preprocess(hits)
        labels = np.unique(self.clusters)
        self._eliminate_outliers(labels,X)

        # clustering others
        max_len = np.max(self.clusters)
        mask = (self.clusters==0)
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',
                             cluster_selection_method='leaf',
                             algorithm='best',
                             leaf_size=50)
        self.clusters[mask] = cl.fit_predict(X[mask]) + max_len
        return self.clusters

class UnrollingHelicesWithScore(object):    
    def __init__(self, rz_scales=[0.65, 0.965, 1.528], size_dz=100,
                 dz0=-0.00070, max_dz=0.01, th_nhits=10, th_score=0.95,
                 dbscan_features = ["sina1", "cosa1", "z1", "x1", "x2"],
                 dbscan_cs       = [1.0,     1.0,     0.75, 0.5, 0.5]):
        self.rz_scales = rz_scales
        self.size_dz = size_dz
        self.dz0 = dz0
        self.max_dz = max_dz
        self.th_nhits = th_nhits
        self.th_score = th_score
        self.dbscan_features = np.array(dbscan_features)
        self.dbscan_cx       = np.array(dbscan_cs)

    def predict(self, dfh):
        dfh['r'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        dfh['z1'] = dfh['z'].values/dfh['rt'].values
        dfh['x2'] = 1/dfh['z1'].values
        
        # dfh['score'] = np.zeros(len(dfh))
        dfh['find'] = np.zeros(len(dfh), dtype=int)
        dfh['track'] = np.zeros(len(dfh), dtype=int)
        dfh['nhits'] = np.zeros(len(dfh), dtype=int)

        dz0 = -0.00070
        stepdz = self.max_dz/self.size_dz
        stepeps = 0.000005
        mm = 1
        for ii in tqdm(range(self.size_dz)):
            mm = mm*(-1)
            dz = mm*(dz0+ii*stepdz)

            # calculate angles for ii th rotation
            dfh['a1'] = dfh['a0'].values+dz*dfh['z'].values*np.sign(dfh['z'].values)
            dfh['sina1'] = np.sin(dfh['a1'].values)
            dfh['cosa1'] = np.cos(dfh['a1'].values)
            dfh['x1'] = dfh['a1'].values/dfh['z1'].values

            # clustering
            ss = StandardScaler()
            df2 = ss.fit_transform(dfh.loc[dfh.find==0, self.dbscan_features].values)
            df2[:,:] = df2[:,:] * self.cx[np.newaxis,:]
            clusters=DBSCAN(eps=0.0035+ii*stepeps,min_samples=1,metric='euclidean',n_jobs=4).fit(df2).labels_

            # memorize results
            dfh.loc[dfh.find==0, "track"] = clusters + dfh.track.max()
            dfh.loc[dfh.find==0, "nhits"] = dfh[dfh.find==0].groupby('track')['track'].transform('count')
            tracks = dfh.track.loc[(dfh.nhits>self.th_nhits)&(dfh.find==0)].unique()            
            for track in tracks:
                dfh2 = dfh[(dfh.find==0)&(dfh.track==track)].sort_values(by="z")
                [x, y] = [np.transpose(np.array([dfh2[lbl]])) for lbl in ["z", "a0"]]
                model = LinearRegression()
                model.fit(x,y)
                score = model.score(x, y)
                if(score > self.th_score):
                    dfh.loc[dfh.track==track,"find"] = 1

        return dfh["track"]
                
            
class UnrollingHelicesRt2(object):
    """
    DBSCAN with Unrolling helices based on https://www.kaggle.com/sionek/bayesian-optimization.
    """
    def __init__(self,
                 dbscan_features= ["sina1", "cosa1", "z1", "z2"],
                 dbscan_weight  = [1.0,     1.0,     0.75, 0.2],
                 niter = 100,
                 coef_rt1 = 1.0,
                 coef_rt2 = 1.0,
                 eps0 = 0.0035,
                 step_eps = 0.0):

        if(len(dbscan_features) != len(dbscan_weight)):
            raise InputError("len(dbscan_features) != len(dbscan_weight)")

        self.dbscan_features = dbscan_features
        self.dbscan_weight = np.array(dbscan_weight)
        self.niter = 100
        self.coef_rt1 = coef_rt1
        self.coef_rt2 = coef_rt2
        self.eps0 = eps0
        self.step_eps = step_eps # 0.000005 is used in mod-dbscan x 100
        
    def predict(self, dfh):
        niter = self.niter
        dfh["s1"] = dfh.hit_id
        dfh["N1"] = 1
        dfh['r']  = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        dfh['z1'] = dfh['z'].values/dfh['rt'].values
        dfh['z2'] = dfh['z'].values/dfh['r'].values

        mm = 1
        for ii in tqdm(range(niter)):
            # unroll helices
            mm = mm*(-1)
            dfh["a1"] = dfh.a0 + mm*(self.coef_rt1 * dfh.rt.values +
                                     self.coef_rt2 * 0.000005 * dfh.rt.values**2) / 1000.0 * (ii/2)/180.0*np.pi
            dfh["sina1"] = np.sin(dfh["a1"].values)
            dfh["cosa1"] = np.cos(dfh["a1"].values)
            
            dfh['x_y'] = dfh['x'].values/dfh['y'].values
            dfh['x_rt'] = dfh['x'].values/dfh['rt'].values
            dfh['y_rt'] = dfh['y'].values/dfh['rt'].values

            # scaling
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[self.dbscan_features].values)
            dfs[:,:] = dfs[:,:] * self.dbscan_weight[np.newaxis,:]

            # clustering
            res=DBSCAN(eps=self.eps0+ii*self.step_eps,min_samples=1,metric='euclidean',n_jobs=4).fit(dfs).labels_
            dfh["s2"] = res
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = np.max(dfh.s1)
            dfh.s1 = np.where((dfh.N2>dfh.N1)&(dfh.N2<20),
                              dfh.s2 + maxs1,
                              dfh.s1)
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

        return dfh["s1"]

class ZAScale(object):
    EPS = 1e-12
    def __init__(self,
                 djs=[-20, 0, 20],
                 dis=[-0.003, 0.0, 0.003]):
        self.djs = djs
        self.dis = dis

        # djs = np.arange(-20, 20+EPS, 10)
        # dis = np.arange(-0.003, 0.003+EPS, 0.00025)
    
    def predict(self, dfh):
        print("size(dfh): {0}".format(len(dfh)))

        if("rt" not in dfh.columns):
            dfh["rt"] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)

        z  = dfh['z'].values
        rt = dfh["rt"].values
        a0 = np.arctan2(dfh['y'].values,dfh['x'].values)
        layer_id = dfh['layer_id'].values.astype(np.float32)

        sys.stderr.write("dbscan for each (z,a) shifting\n")
        scan_labels = []
        for (dj, di) in tqdm(product(self.djs, self.dis), total=len(self.djs)*len(self.dis)):
            ar = a0 + di*rt
            zr = (z+dj)/rt * 0.1
            data2 = np.column_stack([ar, zr])
            
            _,scan_label = dbscan(data2, eps=0.0025, min_samples=1,)
            scan_labels.append(scan_label)

        sys.stderr.write("make candidates\n")
        candidates = []
        for scan_label in tqdm(scan_labels):
            l = scan_label
            unique = np.unique(l)
            for u in unique:
                candidate = np.where(l==u)[0]
                candidates.append(candidate)

        print("# of candidates : {0}".format(len(candidates)))        
        count = np.array([len(candidate) for candidate in candidates])
        sort = np.argsort(-count)
        candidates = [candidates[s] for s in sort]

        max_label = 1
        N = len(dfh)
        label = np.zeros(N,np.int32)
        count = np.zeros(N,np.int32)

        sys.stderr.write("calculate clustering label from candidates\n")
        for candidate in tqdm(candidates):
            n = candidate
            L = len(n)

            n = n[np.argsort(np.fabs(z[n]))]
            layer_id0 = layer_id[n[:-1]]
            layer_id1 = layer_id[n[1: ]]
            ld = layer_id1-layer_id0
            if np.any(ld>2): continue

            m = count[n].max()
            if L<m: continue

            count[n] = L
            label[n] = max_label
            max_label += 1
                        
        return label

class ZAScaleNFilter(object):
    def __init__(self, 
                 djs=[-20, 0, 20],
                 dis=[-0.003, 0.0, 0.003], param_type=0, weight=None,
                 eps=0.0025):
        self.djs = djs
        self.dis = dis
        self.param_type = param_type
        self.weight = weight
        self.eps = eps

    def predict(self, dfh):
        print("len(dfh) : {0}".format(len(dfh)))
        
        if("rt" not in dfh.columns):
            dfh["rt"] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)

        z  = dfh['z'].values
        rt = dfh["rt"].values
        r  = np.sqrt(dfh['x']**2 + dfh['y']**2 + dfh['z']**2)
        a0 = np.arctan2(dfh['y'].values,dfh['x'].values)
        layer_id = dfh['layer_id'].values.astype(np.float32)

        sys.stderr.write("dbscan for each parameters\n")
        scan_labels_list = []
        for (dj, di) in tqdm(product(self.djs, self.dis), total=len(self.djs)*len(self.dis)):
            ar = a0 + di*rt
            zr = (z+dj)/rt * 0.1
            if(self.param_type==0):                
                params = [ar, zr]
            elif(self.param_type==1):
                params = [np.sin(ar), np.cos(ar), zr*10, 1/(10*zr)]
            elif(self.param_type==2):
                params = [np.sin(ar), np.cos(ar), zr]
            else:
                raise RuntimeError("invalid param_type.")

            if(self.weight is None):
                w = np.array([1.0 for _ in params])
            else:
                w = np.array(self.weight)

            ss = StandardScaler()            
            data1 = ss.fit_transform(np.column_stack(params))
            data2 = w[np.newaxis,:] * data1
                
            _,scan_label = dbscan(data2, eps=self.eps, min_samples=1,)
            scan_labels_list.append(scan_label)

        sys.stderr.write("clustering\n")
        dfh["s1"] = dfh.hit_id
        dfh["N1"] = 1
        for scan_labels in scan_labels_list:
            dfh["s2"] = scan_labels
            dfh["N2"] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = np.max(dfh.s1)
            dfh.s1 = np.where((dfh.N2>dfh.N1)&(dfh.N2<20),
                              dfh.s2 + maxs1,
                              dfh.s1)
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        labels = dfh['s1']
        
        return labels

class UnrollingHelicesShiftingZ(object):
    """
    DBSCAN with Unrolling helices based on https://www.kaggle.com/sionek/bayesian-optimization.
    """
    def __init__(self,
                 dbscan_features= ["sina1", "cosa1", "z1", "z2"],
                 dbscan_weight  = [1.0,     1.0,     0.75, 0.2],
                 niter = 100,
                 djs = [-20, 0, 20],
                 coef_rt1 = 1.0,
                 coef_rt2 = 1.0,
                 eps0 = 0.0035,
                 step_eps = 0.0):

        if(len(dbscan_features) != len(dbscan_weight)):
            raise InputError("len(dbscan_features) != len(dbscan_weight)")

        self.dbscan_features = dbscan_features
        self.dbscan_weight = np.array(dbscan_weight)
        self.niter = niter
        self.djs = djs
        self.coef_rt1 = coef_rt1
        self.coef_rt2 = coef_rt2
        self.eps0 = eps0
        self.step_eps = step_eps # 0.000005 is used in mod-dbscan x 100
        
    def predict(self, dfh):
        niter = self.niter
        dfh["s1"] = dfh.hit_id
        dfh["N1"] = 1
        dfh['r']  = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        if("x_y" in self.dbscan_features):
            dfh['x_y'] = dfh['x'].values/dfh['y'].values
        if("x_rt" in self.dbscan_features):
            dfh['x_rt'] = dfh['x'].values/dfh['rt'].values
        if("y_rt" in self.dbscan_features):
            dfh['y_rt'] = dfh['y'].values/dfh['rt'].values
        if("y_r" in self.dbscan_features):
            dfh['y_r'] = dfh['y'].values/dfh['r'].values
        if("x_r" in self.dbscan_features):
            dfh['x_r'] = dfh['x'].values/dfh['r'].values            
        if("rt_r" in self.dbscan_features):
            dfh['rt_r'] = dfh['rt'].values/dfh['r'].values            

        mm = 1
        for (ii, dj) in tqdm(product(range(niter), self.djs),
                             total=niter*len(self.djs)):
            # unroll helices
            mm = mm*(-1)
            dfh['z1'] = (dfh['z'].values+dj)/dfh['rt'].values
            dfh['z2'] = (dfh['z'].values+dj)/dfh['r'].values
            dfh["a1"] = dfh.a0 + mm*(self.coef_rt1 * dfh.rt.values +
                                     self.coef_rt2 * 0.000005 * dfh.rt.values**2) / 1000.0 * (ii/2)/180.0*np.pi
            dfh["sina1"] = np.sin(dfh["a1"].values)
            dfh["cosa1"] = np.cos(dfh["a1"].values)
            
            # scaling
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[self.dbscan_features].values)
            dfs[:,:] = dfs[:,:] * self.dbscan_weight[np.newaxis,:]

            # clustering
            res=DBSCAN(eps=self.eps0+ii*self.step_eps,min_samples=1,metric='euclidean',n_jobs=4).fit(dfs).labels_
            dfh["s2"] = res
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = np.max(dfh.s1)
            dfh.s1 = np.where((dfh.N2>dfh.N1)&(dfh.N2<20),
                              dfh.s2 + maxs1,
                              dfh.s1)
            dfh['s1'] = dfh['s1'].astype('int64')
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')

        return dfh["s1"]
    
