import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# functions which calculate real valued or integer score from submission


def score_by_length(submission, dfh):
    return submission.groupby("track_id")["track_id"].transform("count")

def test_quadric(x):
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


def score_by_quadric(submission, dfh, rz_scales=[0.65, 0.965, 1.528],
                     nmin=6, nmax=25, minscore=-1.0, verbose=False):
    if verbose:
        print("score_by_quadric begin")
        
    rz_scales = np.array(rz_scales)
    
    df = submission.copy()
    df["N"] = score_by_length(submission, dfh)
    df['score'] = minscore

    clusters = submission["track_id"].values
    labels = df[(nmin < df.N) & (df.N < nmax)]["track_id"].unique()
    
    x = dfh.x.values
    y = dfh.y.values
    z = dfh.z.values
    
    r = np.sqrt(x**2 + y**2 + z**2)
    dfh['x_r'] = x/r
    dfh['y_r'] = y/r
    
    r = np.sqrt(x**2 + y**2)
    dfh['z_rt'] = z/r
    
    ss = StandardScaler()
    X = ss.fit_transform(dfh[['x_r', 'y_r', 'z_rt']].values)
    X = X * rz_scales[np.newaxis,:]
    
    if verbose:
        print("len(dfh) : ", len(dfh))
        print("len(labels) : ", len(labels))
        print("scoring for each cluster")
        get_ary = lambda: tqdm(enumerate(labels), total=len(labels))
    else:
        get_ary = lambda: enumerate(labels)
        
    for (i, cluster) in get_ary():
        if cluster==0:
            raise RuntimeError("0 found")
        
        index = np.argwhere(clusters==cluster)
        index = np.reshape(index, (index.shape[0]))
        x = X[index]
        score = -test_quadric(x)
        if score < minscore:
            score = minscore
        
        df.loc[df.track_id==cluster, "score"] = score

    if verbose:
        print("score_by_quadric end")
        
    return df["score"].values

