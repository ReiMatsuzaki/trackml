import os
import sys
import models
import train

if __name__=="__main__":
    model = models.UnrollingHelices(use_outlier=False,
                                    iter_size_helix=100,
                                    dz0    = -0.00070,
                                    stepdz = +0.00001,
                                    dbscan_features = ["sina1", "cosa1", "z1", "x1", "x2"],
                                    dbscan_weight   = [1.0,     1.0,     0.75, 0.5,  0.5])
    train.run(model, "train_1",
              path_to_out=os.path.join("out", sys.argv[0].split(".")[0]),
              1)


    
