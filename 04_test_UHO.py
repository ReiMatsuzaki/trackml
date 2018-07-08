import os
import sys
import models
import train

if __name__=="__main__":
    model = models.UnrollingHelices(use_outlier=True, iter_size_helix=100)
    train.run(model, "test",
              path_to_out=os.path.join("out", sys.argv[0].split(".")[0]))

