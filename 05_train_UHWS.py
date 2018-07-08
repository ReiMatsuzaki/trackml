import os
import sys
import models
import train

if __name__=="__main__":
    model = models.UnrollingHelicesWithScore(iter_size_helix=100)
    train.run(model, "train_1",
              path_to_out=os.path.join("out", sys.argv[0].split(".")[0]),              
              nevents=1)
