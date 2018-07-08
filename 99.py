import sys
import models
import train

if __name__=="__main__":
    model = models.UnrollingHelices(use_outlier=False, iter_size_helix=100)
    train.run(model, "train_1", sys.argv[0].split(".")[0], 1)


    
