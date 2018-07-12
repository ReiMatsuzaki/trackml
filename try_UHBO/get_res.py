import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
import numpy as np
import pandas as pd

log_filenames = ["{0:02}.log".format(i) for i in [2,3,4,5,6]]


for log_filename in log_filenames:    
    df = pd.read_csv(log_filename)
    if(len(df) > 1):
        df = df[df.label=="max"]
    score = df["value"].values[0]

    df["filename"] = log_filename
    cols = list(df.columns)
    cols.remove("label")
    cols.remove("value")
    cols.remove("filename")
    cols = ["filename", "value"] + cols
    df = df[cols]
    
    print(df)
    print("")

