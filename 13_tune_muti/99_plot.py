import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pylab as plt
import numpy as np
tr = np.transpose
import pandas as pd

import glob

indeces = ["00", "01", "06", "08", "09", "10", "11", "12", "13"]
logfiles = [glob.glob(idx+"*.log")[0] for idx in indeces]
logfiles = sorted(logfiles, key=lambda x:x.split("_")[0])

for logfile in logfiles:
    with open(logfile, "r") as f:
        lines = f.readlines()
        if(len(lines) == 0):
            continue
        total_scores = [float(line.split(":")[1] )for line in lines if ("total_score" in line)]
        good_scores = [float(line.split(":")[1] )for line in lines if ("good_score" in line)]

        name = logfile.split(".")[0]
        g, = plt.plot(total_scores, "-",  label=name)
        g, = plt.plot(good_scores,  "--", color=g.get_color())

score0 = 0.47281
plt.plot([score0]*5, "k:", label="1stage")
plt.legend(fontsize=8)
plt.savefig("99_plot_score.png")
plt.close()


for logfile in logfiles:
    with open(logfile, "r") as f:
        lines = f.readlines()
        if(len(lines) == 0):
            continue
        out_nums = [int(line.split(":")[1] )for line in lines if ("# of out" in line)]
        good_nums = [int(line.split(":")[1] )for line in lines if ("# of good" in line)]

        name = logfile.split(".")[0]
        g, = plt.plot(np.log10(np.array(good_nums)/np.array(out_nums)), "-",  label=name)
plt.legend()
plt.grid()
plt.savefig("99_plot_nums.png")
plt.close()


for logfile in logfiles:
    with open(logfile, "r") as f:
        lines = f.readlines()
        if(len(lines) == 0):
            continue
        line = [line.split(":")[1] for line in lines if ("eps " in line)][0].strip()[1:-1]
        if("," in line):
            strs = line.split(",")
        else:
            strs = line.split()
        eps = [ float(x) for x in strs]
        print(logfile, eps)
        name = logfile.split(".")[0]
        g, plt.plot(eps, "-",  label=name)
        #g, = plt.plot(good_scores,  "--", color=g.get_color())

plt.legend(fontsize=8)
plt.savefig("99_plot_eps.png")
plt.close()
