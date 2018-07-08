import os
join = os.path.join
exists = os.path.exists
expanduser = os.path.expanduser
import sys
import numpy as np
tr = np.transpose
import pandas as pd

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission
