import numpy as np
from sklearn import preprocessing

import os
import pandas as pd


submissions_path = "kernels"
all_files = os.listdir(submissions_path)

outs=[]
for i in range(len(all_files)):
    if i==0: continue
    outs.append(pd.read_csv(submissions_path+'/'+all_files[i]))

id_test=outs[0]['id'].values
# print outs[0]

targets=[]
for i in range(len(outs)):
    # Scale
    # target=outs[i]['target'].values
    # target=preprocessing.scale(target)
    # target=preprocessing.minmax_scale(target)
    # targets.append(target)
    # Rank
    target=outs[i]['target'].rank().values
    targets.append(target/(len(target)+0.0))
print targets

targets=np.mean(targets,axis=0)


# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = targets
sub.to_csv('submit/kernel_mix.csv', index=False)



