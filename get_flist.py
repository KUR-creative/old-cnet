import os, sys, random
import utils

dirpath = sys.argv[1]
train_flist_name = sys.argv[2]
valid_flist_name = sys.argv[3]

paths = list(utils.file_paths(dirpath))
num_valids = len(paths) // 10
random.shuffle(paths)

train_paths = paths[:-num_valids]
valid_paths = paths[-num_valids:]
print(num_valids)
print('num_trains:', len(train_paths))
print('num_valids:', len(valid_paths))

with open(train_flist_name, 'w') as f:
    f.write('\n'.join(train_paths))
with open(valid_flist_name, 'w') as f:
    f.write('\n'.join(valid_paths))
