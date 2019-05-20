'''
Utils for File Processing
'''
import os
import re
from pathlib import PurePosixPath, Path

def file_pathseq(root_dirpath):
    ''' generate file path sequence of directory_path ''' 
    it = os.walk(root_dirpath)
    for root,dirs,files in it:
        for path in map(lambda name:PurePosixPath(root) / name,files):
            yield str(path)

def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

def write_text(path, text, mode=0o777, exist_ok=False):
    path = Path(PurePosixPath(path))
    os.makedirs(path.parent, mode, exist_ok)
    path.write_text(text)
