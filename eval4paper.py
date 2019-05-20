import pandas as pd
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
def l1loss(img1, img2): # 0.0 < img[y][x] < 1.0
    #assert img1.max() <= 1.0 and img2.max() <= 1.0
    assert img1.shape == img2.shape, \
        '{} != {}'.format(img1.shape,img2.shape)
    return np.mean(np.abs(img1 - img2))

def l2loss(img1, img2): # 0.0 < img[y][x] < 1.0
    #assert img1.max() <= 1.0 and img2.max() <= 1.0
    assert img1.shape == img2.shape, \
        '{} != {}'.format(img1.shape,img2.shape)
    return np.mean((img1 - img2)**2)

import math
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', default='', type=str,
                    help='The directory name of images to be completed.')
parser.add_argument('--resdir', default='output.png', type=str,
                    help='Where to write output.')
import fp
from futils import human_sorted,file_pathseq
if __name__ == "__main__":
    args = parser.parse_args()

    args.imgdir = './dset4paper/imgs/'
    #args.resdir = './dset4paper/hasT_later/'
    args.resdir = './dset4paper/noT_later/'
    img_paths = fp.pipe(file_pathseq,human_sorted)(args.imgdir)
    res_paths = fp.pipe(file_pathseq,human_sorted)(args.resdir)

    originseq1 = fp.map(cv2.imread, img_paths)
    originseq2 = fp.map(cv2.imread, img_paths)
    originseqf = fp.pipe(
        fp.cmap(cv2.imread),
        fp.cmap(lambda im: im.astype(np.float32) / 255.0)
    )(img_paths)    

    resultseq1 = fp.map(cv2.imread, res_paths)
    resultseq2 = fp.map(cv2.imread, res_paths)
    resultseqf = fp.pipe(
        fp.cmap(cv2.imread),
        fp.cmap(lambda im: im.astype(np.float32) / 255.0)
    )(res_paths)

    start = time.time() #------------------------------

    img_ids  = fp.lmap(lambda p: Path(p).stem, img_paths) + ['mean']
    res_ids  = fp.lmap(lambda p: Path(p).stem, res_paths) + ['mean']
    for iid,rid in zip(img_ids,res_ids): assert iid == rid

    l1losses = fp.lmap(l1loss, originseq1,resultseq1) 
    l2losses = fp.lmap(l2loss, originseq2,resultseq2) 
    psnrs    = fp.lmap(psnr,   originseqf,resultseqf) 
    l1losses += [ np.mean(l1losses) ]
    l2losses += [ np.mean(l2losses) ]
    psnrs    += [ np.mean(psnrs) ]
    rows = zip(img_ids, l1losses, l2losses, psnrs)
    header = [  'id',  'L1 loss','L2 loss','PSNR']
    #means = fp.lmap(np.mean, [l1losses, l2losses, psnrs])

    df = pd.DataFrame(data=rows, columns=header)
    print(df)
    
    end = time.time()  #------------------------------
    print('running_time:', end - start)
'''
print( l1loss(np.ones([10,10,3]),  np.ones([10,10,3])) * 100,'%')
print( l1loss(np.zeros([10,10,3]), np.ones([10,10,3])) * 100,'%')

im   = (cv2.imread('./test_evals/1.png').astype(np.float32) / 255.0)
some = (cv2.imread('./test_evals/1changed.png').astype(np.float32) / 255.0)
many = (cv2.imread('./test_evals/1changed_many.png').astype(np.float32) / 255.0)

#im2   = (im * 255).astype(np.float32)
#some2 = (some * 255).astype(np.float32)
#many2 = (many * 255).astype(np.float32)
im2   = cv2.imread('./test_evals/1.png')
some2 = cv2.imread('./test_evals/1changed.png')
many2 = cv2.imread('./test_evals/1changed_many.png')

print('---- L1 loss ----')
print( l1loss(im,im) * 100, '% expected 0%')
print( l1loss(im,some) * 100, '% expected small %')
print( l1loss(im,many) * 100, '% expected big %')
print( l1loss(im,(1 - im)) * 100, '% expected 0%')

print('---- L2 loss ----')
print( l2loss(im,im) * 100, '% expected 0%')
print( l2loss(im,some) * 100, '% expected small %')
print( l2loss(im,many) * 100, '% expected big %')
print( l2loss(im,(1 - im)) * 100, '% expected 0%')

print('---- psnr ----')
print( psnr(im2,im2) , ' expected 100')
print( psnr(im2,some2) , ' expected big')
print( psnr(im2,many2) , ' expected small')
print( psnr(im2,(1 - im2)) , ' expected very small')


# reconstruction area only?
'''
