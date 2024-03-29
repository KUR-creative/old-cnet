import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import time

from inpaint_model import InpaintCAModel
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', default='', type=str,
                    help='The directory name of images to be completed.')
parser.add_argument('--maskdir', default='', type=str,
                    help='The directory name of masks, value 255 indicates mask.')
parser.add_argument('--outdir', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

def binarization(img, threshold=100):
    binarized = (img >= threshold).astype(np.uint8) * 255
    return binarized

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

def inpaint_or_oom(image, segmap, complnet, dilate_kernel=None):
    ''' If image is too big, return None '''
    mask = binarization(segmap, 0.5)

    assert image.shape == mask.shape 

    org_h, org_w, _ = image.shape

    modulo = 8
    image = modulo_padded(image,8)
    mask = modulo_padded(mask,8)

    image = np.expand_dims(image, 0) # [h,w,c] -> [1,h,w,c]
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    try:
        result = complnet(input_image)
        assert image.shape == result.shape, \
            '{} != {}'.format(image.shape, result.shape)
        return result[0][:org_h, :org_w, ::-1] #---------- remove padding
    except Exception as e: # ResourceExhaustedError:
        print((org_h,org_w), '(Maybe) OOM error while inpainting')
        return None

compl_limit = 657666 #  then.. what is the optimal size?
#compl_limit = 1525920 # it didn't crash, but SLOWER! why..?
#lab-machine #1525920
#compl_limit = 9999999 # 
def inpaint(img, mask, complnet, dilate_kernel=None):
    ''' oom-free inpainting '''
    global compl_limit

    kernel = dilate_kernel

    h,w = img.shape[:2]
    result = None
    if h*w < compl_limit:
        result = inpaint_or_oom(img, mask, complnet, dilate_kernel=kernel)
        if result is None: # compl_limit: Ok but OOM occur!
            compl_limit = h*w
            #print('compl_limit =', compl_limit, 'updated!')
        else:
            assert img.shape == result.shape,\
                'img.{} != result.{} in no limit'.format(img.shape,result.shape)
    else:
        pass
        print('compl_limit exceed! img_size =', h*w, '>', compl_limit, '= compl_limit')

    if result is None: # exceed compl_limit or OOM
        if h > w:
            upper = inpaint(img[:h//2,:], mask[:h//2,:], complnet, kernel) 
            downer= inpaint(img[h//2:,:], mask[h//2:,:], complnet, kernel)
            result = np.concatenate((upper,downer), axis=0)
            assert img.shape == result.shape,\
                'img.{} != result.{} in up + down'.format(img.shape,result.shape)
        else:
            left = inpaint(img[:,:w//2], mask[:,:w//2], complnet, kernel)
            right= inpaint(img[:,w//2:], mask[:,w//2:], complnet, kernel)
            result = np.concatenate((left,right), axis=1)
            assert img.shape == result.shape,\
                'img.{} != result.{} in left + right'.format(img.shape,result.shape)
    assert img.shape == result.shape,\
        'img.{} != result.{} in merged'.format(img.shape,result.shape)
    return result # image inpainted successfully!

import fp
from futils import human_sorted,file_pathseq
if __name__ == "__main__":
    #ng.get_gpus(1,False)
    args = parser.parse_args()

    args.imgdir = './dset4paper/tmp/imgs/'
    args.maskdir= './dset4paper/tmp/masks/'
    args.outdir = './dset4paper/tmp/results'
    args.checkpoint_dir = './model_logs/hasT_350k/'

    img_paths = fp.pipe(file_pathseq,human_sorted)(args.imgdir)
    mask_paths = fp.pipe(file_pathseq,human_sorted)(args.maskdir)

    # build inference model
    # loading_time: 2.5587990283966064 sec
    start = time.time() #------------------------------
    '''
    sess_config = tf.ConfigProto()                                           
    sess_config.gpu_options.allow_growth = True                              
    sess = tf.Session(config=sess_config)                                    
                                                                             
    model = InpaintCAModel()                                                 
    input_image_ph = tf.placeholder(                                         
        tf.float32, shape = (1,None,None,3), name = 'INPUT'
    )                                                                    
    output = model.build_server_graph(input_image_ph)                        
    output = (output + 1.) * 127.5                                           
    output = tf.reverse(output, [-1])                                        
    output = tf.saturate_cast(output, tf.uint8, name='OUTPUT')                              

    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)             
    assign_ops = []                                                          
    for var in vars_list:                                                    
        vname = var.name                                                     
        from_name = vname                                                    
        var_value = tf.contrib.framework.load_variable(                      
            args.checkpoint_dir, from_name
        )                                                                
        assign_ops.append(tf.assign(var, var_value))                         
    sess.run(assign_ops)                                                     
 
    writer = tf.summary.FileWriter('./tmplog')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()

    cnet = make_complnet(input_image_ph, output)
    cnet = lambda input_image: sess.run(
        output, feed_dict={input_image_ph: input_image}
    )
    '''
    # load inference model 
    # loading_time: 0.5577731132507324 sec
    model_path = './dset4paper/cnet.pb'

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    sess = tf.Session()
    cnet_in  = sess.graph.get_tensor_by_name('INPUT:0')
    cnet_out = sess.graph.get_tensor_by_name('OUTPUT:0')
    cnet = lambda input_image: sess.run(
        cnet_out, feed_dict={cnet_in:input_image}
    )
    end = time.time()  #------------------------------
    print('loading_time:', end - start, 'sec')

    def mk_outpath(srcpath):
        return str(
            Path(args.outdir) / Path(srcpath).parts[-1]
        )
    out_paths = fp.lmap(mk_outpath, img_paths)

    def inpainted_time(image, mask):
        start = time.time() #------------------------------
        result = inpaint(image, mask, cnet)
        end = time.time()  #------------------------------
        print('running_time:', end - start)
        return result, end - start

    def tap_img(x,y):
        cv2.imshow('origin',x)
        cv2.imshow('mask',y)
        cv2.waitKey(0)

    def tap_img_time(im,t):
        cv2.imshow('result',im)
        cv2.waitKey(0)
        print(t)

    img_timeseq = fp.pipe(
        fp.cmap(fp.clmap( cv2.imread )),
        fp.cmap(fp.tup( inpainted_time )),
        #fp.cmap(fp.tup( tap_img )),
        #fp.cmap(fp.tup( tap_img_time )),
    )( zip(img_paths,mask_paths) )

    runtimes = []
    for (img,runtime), dstpath in tqdm( zip(img_timeseq,out_paths),
                                        total=len(out_paths)):
        cv2.imwrite(dstpath, img)
        runtimes.append(runtime)
    print(runtimes)
    print(np.mean(runtimes),'sec')
