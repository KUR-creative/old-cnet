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
#parser.add_argument('--image', default='', type=str,
                    #help='The filename of image to be completed.')
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

def inpaint_or_oom(img, segmap, complnet, complnet_ckpt_dir, 
                   dilate_kernel=None):
    ''' If image is too big, return None '''
    image = img.copy()
    mask = segmap.copy()
    mask = binarization(mask, 0.5)

    assert image.shape == mask.shape 
    #print('origin',image.shape)

    org_h, org_w, _ = image.shape
    # TODO:pad

    modulo = 8
    image = modulo_padded(image,8)
    mask = modulo_padded(mask,8)
    #print('Shape of image: {}'.format(image.shape))
    #print('Shape of image: {}'.format(mask.shape))

    image = np.expand_dims(image, 0) # [h,w,c] -> [1,h,w,c]
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32) #const
        output = complnet.build_server_graph(input_image,reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)  # maybe output of entire network?
        # load pretrained complnet
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(complnet_ckpt_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)

        writer = tf.summary.FileWriter('./tmplog')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()
        exit()
        #print('Model loaded.')
        try:
            result = sess.run(output)
            #print('wtf origin',image.shape)
            #print('wtf result',result.shape)
            assert image.shape == result.shape, \
                '{} != {}'.format(image.shape, result.shape)
            return result[0][:org_h, :org_w, ::-1] #---------- remove padding
        except Exception as e: # ResourceExhaustedError:
            logging.error(traceback.format_exc())
            print((org_h,org_w), 'OOM error in inpainting')
            return None

compl_limit = 657666 #lab-machine #1525920
def inpaint(img, mask, complnet, complnet_ckpt_dir, dilate_kernel=None):
    ''' oom-free inpainting '''
    global compl_limit

    cnet_dir = complnet_ckpt_dir
    kernel = dilate_kernel

    h,w = img.shape[:2]
    result = None
    if h*w < compl_limit:
        result = inpaint_or_oom(img, mask, complnet, cnet_dir, dilate_kernel=kernel)
        if result is None: # compl_limit: Ok but OOM occur!
            compl_limit = h*w
            #print('compl_limit =', compl_limit, 'updated!')
        assert img.shape == result.shape,\
            'img.{} != result.{} in no limit'.format(img.shape,result.shape)
    else:
        pass
        print('compl_limit exceed! img_size =', h*w, '>', compl_limit, '= compl_limit')

    if result is None: # exceed compl_limit or OOM
        #print('----->',h,w)
        if h > w:
            upper = inpaint(img[:h//2,:], mask[:h//2,:], complnet, cnet_dir, kernel) 
            downer= inpaint(img[h//2:,:], mask[h//2:,:], complnet, cnet_dir, kernel)
            result = np.concatenate((upper,downer), axis=0)
            #print('u',upper.shape, 'd',downer.shape, 'r',result.shape, '+',np.array(upper.shape)+np.array(downer.shape))
            assert img.shape == result.shape,\
                'img.{} != result.{} in up + down'.format(img.shape,result.shape)
        else:
            left = inpaint(img[:,:w//2], mask[:,:w//2], complnet, cnet_dir, kernel)
            right= inpaint(img[:,w//2:], mask[:,w//2:], complnet, cnet_dir, kernel)
            result = np.concatenate((left,right), axis=1)
            #print('l',left.shape, 'r',right.shape, 'r',result.shape, '+',np.array(left.shape)+np.array(right.shape))
            assert img.shape == result.shape,\
                'img.{} != result.{} in left + right'.format(img.shape,result.shape)
    assert img.shape == result.shape,\
        'img.{} != result.{} in merged'.format(img.shape,result.shape)
    return result # image inpainted successfully!


import fp
from futils import human_sorted,file_pathseq
if __name__ == "__main__":
    ng.get_gpus(1,False)
    args = parser.parse_args()

    model = InpaintCAModel()

    '''
    args.imgdir = './dset4paper/test/imgs/'
    args.maskdir= './dset4paper/test/masks/'
    args.outdir = './dset4paper/test/results'
    args.checkpoint_dir = './model_logs/hasT_350k/'
    '''

    img_paths = fp.pipe(file_pathseq,human_sorted)(args.imgdir)
    mask_paths = fp.pipe(file_pathseq,human_sorted)(args.maskdir)

    print('wtf i',img_paths)
    print('wtf m',mask_paths)
    def mk_outpath(srcpath):
        return str(
            Path(args.outdir) / Path(srcpath).parts[-1]
        )
    out_paths = fp.lmap(mk_outpath, img_paths)

    def inpainted_time(image, mask):
        start = time.time() #------------------------------
        result = inpaint(image, mask, model, args.checkpoint_dir)
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
    '''
    cv2.imwrite(args.output, result)
    cv2.imshow('result',result)
    cv2.waitKey(0)

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        start = time.time() #------------------------------

        sess.run(assign_ops)
        result = sess.run(output)
        #print('Model loaded.')

        end = time.time()  #------------------------------
        print('RunningTime:', end - start)

        cv2.imwrite(args.output, result[0][:, :, ::-1])
        #print(result)
        #print(type(result))
        #print(result.shape)
        cv2.imshow('result',result[0][:, :, ::-1])
        cv2.waitKey(0)
    '''

    #TODO: Use This! from issue in author's repository.
    # https://github.com/JiahuiYu/generative_inpainting/issues/12
    '''
    sess_config = tf.ConfigProto()                                           
    sess_config.gpu_options.allow_growth = True                              
    sess = tf.Session(config=sess_config)                                    
                                                                             
    model = InpaintCAModel()                                                 
    input_image_ph = tf.placeholder(                                         
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))     
    output = model.build_server_graph(input_image_ph)                        
    output = (output + 1.) * 127.5                                           
    output = tf.reverse(output, [-1])                                        
    output = tf.saturate_cast(output, tf.uint8)                              
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)             
    assign_ops = []                                                          
    for var in vars_list:                                                    
        vname = var.name                                                     
        from_name = vname                                                    
        var_value = tf.contrib.framework.load_variable(                      
            args.checkpoint_dir, from_name)                                  
        assign_ops.append(tf.assign(var, var_value))                         
    sess.run(assign_ops)                                                     
    print('Model loaded.')                                                   
                                                                             
    with open(args.flist, 'r') as f:                                         
        lines = f.read().splitlines()                                        
    t = time.time()                                                          
    for line in lines:                                                   
        image, mask, out = line.split()                                      
        base = os.path.basename(mask)                                        
                                                                             
        image = cv2.imread(image)                                            
        mask = cv2.imread(mask)                                              
        image = cv2.resize(image, (args.image_width, args.image_height))     
        mask = cv2.resize(mask, (args.image_width, args.image_height))       
        # cv2.imwrite(out, image*(1-mask/255.) + mask)                       
        # # continue                                                         
        # image = np.zeros((128, 256, 3))                                    
        # mask = np.zeros((128, 256, 3))                                     
                                                                             
        assert image.shape == mask.shape                                     
                                                                             
        h, w, _ = image.shape                                                
        grid = 4                                                             
        image = image[:h//grid*grid, :w//grid*grid, :]                       
        mask = mask[:h//grid*grid, :w//grid*grid, :]                         
        print('Shape of image: {}'.format(image.shape))                      
                                                                             
        image = np.expand_dims(image, 0)                                     
        mask = np.expand_dims(mask, 0)                                       
        input_image = np.concatenate([image, mask], axis=2)                  
                                                                             
        # load pretrained model                                              
        result = sess.run(output, feed_dict={input_image_ph: input_image})   
        print('Processed: {}'.format(out))                                   
        cv2.imwrite(out, result[0][:, :, ::-1])                              
                                                                             
    print('Time total: {}'.format(time.time() - t)) 
    '''

    #TODO: and None,None sized input image
    # https://github.com/JiahuiYu/generative_inpainting/issues/194
    '''
    '''

